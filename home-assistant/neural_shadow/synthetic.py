"""Physically grounded 3-unit, 2-zone synthetic generator strictly for offline pipeline verification.

IMPORTANT: Synthetic data generated here is ALWAYS stamped
data_origin="synthetic_verification_only". It verifies that code, data alignment,
training, sealing, settlement, and tests run properly. It MUST NEVER be claimed
as real-house predictive accuracy or measured energy savings.
"""
from __future__ import annotations

import math
import numpy as np

from . import DataOrigin
from .config import SiteConfig, load_default_example_config
from .data_pipeline import (
    ActionEvent,
    CanonicalHistory,
    PowerReport,
    ProxyStateEvent,
    TempReport,
    _finalize_canonical,
)


def generate_synthetic_history(
    site: SiteConfig | None = None,
    *,
    hours: float = 96.0,
    seed: int = 42,
    start_ts: float = 1_795_000_000.0,
    include_power_gaps: bool = True,
    include_duplicate_cache: bool = True,
    include_manual_and_failed: bool = True,
) -> CanonicalHistory:
    """Simulate a coupled 2-independent-zone, 3-unit house with realistic thermal inertia and inverter power."""
    site = site or load_default_example_config()
    rng = np.random.default_rng(seed)

    z_names = list(site.zone_names)
    u_names = list(site.unit_names)
    n_z = len(z_names)
    n_u = len(u_names)
    inf_mat = np.asarray(site.influence_matrix(), dtype=np.float64)  # [Z, U]
    cpl_mat = np.asarray(site.coupling_matrix(), dtype=np.float64)  # [Z, Z]

    total_minutes = int(round(hours * 60))
    temps = np.full(n_z, 23.8, dtype=np.float64)
    targets = np.full(n_z, 24.0, dtype=np.float64)
    modes: list[str] = ["off"] * n_u
    setpoints = np.full(n_u, 24.0, dtype=np.float64)
    compressor_power = np.full(n_u, 2.0, dtype=np.float64)

    raw_temp: list[TempReport] = []
    raw_env: list[TempReport] = []
    raw_power: list[PowerReport] = []
    proxy_by_unit: dict[str, list[ProxyStateEvent]] = {u: [] for u in u_names}
    actions: list[ActionEvent] = []
    quality_counts: dict[str, int] = {}

    last_temp_sent = np.full(n_z, -1e9, dtype=np.float64)
    last_temp_val = temps.copy()
    last_action_min = np.full(n_u, -60, dtype=np.int64)
    epoch_str = "2026-11-25T00:00:00+00:00"

    # Simulate at 1-minute resolution; emit realistic sensor cadence (~5m temp reports, ~1m power reports).
    for m in range(total_minutes + 1):
        t = start_ts + m * 60.0
        # Outdoor temperature swings smoothly between cool nights (~12C) and warm afternoons (~30C)
        day_phase = 2.0 * math.pi * (m / 1440.0)
        slow_weather = 3.5 * math.sin(2.0 * math.pi * (m / max(1440.0, total_minutes)))
        outdoor = 21.0 + 7.5 * math.sin(day_phase - 1.2) + slow_weather + float(rng.normal(0.0, 0.1))

        # Simple exploratory thermostat + occasional perturbations so training sees both heat & cool & ±1C steps
        for u_idx, u in enumerate(u_names):
            act_map = site.actuator_by_name(u)
            hz_idx = z_names.index(act_map.home_zone)
            err = temps[hz_idx] - targets[hz_idx]
            if m - last_action_min[u_idx] >= 25 and rng.random() < 0.14:
                old_mode = modes[u_idx]
                old_sp = float(setpoints[u_idx])
                new_mode = old_mode
                new_sp = old_sp
                kind = "temperature"
                purpose = "synthetic_exploration"

                if old_mode == "off":
                    if err < -0.6 or (outdoor < 17.5 and rng.random() < 0.25):
                        new_mode = "heat"
                        new_sp = float(rng.choice([22.0, 23.0, 24.0]))
                        kind = "mode"
                    elif err > 0.6 or (outdoor > 25.5 and rng.random() < 0.25):
                        new_mode = "cool"
                        new_sp = float(rng.choice([23.0, 24.0, 25.0]))
                        kind = "mode"
                else:
                    # Prefer same-mode ±1C adjustments; occasionally stop when comfortably reversed
                    low_b, high_b = act_map.soft_limits[old_mode]
                    if (old_mode == "heat" and err > 0.9) or (old_mode == "cool" and err < -0.9):
                        if rng.random() < 0.35:
                            new_mode = "off"
                            kind = "mode"
                        else:
                            step = -1.0 if old_mode == "heat" else 1.0
                            new_sp = float(np.clip(old_sp + step, low_b, high_b))
                    else:
                        if old_mode == "heat":
                            step = 1.0 if err < -0.3 else (-1.0 if err > 0.3 else float(rng.choice([-1.0, 1.0])))
                        else:
                            step = -1.0 if err > 0.3 else (1.0 if err < -0.3 else float(rng.choice([-1.0, 1.0])))
                        new_sp = float(np.clip(old_sp + step, low_b, high_b))

                if new_mode != old_mode or abs(new_sp - old_sp) > 1e-4:
                    is_failed = include_manual_and_failed and rng.random() < 0.04
                    is_manual = include_manual_and_failed and (not is_failed) and rng.random() < 0.06
                    status = "unconfirmed" if is_failed else ("manual_observed" if is_manual else "proxy_echo")
                    src = "manual_external" if is_manual else "controller"
                    flags = []
                    if is_failed:
                        flags.append("failed_call")
                    if is_manual:
                        flags.append("manual_operation")
                    if status == "proxy_echo":
                        flags.append("proxy_only_unverified")
                    for fl in flags:
                        quality_counts[fl] = quality_counts.get(fl, 0) + 1

                    actions.append(
                        ActionEvent(
                            at=t + 3.0,
                            sent_at=t,
                            unit=u,
                            kind="manual_change" if is_manual else kind,
                            before_mode=old_mode,
                            before_setpoint=old_sp,
                            after_mode=new_mode,
                            after_setpoint=new_sp,
                            source=src,
                            receipt_status=status,
                            physical_verified=False,
                            purpose=purpose,
                            target_at_send=float(targets[hz_idx]),
                            quality_flags=tuple(flags),
                        )
                    )
                    if not is_failed:
                        modes[u_idx] = new_mode
                        setpoints[u_idx] = new_sp
                    last_action_min[u_idx] = m

        # Thermal & electrical physics step (1 minute)
        # Keep setpoint, electrical watts, and thermal output strictly distinct!
        unit_q = np.zeros(n_u, dtype=np.float64)
        for u_idx, u in enumerate(u_names):
            act_map = site.actuator_by_name(u)
            hz_idx = z_names.index(act_map.home_zone)
            mode = modes[u_idx]
            sp = setpoints[u_idx]
            if mode == "heat":
                drive = max(0.0, (sp + 0.6) - temps[hz_idx])
                target_w = 18.0 + min(950.0, 210.0 * drive + 45.0 * max(0.0, 20.0 - outdoor) / 10.0)
                cop = max(2.2, 3.4 - 0.04 * max(0.0, 15.0 - outdoor))
                # First-order compressor ramp inertia
                compressor_power[u_idx] = 0.72 * compressor_power[u_idx] + 0.28 * target_w
                # Thermal effect proportional to electrical power above standby * COP
                unit_q[u_idx] = max(0.0, compressor_power[u_idx] - 15.0) * cop * 0.000042
            elif mode == "cool":
                drive = max(0.0, temps[hz_idx] - (sp - 0.6))
                target_w = 18.0 + min(980.0, 220.0 * drive + 40.0 * max(0.0, outdoor - 24.0) / 10.0)
                cop = max(2.4, 3.6 - 0.04 * max(0.0, outdoor - 28.0))
                compressor_power[u_idx] = 0.72 * compressor_power[u_idx] + 0.28 * target_w
                unit_q[u_idx] = -max(0.0, compressor_power[u_idx] - 15.0) * cop * 0.000044
            else:
                compressor_power[u_idx] = 0.55 * compressor_power[u_idx] + 0.45 * 2.0
                unit_q[u_idx] = 0.0

        # Coupled zone temperature update
        new_temps = temps.copy()
        for z_idx in range(n_z):
            ambient_leak = 0.0045 * (outdoor - temps[z_idx])
            couple_flux = sum(cpl_mat[z_idx, j] * 0.012 * (temps[j] - temps[z_idx]) for j in range(n_z) if j != z_idx)
            hvac_flux = float(np.dot(inf_mat[z_idx], unit_q))
            noise = float(rng.normal(0.0, 0.012))
            new_temps[z_idx] = float(np.clip(temps[z_idx] + ambient_leak + couple_flux + hvac_flux + noise, 12.0, 35.0))
        temps = new_temps

        # Emit outdoor report every 10 minutes
        if m % 10 == 0:
            for es in site.active_environment_sensors:
                raw_env.append(
                    TempReport(
                        zone=es.name,
                        role="environment",
                        source_entity=es.entity or es.name,
                        report_at=t + 0.1,
                        value_c=round(outdoor, 2),
                        observed_at=t + 0.5,
                        quality="fresh",
                        is_exact_precision=True,
                    )
                )

        # Emit independent zone temperature reports every 5 minutes (plus occasional same-value new reports)
        for z_idx, z in enumerate(z_names):
            zmap = site.zone_by_name(z)
            quantized = round(float(temps[z_idx]), 1)
            emit_new = (m % 5 == 0) or (abs(quantized - last_temp_val[z_idx]) >= 0.2 and (t - last_temp_sent[z_idx]) >= 180)
            if emit_new:
                rep_ts = t + 0.25 + z_idx * 0.1
                raw_temp.append(
                    TempReport(
                        zone=z,
                        role="zone_primary",
                        source_entity=zmap.primary_entity,
                        report_at=rep_ts,
                        value_c=quantized,
                        observed_at=rep_ts + 1.0,
                        quality="fresh",
                        is_exact_precision=True,
                    )
                )
                last_temp_sent[z_idx] = rep_ts
                last_temp_val[z_idx] = quantized
            elif include_duplicate_cache and last_temp_sent[z_idx] > 0:
                # Repeated minute-level cache copy carrying the OLD report_at (must be deduplicated!)
                raw_temp.append(
                    TempReport(
                        zone=z,
                        role="zone_primary",
                        source_entity=zmap.primary_entity,
                        report_at=float(int(last_temp_sent[z_idx])),
                        value_c=float(last_temp_val[z_idx]),
                        observed_at=t,
                        quality="observing",
                        is_exact_precision=False,
                    )
                )

        # Emit per-unit power reports (1-minute cadence, with optional deliberate gap window)
        in_gap = include_power_gaps and (total_minutes // 3 <= m <= total_minutes // 3 + 12)
        for u_idx, u in enumerate(u_names):
            act_map = site.actuator_by_name(u)
            if not (in_gap and u_idx == 0):
                p_val = round(max(1.0, float(compressor_power[u_idx] + rng.normal(0.0, 2.0))), 1)
                raw_power.append(
                    PowerReport(
                        unit=u,
                        source_entity=act_map.power_entity,
                        observer_epoch=epoch_str,
                        report_at=t + 15.0 + u_idx,
                        watts=p_val,
                    )
                )
            hz_idx = z_names.index(act_map.home_zone)
            proxy_by_unit[u].append(
                ProxyStateEvent(
                    unit=u,
                    at=t,
                    mode=modes[u_idx],
                    setpoint=float(setpoints[u_idx]),
                    target_c=float(targets[hz_idx]),
                    source_fresh=True,
                )
            )

    return _finalize_canonical(
        site=site,
        data_origin=DataOrigin.SYNTHETIC.value,
        source_description=f"synthetic_simulator:hours={hours}:seed={seed}",
        raw_temp=raw_temp,
        raw_env=raw_env,
        raw_power=raw_power,
        proxy_by_unit=proxy_by_unit,
        actions=actions,
        config_versions=[{"at": start_ts, "commit": "synthetic", "observation_version": site.observation_version}],
        quality_counts=quality_counts,
    )
