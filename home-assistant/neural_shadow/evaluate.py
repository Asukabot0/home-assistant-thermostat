"""Offline evaluation: per-zone, per-horizon, slices, Wh/coverage, baselines, and action response."""
from __future__ import annotations

from typing import Any

import numpy as np

from .baselines import ExistingTrendBaseline, PersistenceBaseline, ThermalInertiaLinearBaseline
from .data_pipeline import (
    BuiltDataset,
    build_action_sequence_tensor,
    build_step_action_plan,
)
from .nn_model import ModelArtifact, predict_numpy

Z_80 = 1.2815515655446004


def _scalar_error_stats(
    pred: np.ndarray,
    actual: np.ndarray,
    sigma: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    mask = valid & np.isfinite(pred) & np.isfinite(actual) & np.isfinite(sigma)
    n = int(mask.sum())
    if n == 0:
        return {
            "n_valid": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "p90_abs_err": None,
            "p95_abs_err": None,
            "max_abs_err": None,
            "pi80_coverage": None,
            "pi80_mean_width": None,
        }
    p = pred[mask].astype(np.float64)
    a = actual[mask].astype(np.float64)
    s = np.maximum(sigma[mask].astype(np.float64), 1e-6)
    err = p - a
    abs_err = np.abs(err)
    half_width = Z_80 * s
    covered = abs_err <= half_width + 1e-9
    return {
        "n_valid": n,
        "mae": round(float(np.mean(abs_err)), 4),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "bias": round(float(np.mean(err)), 4),
        "p90_abs_err": round(float(np.quantile(abs_err, 0.90)), 4),
        "p95_abs_err": round(float(np.quantile(abs_err, 0.95)), 4),
        "max_abs_err": round(float(np.max(abs_err)), 4),
        "pi80_coverage": round(float(np.mean(covered)), 4),
        "pi80_mean_width": round(float(np.mean(2.0 * half_width)), 4),
    }


def _slice_masks(dataset: BuiltDataset, indices: np.ndarray) -> dict[str, np.ndarray]:
    n = len(indices)
    masks = {
        "all": np.ones(n, dtype=bool),
        "mode_heating": np.zeros(n, dtype=bool),
        "mode_cooling": np.zeros(n, dtype=bool),
        "mode_off": np.zeros(n, dtype=bool),
        "post_action": np.zeros(n, dtype=bool),
        "over_cool": np.zeros(n, dtype=bool),
        "over_heat": np.zeros(n, dtype=bool),
    }
    for row_i, idx in enumerate(indices):
        meta = dataset.metadata[int(idx)]
        modes = set(meta["unit_modes"].values())
        if "heat" in modes:
            masks["mode_heating"][row_i] = True
        if "cool" in modes:
            masks["mode_cooling"][row_i] = True
        if all(m in {"off", None} for m in meta["unit_modes"].values()):
            masks["mode_off"][row_i] = True
        if meta.get("post_action_15m"):
            masks["post_action"][row_i] = True
        for z in dataset.spec.zone_names:
            t_c = meta["zone_temps"].get(z)
            tgt = meta["zone_targets"].get(z)
            if t_c is None or tgt is None:
                continue
            err = t_c - tgt
            if err < -0.3 and ("cool" in modes or err < -0.5):
                masks["over_cool"][row_i] = True
            if err > 0.3 and ("heat" in modes or err > 0.5):
                masks["over_heat"][row_i] = True
    return masks


def compute_prediction_metrics(
    dataset: BuiltDataset,
    indices: np.ndarray,
    preds: dict[str, np.ndarray],
    *,
    mask_intervened: bool = False,
) -> dict[str, Any]:
    """Compute comprehensive per-zone, per-horizon, Wh/coverage, and operating-slice metrics."""
    spec = dataset.spec
    horizons = spec.horizons_minutes
    n = len(indices)

    y_td = dataset.y_temp_delta[indices]
    y_tv = dataset.y_temp_valid[indices].copy()
    y_uwh = dataset.y_unit_wh[indices]
    y_ucov = dataset.y_unit_cov[indices]
    y_uval = dataset.y_unit_valid[indices].copy()
    y_twh = dataset.y_total_wh[indices]
    y_tcov = dataset.y_total_cov[indices]
    y_tval = dataset.y_total_valid[indices].copy()
    interv = dataset.intervened[indices]

    intervened_excluded: dict[str, int] = {}
    if mask_intervened and n > 0:
        for k, h in enumerate(horizons):
            intervened_excluded[str(h)] = int(interv[:, k].sum())
            y_tv[:, :, k] &= ~interv[:, k : k + 1]
            y_uval[:, :, k] &= ~interv[:, k : k + 1]
            y_tval[:, k] &= ~interv[:, k]
    else:
        for k, h in enumerate(horizons):
            intervened_excluded[str(h)] = 0

    # Per-zone, per-horizon temperature metrics
    temp_by_zone: dict[str, dict[str, Any]] = {}
    for z_idx, zname in enumerate(spec.zone_names):
        temp_by_zone[zname] = {}
        for k, h in enumerate(horizons):
            temp_by_zone[zname][f"{h}m"] = _scalar_error_stats(
                preds["temp_delta"][:, z_idx, k],
                y_td[:, z_idx, k],
                preds["temp_sigma"][:, z_idx, k],
                y_tv[:, z_idx, k],
            )

    # Aggregate temperature per horizon across independent zones
    temp_by_horizon: dict[str, Any] = {}
    for k, h in enumerate(horizons):
        temp_by_horizon[f"{h}m"] = _scalar_error_stats(
            preds["temp_delta"][:, :, k].reshape(-1),
            y_td[:, :, k].reshape(-1),
            preds["temp_sigma"][:, :, k].reshape(-1),
            y_tv[:, :, k].reshape(-1),
        )

    # Energy per unit & total Wh + metering coverage
    wh_by_unit: dict[str, dict[str, Any]] = {}
    for u_idx, uname in enumerate(spec.unit_names):
        wh_by_unit[uname] = {}
        for k, h in enumerate(horizons):
            stats = _scalar_error_stats(
                preds["unit_wh"][:, u_idx, k],
                y_uwh[:, u_idx, k],
                preds["unit_wh_sigma"][:, u_idx, k],
                y_uval[:, u_idx, k],
            )
            cov = y_ucov[:, u_idx, k] if n > 0 else np.zeros((0,))
            stats["metering_coverage_mean"] = round(float(cov.mean()), 4) if cov.size else None
            stats["metering_gap_fraction"] = round(float((cov < 1.0 - 1e-6).mean()), 4) if cov.size else None
            stats["uncovered_excluded_count"] = int((~y_uval[:, u_idx, k]).sum())
            wh_by_unit[uname][f"{h}m"] = stats

    wh_total: dict[str, Any] = {}
    for k, h in enumerate(horizons):
        stats = _scalar_error_stats(
            preds["total_wh"][:, k],
            y_twh[:, k],
            preds["total_wh_sigma"][:, k],
            y_tval[:, k],
        )
        cov = y_tcov[:, k] if n > 0 else np.zeros((0,))
        stats["joint_metering_coverage_mean"] = round(float(cov.mean()), 4) if cov.size else None
        stats["joint_metering_gap_fraction"] = round(float((cov < 1.0 - 1e-6).mean()), 4) if cov.size else None
        stats["uncovered_excluded_count"] = int((~y_tval[:, k]).sum())
        wh_total[f"{h}m"] = stats

    # Operating slices
    slices: dict[str, Any] = {}
    for sname, smask in _slice_masks(dataset, indices).items():
        s_count = int(smask.sum())
        if s_count == 0:
            slices[sname] = {"n_samples": 0, "temp_by_horizon": {}, "total_wh_by_horizon": {}}
            continue
        s_temp = {}
        s_wh = {}
        for k, h in enumerate(horizons):
            s_temp[f"{h}m"] = _scalar_error_stats(
                preds["temp_delta"][smask, :, k].reshape(-1),
                y_td[smask, :, k].reshape(-1),
                preds["temp_sigma"][smask, :, k].reshape(-1),
                y_tv[smask, :, k].reshape(-1),
            )
            s_wh[f"{h}m"] = _scalar_error_stats(
                preds["total_wh"][smask, k],
                y_twh[smask, k],
                preds["total_wh_sigma"][smask, k],
                y_tval[smask, k],
            )
        slices[sname] = {
            "n_samples": s_count,
            "temp_by_horizon": s_temp,
            "total_wh_by_horizon": s_wh,
        }

    # Physical output invariants check
    wh_nonneg = bool(np.all(preds["unit_wh"] >= -1e-6) and np.all(preds["total_wh"] >= -1e-6)) if n > 0 else True
    wh_monotone = (
        bool(
            np.all(np.diff(preds["unit_wh"], axis=-1) >= -1e-5)
            and np.all(np.diff(preds["total_wh"], axis=-1) >= -1e-5)
        )
        if n > 0 and len(horizons) > 1
        else True
    )
    wh_sum_consistent = (
        bool(np.allclose(preds["total_wh"], preds["unit_wh"].sum(axis=1), atol=1e-3)) if n > 0 else True
    )

    return {
        "n_samples": n,
        "mask_intervened": mask_intervened,
        "intervened_excluded_by_horizon": intervened_excluded,
        "temp_by_zone": temp_by_zone,
        "temp_by_horizon": temp_by_horizon,
        "wh_by_unit": wh_by_unit,
        "wh_total": wh_total,
        "slices": slices,
        "invariants": {
            "wh_non_negative": wh_nonneg,
            "wh_monotone_non_decreasing": wh_monotone,
            "total_wh_equals_unit_sum": wh_sum_consistent,
        },
    }


def evaluate_action_response(
    artifact: ModelArtifact,
    dataset: BuiltDataset,
    eval_indices: np.ndarray,
    *,
    min_temp_response_c: float = 0.03,
    min_action_events: int = 30,
    min_active_mode_samples: int = 60,
    min_unit_mode_samples: int = 10,
) -> dict[str, Any]:
    """Verify the neural model responds in the physically correct direction across horizons.

    Strict checks:
    - Evaluates each unit x observed mode separately (requiring 100% pass across observed pairs).
    - Probes natural evaluation samples in that mode whenever present (rather than eval[:48]).
    - Checks V1 actual candidates (same-mode +/-1C vs hold) across ALL horizons (15/30/60m) for
      both temperature sign and non-decreasing electrical Wh.
    - Checks active-vs-off temperature and Wh signs across ALL horizons so inverted responses
      (e.g. cooling hotter than off, heating colder than off, active using less Wh than off)
      always fail.
    """
    spec = dataset.spec
    site = dataset.site
    manifest = dataset.manifest
    horizons = spec.horizons_minutes
    n_k = len(horizons)
    max_h = max(horizons)

    # 1. Historical action coverage audit
    total_actions = int(manifest.get("action_events_total", 0))
    mode_counts = manifest.get("mode_sample_counts", {})
    heat_samples = int(mode_counts.get("any_heat", 0))
    cool_samples = int(mode_counts.get("any_cool", 0))
    action_types = manifest.get("action_type_counts", {})
    sp_actions = sum(v for k, v in action_types.items() if k.startswith("setpoint_"))
    coverage_issues: list[str] = []
    if total_actions < min_action_events:
        coverage_issues.append(f"insufficient_total_actions:{total_actions}<{min_action_events}")
    if sp_actions < max(6, min_action_events // 4):
        coverage_issues.append(f"insufficient_setpoint_actions:{sp_actions}")
    if max(heat_samples, cool_samples) < min_active_mode_samples:
        coverage_issues.append(f"insufficient_active_mode_samples:heat={heat_samples},cool={cool_samples}")
    action_coverage_sufficient = len(coverage_issues) == 0

    if eval_indices.size == 0:
        return {
            "action_coverage_sufficient": False,
            "action_coverage_issues": coverage_issues + ["empty_eval_split"],
            "action_response_passed": False,
            "directional_checks_passed": 0,
            "directional_checks_total": 0,
            "failed_probes": ["empty_eval_split"],
            "probes": {},
            "note": "No evaluation samples available for action-response check.",
        }

    def _select_evenly(indices: np.ndarray, limit: int = 48) -> np.ndarray:
        if indices.size <= limit:
            return indices
        pos = np.linspace(0, indices.size - 1, limit, dtype=np.int64)
        return indices[pos]

    probe_results: dict[str, Any] = {}
    directional_checks_passed = 0
    directional_checks_total = 0
    failed_probes: list[str] = []

    for u_idx, uname in enumerate(spec.unit_names):
        act_map = site.actuator_by_name(uname)
        z_idx = spec.zone_names.index(act_map.home_zone)
        dataset_off_count = sum(1 for m in dataset.metadata if m["unit_modes"].get(uname) == "off")
        unit_probes: dict[str, Any] = {}
        for mode in ("heat", "cool"):
            dataset_mode_count = sum(1 for m in dataset.metadata if m["unit_modes"].get(uname) == mode)
            natural_eval = np.asarray(
                [int(idx) for idx in eval_indices if dataset.metadata[int(idx)]["unit_modes"].get(uname) == mode],
                dtype=np.int64,
            )
            natural_count = int(natural_eval.size)
            mode_observed = dataset_mode_count >= min_unit_mode_samples
            used_natural = natural_count > 0
            probe_idx = _select_evenly(natural_eval if used_natural else eval_indices, limit=48)
            xh = dataset.X_hist[probe_idx]

            lo, hi = act_map.soft_limits[mode]
            default_sp = 23.0 if mode == "heat" else 24.0
            xa_hold = []
            xa_plus = []
            xa_minus = []
            xa_off = []
            for idx in probe_idx:
                meta = dataset.metadata[int(idx)]
                modes_base = dict(meta["unit_modes"])
                sps_base = dict(meta["unit_setpoints"])
                raw_sp = sps_base.get(uname) if modes_base.get(uname) == mode else None
                cand_sp = float(raw_sp) if raw_sp is not None else default_sp
                if hi - lo >= 2.0:
                    base_sp = float(np.clip(round(cand_sp), lo + 1.0, hi - 1.0))
                else:
                    base_sp = float(np.clip(round(cand_sp), lo, hi))
                sp_plus = min(hi, base_sp + 1.0)
                sp_minus = max(lo, base_sp - 1.0)
                modes_base[uname] = mode
                sps_base[uname] = base_sp

                plan_h = build_step_action_plan(spec, modes_base, sps_base, unit=uname, new_mode=mode, new_setpoint=base_sp)
                plan_p = build_step_action_plan(spec, modes_base, sps_base, unit=uname, new_mode=mode, new_setpoint=sp_plus)
                plan_m = build_step_action_plan(spec, modes_base, sps_base, unit=uname, new_mode=mode, new_setpoint=sp_minus)
                plan_o = build_step_action_plan(spec, modes_base, sps_base, unit=uname, new_mode="off", new_setpoint=base_sp)

                xa_hold.append(
                    build_action_sequence_tensor(spec, site, modes_base, sps_base, meta["zone_temps"], plan_h)
                )
                xa_plus.append(
                    build_action_sequence_tensor(spec, site, modes_base, sps_base, meta["zone_temps"], plan_p)
                )
                xa_minus.append(
                    build_action_sequence_tensor(spec, site, modes_base, sps_base, meta["zone_temps"], plan_m)
                )
                xa_off.append(
                    build_action_sequence_tensor(spec, site, modes_base, sps_base, meta["zone_temps"], plan_o)
                )

            pred_h = predict_numpy(artifact, xh, np.stack(xa_hold, axis=0))
            pred_p = predict_numpy(artifact, xh, np.stack(xa_plus, axis=0))
            pred_m = predict_numpy(artifact, xh, np.stack(xa_minus, axis=0))
            pred_o = predict_numpy(artifact, xh, np.stack(xa_off, axis=0))

            if mode == "heat":
                pred_stronger, pred_weaker = pred_p, pred_m
                expected_temp_sign = 1.0
            else:
                pred_stronger, pred_weaker = pred_m, pred_p
                expected_temp_sign = -1.0

            by_horizon: dict[str, Any] = {}
            probe_failures: list[str] = []
            for k, h_min in enumerate(horizons):
                h_key = f"{h_min}m"
                sw_temp = float(np.mean(pred_stronger["temp_delta"][:, z_idx, k] - pred_weaker["temp_delta"][:, z_idx, k]))
                sh_temp = float(np.mean(pred_stronger["temp_delta"][:, z_idx, k] - pred_h["temp_delta"][:, z_idx, k]))
                hw_temp = float(np.mean(pred_h["temp_delta"][:, z_idx, k] - pred_weaker["temp_delta"][:, z_idx, k]))
                sw_wh = float(np.mean(pred_stronger["unit_wh"][:, u_idx, k] - pred_weaker["unit_wh"][:, u_idx, k]))
                ao_temp = float(np.mean(pred_stronger["temp_delta"][:, z_idx, k] - pred_o["temp_delta"][:, z_idx, k]))
                ao_wh = float(np.mean(pred_stronger["unit_wh"][:, u_idx, k] - pred_o["unit_wh"][:, u_idx, k]))

                signed_sw = sw_temp * expected_temp_sign
                signed_sh = sh_temp * expected_temp_sign
                signed_hw = hw_temp * expected_temp_sign
                signed_ao = ao_temp * expected_temp_sign

                h_fails: list[str] = []
                # 1. V1 same-mode +/-1C temperature direction at EVERY horizon (no sign inversion or null response)
                if k == n_k - 1:
                    if signed_sw < min_temp_response_c:
                        h_fails.append(f"stronger_vs_weaker_temp_{h_key}:{sw_temp:+.4f}C")
                    if signed_sh < -1e-4:
                        h_fails.append(f"stronger_vs_hold_temp_inverted_{h_key}:{sh_temp:+.4f}C")
                    if signed_hw < -1e-4:
                        h_fails.append(f"hold_vs_weaker_temp_inverted_{h_key}:{hw_temp:+.4f}C")
                else:
                    min_h_temp = min_temp_response_c * 0.5 if h_min >= 30 and max_h > 30 else 0.0
                    if signed_sw < min_h_temp - 1e-6:
                        h_fails.append(f"stronger_vs_weaker_temp_{h_key}:{sw_temp:+.4f}C")
                    if signed_sh < -0.005:
                        h_fails.append(f"stronger_vs_hold_temp_inverted_{h_key}:{sh_temp:+.4f}C")
                    if signed_hw < -0.005:
                        h_fails.append(f"hold_vs_weaker_temp_inverted_{h_key}:{hw_temp:+.4f}C")

                # 2. V1 same-mode +/-1C electrical Wh must not decrease when strengthening output
                if sw_wh < -1e-3:
                    h_fails.append(f"stronger_vs_weaker_wh_negative_{h_key}:{sw_wh:+.2f}Wh")

                # 3. Active vs Off physical sanity: NEVER inverted temperature sign or negative Wh
                if signed_ao < -1e-4:
                    h_fails.append(f"active_vs_off_temp_inverted_{h_key}:{ao_temp:+.4f}C")
                if ao_wh < -1e-3:
                    h_fails.append(f"active_vs_off_wh_negative_{h_key}:{ao_wh:+.2f}Wh")
                if k == n_k - 1 and dataset_off_count >= min_unit_mode_samples:
                    if signed_ao < min_temp_response_c:
                        h_fails.append(f"active_vs_off_temp_weak_{h_key}:{ao_temp:+.4f}C")
                    if ao_wh <= 5.0:
                        h_fails.append(f"active_vs_off_wh_weak_{h_key}:{ao_wh:+.2f}Wh")

                probe_failures.extend(h_fails)
                by_horizon[h_key] = {
                    "stronger_minus_weaker_temp_c": round(sw_temp, 4),
                    "stronger_minus_hold_temp_c": round(sh_temp, 4),
                    "hold_minus_weaker_temp_c": round(hw_temp, 4),
                    "stronger_minus_weaker_unit_wh": round(sw_wh, 3),
                    "active_minus_off_temp_c": round(ao_temp, 4),
                    "active_minus_off_unit_wh": round(ao_wh, 3),
                    "passed": len(h_fails) == 0,
                    "failures": h_fails,
                }

            passed_dir = len(probe_failures) == 0
            if mode_observed:
                directional_checks_total += 1
                if passed_dir:
                    directional_checks_passed += 1
                else:
                    failed_probes.append(f"{uname}:{mode}")

            last_h = by_horizon[f"{horizons[-1]}m"]
            unit_probes[mode] = {
                "dataset_samples_in_mode": dataset_mode_count,
                "dataset_off_samples": dataset_off_count,
                "natural_eval_samples_in_mode": natural_count,
                "probe_sample_count": int(probe_idx.size),
                "used_natural_mode_samples": used_natural,
                "mode_observed_in_dataset": mode_observed,
                "stronger_minus_weaker_temp_60m_c": last_h["stronger_minus_weaker_temp_c"],
                "stronger_minus_weaker_unit_wh_60m": last_h["stronger_minus_weaker_unit_wh"],
                "active_minus_off_temp_60m_c": last_h["active_minus_off_temp_c"],
                "active_minus_off_unit_wh_60m": last_h["active_minus_off_unit_wh"],
                "expected_temp_sign": expected_temp_sign,
                "by_horizon": by_horizon,
                "directional_check_passed": bool(passed_dir),
                "failure_reasons": probe_failures,
            }
        probe_results[uname] = unit_probes

    action_response_passed = bool(
        directional_checks_total > 0 and directional_checks_passed == directional_checks_total
    )
    return {
        "action_coverage_sufficient": action_coverage_sufficient,
        "action_coverage_issues": coverage_issues,
        "action_events_total": total_actions,
        "setpoint_action_events": sp_actions,
        "mode_sample_counts": mode_counts,
        "action_response_passed": action_response_passed,
        "directional_checks_passed": directional_checks_passed,
        "directional_checks_total": directional_checks_total,
        "failed_probes": failed_probes,
        "probes": probe_results,
    }


def evaluate_all_models(
    artifact: ModelArtifact,
    dataset: BuiltDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    min_temp_response_c: float = 0.03,
    min_action_events: int = 30,
    min_active_mode_samples: int = 60,
) -> dict[str, Any]:
    """Compare ThermostatShadowGRU against Persistence, ExistingTrend, and ThermalInertiaLinear."""
    spec = dataset.spec
    b_pers = PersistenceBaseline(spec).fit(dataset, train_idx)
    b_trend = ExistingTrendBaseline(spec).fit(dataset, train_idx)
    b_lin = ThermalInertiaLinearBaseline(spec).fit(dataset, train_idx)

    def _eval_split(split_idx: np.ndarray) -> dict[str, Any]:
        if split_idx.size == 0:
            return {"n_samples": 0}
        # 1. Historical action-conditioned replay (uses actual executed future actions as conditioning)
        gru_replay = predict_numpy(artifact, dataset.X_hist[split_idx], dataset.X_act_actual[split_idx])
        lin_replay = b_lin.predict(dataset, split_idx, use_actual_actions=True)
        pers_pred = b_pers.predict(dataset, split_idx, use_actual_actions=False)
        trend_pred = b_trend.predict(dataset, split_idx, use_actual_actions=False)

        # 2. Online-style evaluation (uses ONLY known-at-t0 hold actions; masks mid-horizon intervened intervals)
        gru_online = predict_numpy(artifact, dataset.X_hist[split_idx], dataset.X_act_hold[split_idx])
        lin_online = b_lin.predict(dataset, split_idx, use_actual_actions=False)

        return {
            "n_samples": int(split_idx.size),
            "historical_replay_action_conditioned": {
                "neural_gru": compute_prediction_metrics(dataset, split_idx, gru_replay, mask_intervened=False),
                "thermal_inertia_linear": compute_prediction_metrics(dataset, split_idx, lin_replay, mask_intervened=False),
                "existing_trend": compute_prediction_metrics(dataset, split_idx, trend_pred, mask_intervened=False),
                "persistence": compute_prediction_metrics(dataset, split_idx, pers_pred, mask_intervened=False),
            },
            "online_style_known_at_t0_non_intervened": {
                "neural_gru": compute_prediction_metrics(dataset, split_idx, gru_online, mask_intervened=True),
                "thermal_inertia_linear": compute_prediction_metrics(dataset, split_idx, lin_online, mask_intervened=True),
                "existing_trend": compute_prediction_metrics(dataset, split_idx, trend_pred, mask_intervened=True),
                "persistence": compute_prediction_metrics(dataset, split_idx, pers_pred, mask_intervened=True),
            },
        }

    action_audit = evaluate_action_response(
        artifact,
        dataset,
        test_idx if test_idx.size > 0 else val_idx,
        min_temp_response_c=min_temp_response_c,
        min_action_events=min_action_events,
        min_active_mode_samples=min_active_mode_samples,
    )
    return {
        "model_id": artifact.model_id,
        "model_state": artifact.model_state.value,
        "data_origin": dataset.manifest.get("data_origin"),
        "parameter_count": artifact.model.parameter_count(),
        "split_sizes": {
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
        },
        "validation": _eval_split(val_idx),
        "test": _eval_split(test_idx),
        "action_response_audit": action_audit,
    }
