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
) -> dict[str, Any]:
    """Verify the neural model genuinely responds to candidate actions and audit action coverage."""
    spec = dataset.spec
    site = dataset.site
    manifest = dataset.manifest

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
            "probes": {},
            "note": "No evaluation samples available for action-response check.",
        }

    # Select up to 48 representative evaluation samples
    probe_idx = eval_indices[: min(48, eval_indices.size)]
    xh = dataset.X_hist[probe_idx]
    probe_results: dict[str, Any] = {}
    any_significant_response = False
    directional_checks_passed = 0
    directional_checks_total = 0

    for u_idx, uname in enumerate(spec.unit_names):
        act_map = site.actuator_by_name(uname)
        z_idx = spec.zone_names.index(act_map.home_zone)
        unit_probes: dict[str, Any] = {}
        for mode in ("heat", "cool"):
            # Count naturally occurring samples in this mode on eval split
            natural_count = sum(
                1 for idx in eval_indices if dataset.metadata[int(idx)]["unit_modes"].get(uname) == mode
            )
            base_sp = 23.0 if mode == "heat" else 24.0
            sp_plus = min(act_map.soft_limits[mode][1], base_sp + 1.0)
            sp_minus = max(act_map.soft_limits[mode][0], base_sp - 1.0)

            xa_plus = []
            xa_minus = []
            xa_off = []
            for idx in probe_idx:
                meta = dataset.metadata[int(idx)]
                modes_base = dict(meta["unit_modes"])
                sps_base = dict(meta["unit_setpoints"])
                modes_base[uname] = mode
                sps_base[uname] = base_sp

                plan_p = build_step_action_plan(spec, modes_base, sps_base, unit=uname, new_mode=mode, new_setpoint=sp_plus)
                plan_m = build_step_action_plan(spec, modes_base, sps_base, unit=uname, new_mode=mode, new_setpoint=sp_minus)
                plan_o = build_step_action_plan(spec, modes_base, sps_base, unit=uname, new_mode="off", new_setpoint=base_sp)

                xa_plus.append(
                    build_action_sequence_tensor(spec, site, modes_base, sps_base, meta["zone_temps"], plan_p)
                )
                xa_minus.append(
                    build_action_sequence_tensor(spec, site, modes_base, sps_base, meta["zone_temps"], plan_m)
                )
                xa_off.append(
                    build_action_sequence_tensor(spec, site, modes_base, sps_base, meta["zone_temps"], plan_o)
                )

            pred_p = predict_numpy(artifact, xh, np.stack(xa_plus, axis=0))
            pred_m = predict_numpy(artifact, xh, np.stack(xa_minus, axis=0))
            pred_o = predict_numpy(artifact, xh, np.stack(xa_off, axis=0))

            # Compare 60m temperature delta and 60m unit Wh for stronger vs weaker setting
            # Heat stronger = sp_plus; Cool stronger = sp_minus
            if mode == "heat":
                temp_diff = float(np.mean(pred_p["temp_delta"][:, z_idx, -1] - pred_m["temp_delta"][:, z_idx, -1]))
                wh_diff = float(np.mean(pred_p["unit_wh"][:, u_idx, -1] - pred_m["unit_wh"][:, u_idx, -1]))
                active_vs_off_temp = float(np.mean(pred_p["temp_delta"][:, z_idx, -1] - pred_o["temp_delta"][:, z_idx, -1]))
                active_vs_off_wh = float(np.mean(pred_p["unit_wh"][:, u_idx, -1] - pred_o["unit_wh"][:, u_idx, -1]))
                expected_temp_sign = 1.0
            else:
                temp_diff = float(np.mean(pred_m["temp_delta"][:, z_idx, -1] - pred_p["temp_delta"][:, z_idx, -1]))
                wh_diff = float(np.mean(pred_m["unit_wh"][:, u_idx, -1] - pred_p["unit_wh"][:, u_idx, -1]))
                active_vs_off_temp = float(np.mean(pred_m["temp_delta"][:, z_idx, -1] - pred_o["temp_delta"][:, z_idx, -1]))
                active_vs_off_wh = float(np.mean(pred_m["unit_wh"][:, u_idx, -1] - pred_o["unit_wh"][:, u_idx, -1]))
                expected_temp_sign = -1.0

            signed_temp_resp = temp_diff * expected_temp_sign
            signed_active_resp = active_vs_off_temp * expected_temp_sign
            # Only require directional check on modes actually observed in the dataset
            mode_observed = (heat_samples >= 10) if mode == "heat" else (cool_samples >= 10)
            passed_dir = (signed_temp_resp >= min_temp_response_c) or (
                signed_active_resp >= min_temp_response_c and active_vs_off_wh > 5.0
            )
            if mode_observed:
                directional_checks_total += 1
                if passed_dir:
                    directional_checks_passed += 1
            if abs(temp_diff) >= min_temp_response_c or abs(active_vs_off_temp) >= min_temp_response_c:
                any_significant_response = True

            unit_probes[mode] = {
                "natural_eval_samples_in_mode": natural_count,
                "mode_observed_in_dataset": mode_observed,
                "stronger_minus_weaker_temp_60m_c": round(temp_diff, 4),
                "stronger_minus_weaker_unit_wh_60m": round(wh_diff, 3),
                "active_minus_off_temp_60m_c": round(active_vs_off_temp, 4),
                "active_minus_off_unit_wh_60m": round(active_vs_off_wh, 3),
                "expected_temp_sign": expected_temp_sign,
                "directional_check_passed": bool(passed_dir),
            }
        probe_results[uname] = unit_probes

    action_response_passed = bool(
        any_significant_response
        and directional_checks_total > 0
        and directional_checks_passed >= max(1, directional_checks_total // 2)
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
        "probes": probe_results,
    }


def evaluate_all_models(
    artifact: ModelArtifact,
    dataset: BuiltDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
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

    action_audit = evaluate_action_response(artifact, dataset, test_idx if test_idx.size > 0 else val_idx)
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
