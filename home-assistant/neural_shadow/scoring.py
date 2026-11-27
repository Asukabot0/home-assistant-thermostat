"""Worst-zone comfort-first candidate scoring, energy tie-breaking, and abstention policy.

Key invariants:
1. Never average signed temperatures across rooms: master-bedroom over-cool (-1.2C)
   and living-room over-heat (+1.2C) MUST NOT cancel out.
2. Prioritize reducing worst-zone discomfort and persistent per-zone bias first.
3. Only when comfort is comparable across candidates, compare total 3-unit Wh.
4. Decline to recommend when the model is not READY_TO_USE, uncertain, data is missing,
   or the state is outside training coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from . import ModelState
from .config import SiteConfig
from .constraints import CandidateAction
from .data_pipeline import FeatureSpec, build_action_sequence_tensor
from .nn_model import ModelArtifact, predict_numpy


@dataclass(frozen=True)
class ScoringPolicy:
    """Thresholds for comfort priority, comparable-comfort energy tie-break, and abstention."""

    tracking_band_c: float = 0.3
    ideal_band_c: float = 0.5
    comfort_band_c: float = 0.7
    horizon_weights: tuple[float, ...] = (0.25, 0.35, 0.40)
    comfort_comparable_worst_tol: float = 0.05
    comfort_comparable_sum_tol: float = 0.08
    min_comfort_gain_over_hold: float = 0.05
    min_energy_saving_wh_when_comparable: float = 15.0
    max_peer_regression: float = 0.08
    max_uncertainty_c: float = 0.80
    ood_temp_margin_c: float = 3.0
    ood_outdoor_margin_c: float = 5.0


def compute_zone_discomfort(
    pred_errors_by_horizon: list[float],
    pred_sigmas_by_horizon: list[float],
    policy: ScoringPolicy,
) -> dict[str, float]:
    """Compute non-negative discomfort & persistent-bias score for ONE independent zone."""
    weights = policy.horizon_weights
    if len(weights) != len(pred_errors_by_horizon):
        weights = tuple(1.0 / len(pred_errors_by_horizon) for _ in pred_errors_by_horizon)

    step_cost = 0.0
    max_abs_err = 0.0
    for w, err, sig in zip(weights, pred_errors_by_horizon, pred_sigmas_by_horizon):
        ae = abs(err)
        max_abs_err = max(max_abs_err, ae)
        track_ex = max(0.0, ae - policy.tracking_band_c)
        ideal_ex = max(0.0, ae - policy.ideal_band_c)
        comfort_ex = max(0.0, ae - policy.comfort_band_c)
        risk_ex = max(0.0, (ae + 0.5 * sig) - policy.comfort_band_c)
        c = track_ex + 2.0 * ideal_ex + 4.0 * comfort_ex + 6.0 * (comfort_ex ** 2) + 0.35 * risk_ex
        step_cost += w * c

    # Persistent bias penalty across horizons (same-sign sustained over-cool or over-heat)
    mean_signed = float(np.mean(pred_errors_by_horizon))
    persistent_bias = max(0.0, abs(mean_signed) - policy.tracking_band_c)
    total = step_cost + 0.5 * persistent_bias
    return {
        "discomfort": round(float(total), 5),
        "persistent_bias_c": round(float(mean_signed), 4),
        "max_abs_error_c": round(float(max_abs_err), 4),
    }


def score_candidate_prediction(
    spec: FeatureSpec,
    t0_zone_temps: dict[str, float | None],
    t0_zone_targets: dict[str, float | None],
    pred_temp_delta: np.ndarray,  # [Z, K]
    pred_temp_sigma: np.ndarray,  # [Z, K]
    pred_unit_wh: np.ndarray,  # [U, K]
    pred_total_wh: np.ndarray,  # [K]
    pred_total_wh_sigma: np.ndarray,  # [K]
    policy: ScoringPolicy = ScoringPolicy(),
) -> dict[str, Any]:
    """Compute per-zone comfort metrics, worst-zone score, and 3-unit total Wh for one candidate."""
    per_zone: dict[str, Any] = {}
    zone_discomforts: list[float] = []
    max_sigma = 0.0

    for z_idx, zname in enumerate(spec.zone_names):
        t0 = t0_zone_temps.get(zname)
        tgt = t0_zone_targets.get(zname)
        if t0 is None or tgt is None:
            per_zone[zname] = {"missing": True}
            continue
        deltas = [float(pred_temp_delta[z_idx, k]) for k in range(len(spec.horizons_minutes))]
        sigmas = [float(pred_temp_sigma[z_idx, k]) for k in range(len(spec.horizons_minutes))]
        pred_temps = [t0 + d for d in deltas]
        pred_errors = [pt - tgt for pt in pred_temps]
        max_sigma = max(max_sigma, max(sigmas))
        zscore = compute_zone_discomfort(pred_errors, sigmas, policy)
        zone_discomforts.append(zscore["discomfort"])
        per_zone[zname] = {
            **zscore,
            "pred_temp_c": {f"{h}m": round(pred_temps[k], 3) for k, h in enumerate(spec.horizons_minutes)},
            "pred_delta_c": {f"{h}m": round(deltas[k], 3) for k, h in enumerate(spec.horizons_minutes)},
            "pred_error_c": {f"{h}m": round(pred_errors[k], 3) for k, h in enumerate(spec.horizons_minutes)},
            "pred_sigma_c": {f"{h}m": round(sigmas[k], 3) for k, h in enumerate(spec.horizons_minutes)},
        }

    worst_zone = max(zone_discomforts) if zone_discomforts else float("inf")
    sum_zone = sum(zone_discomforts) if zone_discomforts else float("inf")
    unit_wh_dict = {
        uname: {f"{h}m": round(float(pred_unit_wh[u_idx, k]), 2) for k, h in enumerate(spec.horizons_minutes)}
        for u_idx, uname in enumerate(spec.unit_names)
    }
    total_wh_dict = {f"{h}m": round(float(pred_total_wh[k]), 2) for k, h in enumerate(spec.horizons_minutes)}
    total_wh_sigma_dict = {
        f"{h}m": round(float(pred_total_wh_sigma[k]), 2) for k, h in enumerate(spec.horizons_minutes)
    }
    return {
        "zones": per_zone,
        "worst_zone_discomfort": round(float(worst_zone), 5) if math.isfinite(worst_zone) else None,
        "sum_zone_discomfort": round(float(sum_zone), 5) if math.isfinite(sum_zone) else None,
        "max_temp_sigma_c": round(float(max_sigma), 4),
        "unit_wh": unit_wh_dict,
        "total_wh": total_wh_dict,
        "total_wh_sigma": total_wh_sigma_dict,
        "total_wh_60m": round(float(pred_total_wh[-1]), 2),
    }


def check_applicability_and_abstention(
    artifact: ModelArtifact,
    spec: FeatureSpec,
    meta: dict[str, Any],
    hold_score: dict[str, Any] | None,
    policy: ScoringPolicy = ScoringPolicy(),
) -> tuple[bool, list[str], dict[str, Any]]:
    """Evaluate applicability scope and determine whether shadow runtime must decline recommendations."""
    reasons: list[str] = []
    ood_flags: list[str] = []

    if artifact.model_state != ModelState.READY_TO_USE:
        reasons.append(f"model_state_not_ready:{artifact.model_state.value}")
    if artifact.normalizer is None:
        reasons.append("missing_feature_normalizer")
    if artifact.metadata.get("version_mismatches"):
        reasons.append("model_version_mismatch")

    # Data completeness checks for all independent measurement zones
    missing_zones = [
        z for z in spec.zone_names if meta["zone_temps"].get(z) is None or meta["zone_targets"].get(z) is None
    ]
    if missing_zones:
        reasons.append("missing_zone_observation:" + ",".join(missing_zones))

    # OOD checks against training coverage envelope
    envelope = artifact.normalizer.coverage_envelope if artifact.normalizer is not None else {}
    z_ranges = envelope.get("zone_temp_min_max_c") or {}
    for z in spec.zone_names:
        val = meta["zone_temps"].get(z)
        rng = z_ranges.get(z)
        if val is not None and rng is not None:
            if val < rng[0] - policy.ood_temp_margin_c or val > rng[1] + policy.ood_temp_margin_c:
                ood_flags.append(f"zone_temp_ood:{z}:{val:.1f}C_outside_[{rng[0]:.1f},{rng[1]:.1f}]")

    e_ranges = envelope.get("env_temp_min_max_c") or {}
    for e in spec.env_sensor_names:
        val = meta["env_temps"].get(e)
        rng = e_ranges.get(e)
        if val is not None and rng is not None:
            if val < rng[0] - policy.ood_outdoor_margin_c or val > rng[1] + policy.ood_outdoor_margin_c:
                ood_flags.append(f"outdoor_temp_ood:{e}:{val:.1f}C_outside_[{rng[0]:.1f},{rng[1]:.1f}]")

    seen_modes = envelope.get("unit_seen_modes") or {}
    for u in spec.unit_names:
        mode = meta["unit_modes"].get(u)
        if mode in {"heat", "cool"} and seen_modes.get(u) and mode not in seen_modes[u]:
            ood_flags.append(f"unseen_unit_mode:{u}:{mode}")

    if ood_flags:
        reasons.append("outside_training_coverage")

    # Model uncertainty check
    if hold_score is not None:
        max_sig = hold_score.get("max_temp_sigma_c")
        if max_sig is None or not math.isfinite(max_sig):
            reasons.append("non_finite_prediction")
        elif max_sig > policy.max_uncertainty_c:
            reasons.append(f"high_prediction_uncertainty:{max_sig:.3f}C>{policy.max_uncertainty_c}C")

    scope = {
        "in_scope": len(reasons) == 0,
        "ood_flags": ood_flags,
        "missing_zones": missing_zones,
        "quality_flags": meta.get("quality_flags", {}),
    }
    return len(reasons) == 0, reasons, scope


def evaluate_and_rank_candidates(
    artifact: ModelArtifact,
    site: SiteConfig,
    spec: FeatureSpec,
    x_hist_single: np.ndarray,  # [H, D_hist]
    meta: dict[str, Any],
    candidates: list[CandidateAction],
    policy: ScoringPolicy = ScoringPolicy(),
) -> dict[str, Any]:
    """Score all valid V1 candidates and select recommendation (or decline)."""
    if not candidates:
        return {
            "candidate_scores": [],
            "recommended_candidate_id": None,
            "declined": True,
            "decline_reasons": ["no_valid_candidates"],
            "applicability_scope": {"in_scope": False, "ood_flags": [], "missing_zones": []},
            "selection_reason": "no_valid_candidates",
        }

    # Batch inference across all candidates at once
    x_hist_batch = np.repeat(x_hist_single[None, :, :], len(candidates), axis=0)
    x_act_batch = np.stack(
        [
            build_action_sequence_tensor(
                spec, site, meta["unit_modes"], meta["unit_setpoints"], meta["zone_temps"], cand.step_plans
            )
            for cand in candidates
        ],
        axis=0,
    )
    preds = predict_numpy(artifact, x_hist_batch, x_act_batch)

    scored: list[dict[str, Any]] = []
    hold_entry: dict[str, Any] | None = None
    for idx, cand in enumerate(candidates):
        sc = score_candidate_prediction(
            spec,
            meta["zone_temps"],
            meta["zone_targets"],
            preds["temp_delta"][idx],
            preds["temp_sigma"][idx],
            preds["unit_wh"][idx],
            preds["total_wh"][idx],
            preds["total_wh_sigma"][idx],
            policy=policy,
        )
        entry = {"candidate": cand.to_public_dict(), "prediction": sc}
        scored.append(entry)
        if cand.candidate_id == "hold":
            hold_entry = entry

    can_recommend, decline_reasons, scope = check_applicability_and_abstention(
        artifact, spec, meta, hold_entry["prediction"] if hold_entry else None, policy=policy
    )
    if not can_recommend or hold_entry is None:
        return {
            "candidate_scores": scored,
            "recommended_candidate_id": None,
            "declined": True,
            "decline_reasons": decline_reasons or ["missing_hold_baseline"],
            "applicability_scope": scope,
            "selection_reason": "declined:" + ",".join(decline_reasons),
        }

    # Filter out candidates with excessive uncertainty or unacceptable peer zone regression
    hold_pred = hold_entry["prediction"]
    eligible: list[dict[str, Any]] = []
    peer_rejected: list[dict[str, Any]] = []
    for entry in scored:
        p = entry["prediction"]
        cid = entry["candidate"]["candidate_id"]
        if p["worst_zone_discomfort"] is None or p["max_temp_sigma_c"] > policy.max_uncertainty_c:
            continue
        if cid != "hold":
            harmful = False
            for zname in spec.zone_names:
                z_new = p["zones"].get(zname, {})
                z_old = hold_pred["zones"].get(zname, {})
                if z_new.get("missing") or z_old.get("missing"):
                    continue
                if (
                    z_new["max_abs_error_c"] > policy.comfort_band_c
                    and z_new["discomfort"] > z_old["discomfort"] + policy.max_peer_regression
                ):
                    harmful = True
                    peer_rejected.append({"candidate_id": cid, "harmed_zone": zname})
                    break
            if harmful:
                continue
        eligible.append(entry)

    if not eligible:
        return {
            "candidate_scores": scored,
            "recommended_candidate_id": None,
            "declined": True,
            "decline_reasons": ["no_safe_low_uncertainty_candidate"],
            "applicability_scope": scope,
            "selection_reason": "declined:no_safe_low_uncertainty_candidate",
        }

    # 1. Comfort first: minimize worst-zone discomfort, then sum of per-zone discomforts.
    best_worst = min(e["prediction"]["worst_zone_discomfort"] for e in eligible)
    best_sum = min(e["prediction"]["sum_zone_discomfort"] for e in eligible)

    comparable = [
        e
        for e in eligible
        if e["prediction"]["worst_zone_discomfort"] <= best_worst + policy.comfort_comparable_worst_tol + 1e-9
        and e["prediction"]["sum_zone_discomfort"] <= best_sum + policy.comfort_comparable_sum_tol + 1e-9
    ]
    if not comparable:
        comparable = [
            min(
                eligible,
                key=lambda e: (
                    e["prediction"]["worst_zone_discomfort"],
                    e["prediction"]["sum_zone_discomfort"],
                    e["prediction"]["total_wh_60m"],
                ),
            )
        ]

    # 2. Among comfort-comparable candidates, minimize total 3-unit Wh.
    chosen = min(
        comparable,
        key=lambda e: (
            e["prediction"]["total_wh_60m"],
            e["prediction"]["worst_zone_discomfort"],
            e["prediction"]["sum_zone_discomfort"],
            0 if e["candidate"]["candidate_id"] == "hold" else 1,
        ),
    )

    chosen_id = chosen["candidate"]["candidate_id"]
    selection_reason = "hold_optimal"
    if chosen_id != "hold":
        cp = chosen["prediction"]
        worst_gain = hold_pred["worst_zone_discomfort"] - cp["worst_zone_discomfort"]
        sum_gain = hold_pred["sum_zone_discomfort"] - cp["sum_zone_discomfort"]
        wh_saving = hold_pred["total_wh_60m"] - cp["total_wh_60m"]
        comfort_improved = (
            worst_gain >= policy.min_comfort_gain_over_hold or sum_gain >= policy.min_comfort_gain_over_hold * 1.5
        )
        energy_improved = (
            cp["worst_zone_discomfort"] <= hold_pred["worst_zone_discomfort"] + policy.comfort_comparable_worst_tol
            and cp["sum_zone_discomfort"] <= hold_pred["sum_zone_discomfort"] + policy.comfort_comparable_sum_tol
            and wh_saving >= policy.min_energy_saving_wh_when_comparable
        )
        if comfort_improved:
            selection_reason = f"comfort_priority_gain(worst_delta={worst_gain:+.3f},wh_delta={-wh_saving:+.1f})"
        elif energy_improved:
            selection_reason = f"comparable_comfort_energy_saving(wh_saving={wh_saving:.1f}Wh)"
        else:
            chosen_id = "hold"
            selection_reason = "hold_within_hysteresis_margin"

    return {
        "candidate_scores": scored,
        "peer_rejected": peer_rejected,
        "recommended_candidate_id": chosen_id,
        "declined": False,
        "decline_reasons": [],
        "applicability_scope": scope,
        "selection_reason": selection_reason,
    }
