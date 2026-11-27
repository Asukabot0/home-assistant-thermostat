"""Reproducible forward-chaining training, validation calibration, and model readiness gating."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import DataOrigin, FEATURE_VERSION, MODEL_ARCH_VERSION, ModelState
from .data_pipeline import (
    BuiltDataset,
    DEFAULT_SETTLE_WINDOW_SECONDS,
    MAX_POWER_GAP_SECONDS,
)
from .evaluate import Z_80, evaluate_all_models
from .nn_model import (
    FeatureNormalizer,
    ModelArtifact,
    ThermostatShadowGRU,
    build_untrained_artifact,
    predict_numpy,
    save_checkpoint,
)


@dataclass(frozen=True)
class ReadinessGates:
    """Explicit gates separating insufficient_data, uncalibrated, and ready_to_use models."""

    min_real_history_hours: float = 72.0
    min_train_samples: int = 400
    min_val_samples: int = 120
    min_test_samples: int = 120
    min_unique_temp_reports_per_zone: int = 150
    min_mean_power_coverage: float = 0.75
    min_action_events_total: int = 30
    min_active_mode_samples: int = 60
    min_pi80_coverage: float = 0.65
    max_pi80_coverage: float = 0.95
    min_action_response_temp_c: float = 0.03
    allow_synthetic_ready: bool = False  # Production default: synthetic NEVER passes ready_to_use


@dataclass(frozen=True)
class ForwardChainSplit:
    """Time-ordered train/val/test split with explicit max-horizon embargo gaps."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_cutoff_ts: float | None
    val_cutoff_ts: float | None
    embargo_seconds: float


def forward_chain_split(
    cutoff_ts: np.ndarray,
    *,
    max_horizon_seconds: float = 3600.0,
    settle_window_seconds: float = DEFAULT_SETTLE_WINDOW_SECONDS,
    max_power_gap_seconds: float = MAX_POWER_GAP_SECONDS,
    embargo_seconds: float | None = None,
    max_label_ts: np.ndarray | None = None,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> ForwardChainSplit:
    """Chronological forward-chaining split guaranteeing zero future label leakage across boundaries.

    The embargo covers the full worst-case label look-ahead:
      embargo_seconds = max_horizon_seconds + settle_window_seconds + max_power_gap_seconds
    and, when per-sample ``max_label_ts`` is supplied, every sample's actual latest label
    observation timestamp must also precede the next split boundary.
    """
    effective_embargo = (
        float(embargo_seconds)
        if embargo_seconds is not None
        else float(max_horizon_seconds) + float(settle_window_seconds) + float(max_power_gap_seconds)
    )
    n = int(cutoff_ts.shape[0])
    if n < 6:
        empty = np.zeros((0,), dtype=np.int64)
        return ForwardChainSplit(empty, empty, empty, None, None, effective_embargo)

    cutoff_arr = np.asarray(cutoff_ts, dtype=np.float64)
    label_end = cutoff_arr + effective_embargo
    if max_label_ts is not None:
        max_label_arr = np.asarray(max_label_ts, dtype=np.float64)
        if max_label_arr.shape != cutoff_arr.shape:
            raise ValueError("max_label_ts must match cutoff_ts shape")
        label_end = np.maximum(label_end, max_label_arr)

    order = np.argsort(cutoff_arr, kind="mergesort")
    ts_sorted = cutoff_arr[order]

    t_start = float(ts_sorted[0])
    t_end = float(ts_sorted[-1])
    total_span = t_end - t_start
    if total_span <= 2.5 * effective_embargo:
        tr_pos = max(1, int(n * train_frac))
        va_pos = max(tr_pos + 1, int(n * (train_frac + val_frac)))
        train_cutoff = float(ts_sorted[min(tr_pos, n - 1)])
        val_cutoff = float(ts_sorted[min(va_pos, n - 1)])
    else:
        train_cutoff = float(np.quantile(ts_sorted, train_frac))
        val_cutoff = float(np.quantile(ts_sorted, train_frac + val_frac))

    train_mask = label_end <= train_cutoff + 1e-6
    val_mask = (cutoff_arr >= train_cutoff - 1e-6) & (label_end <= val_cutoff + 1e-6)
    test_mask = cutoff_arr >= val_cutoff - 1e-6

    train_idx = np.where(train_mask)[0].astype(np.int64)
    val_idx = np.where(val_mask)[0].astype(np.int64)
    test_idx = np.where(test_mask)[0].astype(np.int64)

    # Assert mathematically zero overlap and zero label bleed across boundaries.
    if train_idx.size > 0 and val_idx.size > 0:
        assert float(label_end[train_idx].max()) <= float(cutoff_arr[val_idx].min()) + 1e-5
        assert float(cutoff_arr[train_idx].max()) + effective_embargo <= float(cutoff_arr[val_idx].min()) + 1e-5
    if val_idx.size > 0 and test_idx.size > 0:
        assert float(label_end[val_idx].max()) <= float(cutoff_arr[test_idx].min()) + 1e-5
        assert float(cutoff_arr[val_idx].max()) + effective_embargo <= float(cutoff_arr[test_idx].min()) + 1e-5

    return ForwardChainSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_cutoff_ts=train_cutoff,
        val_cutoff_ts=val_cutoff,
        embargo_seconds=effective_embargo,
    )


def forward_chain_split_dataset(
    dataset: BuiltDataset,
    *,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> ForwardChainSplit:
    """Split a BuiltDataset using its full worst-case label look-ahead and actual per-sample max_label_ts."""
    max_h = float(max(dataset.spec.horizons_minutes) * 60)
    settle_w = float(dataset.manifest.get("settle_window_seconds", DEFAULT_SETTLE_WINDOW_SECONDS))
    power_gap = float(dataset.manifest.get("max_power_gap_seconds", MAX_POWER_GAP_SECONDS))
    embargo_s = max_h + settle_w + power_gap
    max_label_ts = np.asarray(
        [
            float(m.get("max_label_ts", float(c) + embargo_s))
            for c, m in zip(dataset.cutoff_ts, dataset.metadata)
        ],
        dtype=np.float64,
    )
    return forward_chain_split(
        dataset.cutoff_ts,
        max_horizon_seconds=max_h,
        settle_window_seconds=settle_w,
        max_power_gap_seconds=power_gap,
        embargo_seconds=embargo_s,
        max_label_ts=max_label_ts,
        train_frac=train_frac,
        val_frac=val_frac,
    )


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _masked_gaussian_huber_loss(
    pred: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    huber_beta: float = 0.5,
    nll_weight: float = 0.35,
) -> torch.Tensor:
    mask = valid & torch.isfinite(target)
    count = mask.sum()
    if int(count.item()) == 0:
        return pred.sum() * 0.0
    p = pred[mask]
    s = torch.clamp(sigma[mask], min=1e-3)
    y = torch.nan_to_num(target[mask], nan=0.0)
    huber = F.smooth_l1_loss(p, y, beta=huber_beta, reduction="mean")
    nll = torch.mean(0.5 * ((y - p) / s) ** 2 + torch.log(s))
    return huber + nll_weight * nll


def _compute_batch_loss(
    out: dict[str, torch.Tensor],
    y_td: torch.Tensor,
    y_tv: torch.Tensor,
    y_uwh: torch.Tensor,
    y_uval: torch.Tensor,
    y_twh: torch.Tensor,
    y_tval: torch.Tensor,
) -> torch.Tensor:
    # Temperature loss (C)
    loss_t = _masked_gaussian_huber_loss(out["temp_delta"], out["temp_sigma"], y_td, y_tv, huber_beta=0.35, nll_weight=0.30)
    # Energy loss on scaled Wh (100 Wh ~ 1.0 unit) to balance gradients cleanly with C errors
    scale = 100.0
    loss_uwh = _masked_gaussian_huber_loss(
        out["unit_wh"] / scale,
        out["unit_wh_sigma"] / scale,
        y_uwh / scale,
        y_uval,
        huber_beta=0.5,
        nll_weight=0.25,
    )
    loss_twh = _masked_gaussian_huber_loss(
        out["total_wh"] / scale,
        out["total_wh_sigma"] / scale,
        y_twh / scale,
        y_tval,
        huber_beta=0.5,
        nll_weight=0.25,
    )
    return loss_t + 0.5 * loss_uwh + 0.5 * loss_twh


def _calibrate_uncertainty_on_val(
    artifact: ModelArtifact,
    dataset: BuiltDataset,
    val_idx: np.ndarray,
) -> dict[str, Any]:
    """Fit multiplicative conformal calibration scales STRICTLY on validation split for 80% PI."""
    if val_idx.size < 10:
        return {"calibrated": False, "reason": f"val_split_too_small:{val_idx.size}"}

    # Reset calibration scales to 1.0 before computing raw validation z-scores.
    artifact.model.temp_cal_scale.fill_(1.0)
    artifact.model.unit_wh_cal_scale.fill_(1.0)
    artifact.model.total_wh_cal_scale.fill_(1.0)

    preds = predict_numpy(artifact, dataset.X_hist[val_idx], dataset.X_act_actual[val_idx])
    y_td = dataset.y_temp_delta[val_idx]
    y_tv = dataset.y_temp_valid[val_idx]
    y_uwh = dataset.y_unit_wh[val_idx]
    y_uval = dataset.y_unit_valid[val_idx]
    y_twh = dataset.y_total_wh[val_idx]
    y_tval = dataset.y_total_valid[val_idx]

    n_z = artifact.model.n_zones
    n_u = artifact.model.n_units
    n_k = artifact.model.n_horizons

    calibrated_cells = 0
    for z in range(n_z):
        for k in range(n_k):
            mask = y_tv[:, z, k] & np.isfinite(y_td[:, z, k])
            if mask.sum() >= 10:
                z_scores = np.abs(y_td[mask, z, k] - preds["temp_delta"][mask, z, k]) / np.maximum(
                    preds["temp_sigma"][mask, z, k], 1e-4
                )
                q80 = float(np.quantile(z_scores, 0.80))
                mult = float(np.clip(q80 / Z_80, 0.5, 4.0))
                artifact.model.temp_cal_scale[z, k] = mult
                calibrated_cells += 1

    for u in range(n_u):
        for k in range(n_k):
            mask = y_uval[:, u, k] & np.isfinite(y_uwh[:, u, k])
            if mask.sum() >= 10:
                z_scores = np.abs(y_uwh[mask, u, k] - preds["unit_wh"][mask, u, k]) / np.maximum(
                    preds["unit_wh_sigma"][mask, u, k], 1.0
                )
                q80 = float(np.quantile(z_scores, 0.80))
                artifact.model.unit_wh_cal_scale[u, k] = float(np.clip(q80 / Z_80, 0.5, 4.0))

    for k in range(n_k):
        mask = y_tval[:, k] & np.isfinite(y_twh[:, k])
        if mask.sum() >= 10:
            z_scores = np.abs(y_twh[mask, k] - preds["total_wh"][mask, k]) / np.maximum(
                preds["total_wh_sigma"][mask, k], 2.0
            )
            q80 = float(np.quantile(z_scores, 0.80))
            artifact.model.total_wh_cal_scale[k] = float(np.clip(q80 / Z_80, 0.5, 4.0))

    calibrated = calibrated_cells == (n_z * n_k)
    return {
        "calibrated": calibrated,
        "calibrated_zone_horizon_cells": calibrated_cells,
        "total_zone_horizon_cells": n_z * n_k,
        "temp_cal_scale": artifact.model.temp_cal_scale.detach().cpu().tolist(),
        "total_wh_cal_scale": artifact.model.total_wh_cal_scale.detach().cpu().tolist(),
    }


def determine_model_state(
    dataset: BuiltDataset,
    split: ForwardChainSplit,
    cal_info: dict[str, Any],
    eval_report: dict[str, Any],
    gates: ReadinessGates,
) -> tuple[ModelState, list[str]]:
    """Classify the model lifecycle state honestly based on data sufficiency, calibration, and action response."""
    manifest = dataset.manifest
    data_origin = manifest.get("data_origin", DataOrigin.REAL.value)
    insufficiency_reasons: list[str] = []
    uncalibrated_reasons: list[str] = []

    if data_origin == DataOrigin.SYNTHETIC.value and not gates.allow_synthetic_ready:
        insufficiency_reasons.append("synthetic_verification_data_not_eligible_for_deployment")

    span_h = float(manifest.get("observed_span_hours", 0.0))
    if span_h < gates.min_real_history_hours:
        insufficiency_reasons.append(f"history_span_hours:{span_h:.2f}<{gates.min_real_history_hours}")

    if split.train_idx.size < gates.min_train_samples:
        insufficiency_reasons.append(f"train_samples:{split.train_idx.size}<{gates.min_train_samples}")
    if split.val_idx.size < gates.min_val_samples:
        insufficiency_reasons.append(f"val_samples:{split.val_idx.size}<{gates.min_val_samples}")
    if split.test_idx.size < gates.min_test_samples:
        insufficiency_reasons.append(f"test_samples:{split.test_idx.size}<{gates.min_test_samples}")

    for zname, cnt in (manifest.get("unique_temp_reports_by_zone") or {}).items():
        if int(cnt) < gates.min_unique_temp_reports_per_zone:
            insufficiency_reasons.append(
                f"unique_temp_reports_{zname}:{cnt}<{gates.min_unique_temp_reports_per_zone}"
            )

    cov_vals = [
        float(v.get("coverage_ratio", 0.0)) for v in (manifest.get("power_coverage_by_unit") or {}).values()
    ]
    mean_cov = float(np.mean(cov_vals)) if cov_vals else 0.0
    if mean_cov < gates.min_mean_power_coverage:
        insufficiency_reasons.append(f"mean_power_coverage:{mean_cov:.3f}<{gates.min_mean_power_coverage}")

    action_audit = eval_report.get("action_response_audit") or {}
    if not action_audit.get("action_coverage_sufficient", False):
        insufficiency_reasons.extend(action_audit.get("action_coverage_issues", ["action_coverage_insufficient"]))

    if insufficiency_reasons:
        return ModelState.INSUFFICIENT_DATA, insufficiency_reasons

    # Data is sufficient; check calibration, action responsiveness, and baseline sanity.
    if not cal_info.get("calibrated", False):
        uncalibrated_reasons.append(str(cal_info.get("reason", "validation_calibration_incomplete")))

    if not action_audit.get("action_response_passed", False):
        uncalibrated_reasons.append("action_response_sensitivity_failed")

    test_replay = (eval_report.get("test") or {}).get("historical_replay_action_conditioned") or {}
    gru_metrics = test_replay.get("neural_gru") or {}
    pers_metrics = test_replay.get("persistence") or {}

    for h_key, stats in (gru_metrics.get("temp_by_horizon") or {}).items():
        cov = stats.get("pi80_coverage")
        if cov is None or not (gates.min_pi80_coverage <= cov <= gates.max_pi80_coverage):
            uncalibrated_reasons.append(f"test_pi80_coverage_{h_key}:{cov}")

    # Model should not be materially worse than naive persistence on 60m temperature MAE.
    gru_mae_60 = (gru_metrics.get("temp_by_horizon") or {}).get("60m", {}).get("mae")
    pers_mae_60 = (pers_metrics.get("temp_by_horizon") or {}).get("60m", {}).get("mae")
    if gru_mae_60 is not None and pers_mae_60 is not None and gru_mae_60 > pers_mae_60 * 1.05:
        uncalibrated_reasons.append(f"gru_60m_mae_exceeds_persistence:{gru_mae_60}>{pers_mae_60}")

    if uncalibrated_reasons:
        return ModelState.UNCALIBRATED, uncalibrated_reasons

    return ModelState.READY_TO_USE, ["all_data_calibration_and_action_response_gates_passed"]


def train_and_evaluate(
    dataset: BuiltDataset,
    *,
    output_dir: Path | None = None,
    seed: int = 42,
    epochs: int = 45,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    gates: ReadinessGates | None = None,
) -> tuple[ModelArtifact, dict[str, Any]]:
    """Train ThermostatShadowGRU with forward-chaining split, calibrate on val, and evaluate."""
    _set_deterministic_seed(seed)
    gates = gates or ReadinessGates()
    split = forward_chain_split_dataset(dataset)

    base_artifact = build_untrained_artifact(dataset.site, dataset.spec, seed=seed)
    model = base_artifact.model

    if split.train_idx.size < 8 or split.val_idx.size < 4:
        # Not enough samples even for a diagnostic training fit — keep random weights explicitly marked INSUFFICIENT_DATA.
        meta = dict(base_artifact.metadata)
        meta.update(
            model_id=f"insufficient-{dataset.manifest['data_sha256'][:10]}-s{seed}",
            model_state=ModelState.INSUFFICIENT_DATA.value,
            data_origin=dataset.manifest.get("data_origin"),
            training_cutoff_ts=split.train_cutoff_ts,
            val_cutoff_ts=split.val_cutoff_ts,
            trained_at_utc=datetime.now(timezone.utc).isoformat(),
            data_sha256=dataset.manifest.get("data_sha256"),
            readiness_reasons=[
                f"cannot_train_split_too_small:train={split.train_idx.size},val={split.val_idx.size},test={split.test_idx.size}"
            ],
            weights_trained=False,
        )
        artifact = ModelArtifact(model=model, normalizer=None, model_state=ModelState.INSUFFICIENT_DATA, metadata=meta)
        report = {
            "model_id": artifact.model_id,
            "model_state": artifact.model_state.value,
            "weights_trained": False,
            "data_origin": dataset.manifest.get("data_origin"),
            "parameter_count": model.parameter_count(),
            "seed": seed,
            "training_cutoff_ts": split.train_cutoff_ts,
            "val_cutoff_ts": split.val_cutoff_ts,
            "readiness_reasons": meta["readiness_reasons"],
            "split_sizes": {
                "train": int(split.train_idx.size),
                "val": int(split.val_idx.size),
                "test": int(split.test_idx.size),
            },
            "manifest": dataset.manifest,
        }
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(artifact, output_dir / "thermostat_shadow_gru.pt")
            (output_dir / "evaluation_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        return artifact, report

    # Fit normalizer & missing-value imputer STRICTLY on training split only!
    train_meta = [dataset.metadata[int(i)] for i in split.train_idx]
    normalizer = FeatureNormalizer.fit(
        dataset.X_hist[split.train_idx],
        dataset.X_act_actual[split.train_idx],
        train_meta,
        dataset.spec,
    )

    xh_tr = torch.from_numpy(normalizer.transform_hist(dataset.X_hist[split.train_idx]))
    xa_tr = torch.from_numpy(normalizer.transform_act(dataset.X_act_actual[split.train_idx]))
    ytd_tr = torch.from_numpy(dataset.y_temp_delta[split.train_idx])
    ytv_tr = torch.from_numpy(dataset.y_temp_valid[split.train_idx])
    yuwh_tr = torch.from_numpy(dataset.y_unit_wh[split.train_idx])
    yuval_tr = torch.from_numpy(dataset.y_unit_valid[split.train_idx])
    ytwh_tr = torch.from_numpy(dataset.y_total_wh[split.train_idx])
    ytval_tr = torch.from_numpy(dataset.y_total_valid[split.train_idx])

    xh_va = torch.from_numpy(normalizer.transform_hist(dataset.X_hist[split.val_idx]))
    xa_va = torch.from_numpy(normalizer.transform_act(dataset.X_act_actual[split.val_idx]))
    ytd_va = torch.from_numpy(dataset.y_temp_delta[split.val_idx])
    ytv_va = torch.from_numpy(dataset.y_temp_valid[split.val_idx])
    yuwh_va = torch.from_numpy(dataset.y_unit_wh[split.val_idx])
    yuval_va = torch.from_numpy(dataset.y_unit_valid[split.val_idx])
    ytwh_va = torch.from_numpy(dataset.y_total_wh[split.val_idx])
    ytval_va = torch.from_numpy(dataset.y_total_valid[split.val_idx])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val_loss = float("inf")
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    stale_epochs = 0
    history_losses: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)

    n_tr = xh_tr.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_tr)
        epoch_loss_sum = 0.0
        n_batches = 0
        for start in range(0, n_tr, batch_size):
            b_idx = torch.from_numpy(perm[start : start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            out = model(xh_tr[b_idx], xa_tr[b_idx])
            loss = _compute_batch_loss(
                out,
                ytd_tr[b_idx],
                ytv_tr[b_idx],
                yuwh_tr[b_idx],
                yuval_tr[b_idx],
                ytwh_tr[b_idx],
                ytval_tr[b_idx],
            )
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss_sum += float(loss.item())
                n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(xh_va, xa_va)
            val_loss = float(_compute_batch_loss(val_out, ytd_va, ytv_va, yuwh_va, yuval_va, ytwh_va, ytval_va).item())

        tr_loss = epoch_loss_sum / max(1, n_batches)
        history_losses.append({"epoch": epoch, "train_loss": round(tr_loss, 5), "val_loss": round(val_loss, 5)})
        if math.isfinite(val_loss) and val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state_dict)
    model.eval()

    model_id = (
        f"ns-gru-v1-{dataset.manifest['data_sha256'][:10]}-s{seed}"
        f"-cut{int(split.train_cutoff_ts or 0)}"
    )
    provisional_meta = {
        "model_id": model_id,
        "model_state": ModelState.UNCALIBRATED.value,
        "model_arch_version": MODEL_ARCH_VERSION,
        "feature_version": FEATURE_VERSION,
        "config_fingerprint": dataset.site.config_fingerprint,
        "observation_version": dataset.site.observation_version,
        "parameter_count": model.parameter_count(),
        "trainable_parameter_count": model.trainable_parameter_count(),
        "training_cutoff_ts": split.train_cutoff_ts,
        "training_cutoff_utc": datetime.fromtimestamp(split.train_cutoff_ts, timezone.utc).isoformat()
        if split.train_cutoff_ts is not None
        else None,
        "val_cutoff_ts": split.val_cutoff_ts,
        "embargo_seconds": split.embargo_seconds,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_origin": dataset.manifest.get("data_origin"),
        "data_sha256": dataset.manifest.get("data_sha256"),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 5),
        "weights_trained": True,
        "feature_spec": dataset.spec.to_dict(),
    }
    artifact = ModelArtifact(
        model=model,
        normalizer=normalizer,
        model_state=ModelState.UNCALIBRATED,
        metadata=provisional_meta,
    )

    # Calibrate uncertainty strictly on validation split
    cal_info = _calibrate_uncertainty_on_val(artifact, dataset, split.val_idx)
    eval_report = evaluate_all_models(artifact, dataset, split.train_idx, split.val_idx, split.test_idx)
    final_state, reasons = determine_model_state(dataset, split, cal_info, eval_report, gates)

    artifact.model_state = final_state
    artifact.metadata["model_state"] = final_state.value
    artifact.metadata["readiness_reasons"] = reasons
    artifact.metadata["calibration"] = cal_info
    artifact.metadata["deployment_eligible"] = (
        final_state == ModelState.READY_TO_USE and dataset.manifest.get("data_origin") == DataOrigin.REAL.value
    )

    full_report = {
        **eval_report,
        "model_state": final_state.value,
        "deployment_eligible": artifact.metadata["deployment_eligible"],
        "readiness_reasons": reasons,
        "weights_trained": True,
        "seed": seed,
        "training_cutoff_ts": split.train_cutoff_ts,
        "training_cutoff_utc": artifact.metadata["training_cutoff_utc"],
        "val_cutoff_ts": split.val_cutoff_ts,
        "embargo_seconds": split.embargo_seconds,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 5),
        "calibration": cal_info,
        "loss_curve_tail": history_losses[-5:],
        "manifest": dataset.manifest,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(artifact, output_dir / "thermostat_shadow_gru.pt")
        (output_dir / "evaluation_report.json").write_text(
            json.dumps(full_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return artifact, full_report
