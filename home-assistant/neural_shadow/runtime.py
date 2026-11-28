"""Independent read-only neural shadow runtime: seal predictions, settle labels, report.

Strict isolation guarantees:
- Zero actuation path (no HTTP clients, no HA service calls, no write access to pilot/HA dirs).
- Immutable sealing BEFORE label settlement (integrity-checked SHA-256 digest `seal_sha256`).
- Strict factual vs counterfactual separation and mid-horizon intervention detection.
- Full exception and timeout isolation so failures only affect shadow outputs.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
from tempfile import NamedTemporaryFile
import threading
import time
from traceback import extract_tb
from typing import Any

import numpy as np

from . import (
    CONSTRAINT_VERSION,
    DEFAULT_HISTORY_MINUTES,
    FEATURE_VERSION,
    HORIZONS_MINUTES,
    MODEL_ARCH_VERSION,
    ModelState,
    STEP_MINUTES,
)
from .config import SiteConfig, canonical_hash
from .constraints import generate_v1_candidates
from .data_pipeline import (
    CanonicalHistory,
    DEFAULT_MIN_WH_COVERAGE,
    DEFAULT_SETTLE_WINDOW_SECONDS,
    FeatureSpec,
    _HistoryIndex,
    build_feature_spec,
    extract_history_window,
    integrate_power_interval,
)
from .evaluate import Z_80
from .nn_model import (
    ModelArtifact,
    ModelVersionMismatchError,
    build_untrained_artifact,
    load_checkpoint,
)
from .scoring import ScoringPolicy, evaluate_and_rank_candidates

SHADOW_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS sealed_predictions (
    prediction_id TEXT PRIMARY KEY,
    sealed_at REAL NOT NULL,
    cutoff_ts REAL NOT NULL,
    model_id TEXT NOT NULL,
    model_state TEXT NOT NULL,
    model_arch_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    training_cutoff_ts REAL,
    recommended_candidate_id TEXT,
    declined INTEGER NOT NULL,
    seal_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sealed_cutoff ON sealed_predictions (cutoff_ts);

CREATE TABLE IF NOT EXISTS settlements (
    prediction_id TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    settled_at REAL NOT NULL,
    cutoff_ts REAL NOT NULL,
    horizon_end_ts REAL NOT NULL,
    intervened INTEGER NOT NULL,
    factual_settlement_kind TEXT NOT NULL,
    recommended_execution_status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (prediction_id, horizon_minutes),
    FOREIGN KEY (prediction_id) REFERENCES sealed_predictions(prediction_id)
);
CREATE INDEX IF NOT EXISTS idx_settlements_end ON settlements (horizon_end_ts);
"""


def _iso_utc(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def _init_shadow_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=5)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SHADOW_DB_SCHEMA)
    return db


class ShadowCycleTimeout(TimeoutError):
    """Raised when a single shadow cycle exceeds its local execution budget."""


@dataclass
class ShadowRuntimeConfig:
    state_dir: Path
    output_dir: Path
    history_minutes: int = DEFAULT_HISTORY_MINUTES
    step_minutes: int = STEP_MINUTES
    horizons_minutes: tuple[int, ...] = HORIZONS_MINUTES
    settle_window_seconds: float = DEFAULT_SETTLE_WINDOW_SECONDS
    min_wh_label_coverage: float = DEFAULT_MIN_WH_COVERAGE
    poll_interval_seconds: float = 300.0
    cycle_timeout_seconds: float = 25.0
    scoring_policy: ScoringPolicy = ScoringPolicy()


def write_fail_closed_outputs(
    output_dir: Path,
    *,
    health_state: str,
    model_state: str,
    error_type: str,
    error_detail: str | None = None,
    poll_interval_seconds: float = 300.0,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Atomically write fail-closed status, health, and report outputs so stale ready_to_use never persists."""
    wall_now = time.time() if now_ts is None else float(now_ts)
    output_dir.mkdir(parents=True, exist_ok=True)
    err_info = {
        "at": wall_now,
        "at_utc": _iso_utc(wall_now),
        "type": error_type,
        "detail": error_detail,
    }
    status = {
        "schema_version": 1,
        "shadow_only": True,
        "actuation_capable": False,
        "generated_at": wall_now,
        "generated_at_utc": _iso_utc(wall_now),
        "poll_interval_seconds": float(poll_interval_seconds),
        "state": health_state,
        "declined": True,
        "model": {
            "model_id": None,
            "model_state": model_state,
            "deployment_eligible": False,
            "model_arch_version": MODEL_ARCH_VERSION,
            "feature_version": FEATURE_VERSION,
            "constraint_version": CONSTRAINT_VERSION,
            "config_fingerprint": None,
            "training_cutoff_ts": None,
            "training_cutoff_utc": None,
            "parameter_count": None,
            "data_origin": None,
            "readiness_reasons": [health_state],
        },
        "counts": {
            "sealed_predictions": 0,
            "declined_predictions": 0,
            "settled_horizons": 0,
            "factual_settlement_kinds": {},
            "recommended_execution_statuses": {},
            "decline_reasons": {},
        },
        "factual_scores": {},
        "latest_sealed": None,
        "last_error": err_info,
    }
    health = {
        "at": wall_now,
        "at_utc": _iso_utc(wall_now),
        "state": health_state,
        "model_state": model_state,
        "declined": True,
        "poll_interval_seconds": float(poll_interval_seconds),
        "last_error": err_info,
    }
    report_md = "\n".join(
        [
            "# Neural-Network Thermostat Shadow Report",
            "",
            f"- Generated (UTC): `{_iso_utc(wall_now)}`",
            f"- Status: **{health_state}** (fail-closed; recommendations declined)",
            f"- Model state: `{model_state}` (`deployment_eligible=false`)",
            f"- Error type: `{error_type}`",
            "",
        ]
    )
    _atomic_write_text(
        output_dir / "neural-shadow-status.json",
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_write_text(
        output_dir / "neural-shadow-health.json",
        json.dumps(health, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_write_text(output_dir / "neural-shadow-report.md", report_md)
    return {"status": status, "health": health}


class ShadowRuntime:
    """Manages sealing, settling, reporting, and isolated execution for the neural shadow system."""

    def __init__(
        self,
        site: SiteConfig,
        runtime_cfg: ShadowRuntimeConfig,
        artifact: ModelArtifact | None = None,
    ):
        self.site = site
        self.cfg = runtime_cfg
        self.spec: FeatureSpec = build_feature_spec(
            site,
            history_minutes=runtime_cfg.history_minutes,
            step_minutes=runtime_cfg.step_minutes,
            horizons_minutes=runtime_cfg.horizons_minutes,
        )
        self.artifact = artifact if artifact is not None else build_untrained_artifact(site, self.spec)
        self.db_path = runtime_cfg.state_dir / "neural_shadow.sqlite3"
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        with closing(_init_shadow_db(self.db_path)):
            pass

    @classmethod
    def from_checkpoint_path(
        cls,
        site: SiteConfig,
        runtime_cfg: ShadowRuntimeConfig,
        checkpoint_path: Path | None,
        *,
        expected_arch_version: str = MODEL_ARCH_VERSION,
        expected_feature_version: str = FEATURE_VERSION,
    ) -> "ShadowRuntime":
        if expected_arch_version != MODEL_ARCH_VERSION:
            raise ModelVersionMismatchError(
                f"configured_arch_version_mismatch:{expected_arch_version}!={MODEL_ARCH_VERSION}"
            )
        if expected_feature_version != FEATURE_VERSION:
            raise ModelVersionMismatchError(
                f"configured_feature_version_mismatch:{expected_feature_version}!={FEATURE_VERSION}"
            )
        spec = build_feature_spec(
            site,
            history_minutes=runtime_cfg.history_minutes,
            step_minutes=runtime_cfg.step_minutes,
            horizons_minutes=runtime_cfg.horizons_minutes,
        )
        if checkpoint_path is None or not checkpoint_path.is_file():
            artifact = build_untrained_artifact(site, spec)
            artifact.metadata["readiness_reasons"] = ["checkpoint_file_missing"]
            return cls(site, runtime_cfg, artifact)
        # Strict validation: version/config mismatch raises ModelVersionMismatchError.
        artifact = load_checkpoint(
            checkpoint_path,
            expected_site=site,
            expected_spec=spec,
            expected_arch_version=expected_arch_version,
            expected_feature_version=expected_feature_version,
            strict=True,
        )
        return cls(site, runtime_cfg, artifact)

    def _validate_artifact_contract(self) -> list[str]:
        mismatches: list[str] = []
        meta = self.artifact.metadata
        if meta.get("model_arch_version") != MODEL_ARCH_VERSION:
            mismatches.append("model_arch_version_mismatch")
        if meta.get("feature_version") != self.spec.feature_version:
            mismatches.append("feature_version_mismatch")
        if meta.get("config_fingerprint") != self.site.config_fingerprint:
            mismatches.append("config_fingerprint_mismatch")
        if self.artifact.model.hist_dim != self.spec.hist_dim or self.artifact.model.act_dim != self.spec.act_dim:
            mismatches.append("feature_dim_mismatch")
        if self.artifact.model.n_zones != self.spec.n_zones or self.artifact.model.n_units != self.spec.n_units:
            mismatches.append("topology_dim_mismatch")
        return mismatches

    def seal_prediction(
        self,
        history: CanonicalHistory,
        cutoff_ts: float,
        *,
        sealed_at: float | None = None,
    ) -> dict[str, Any]:
        """Compute and immutably seal predictions at cutoff_ts BEFORE any label settlement."""
        mismatches = self._validate_artifact_contract()
        if mismatches:
            raise ModelVersionMismatchError("; ".join(mismatches))

        now_ts = time.time() if sealed_at is None else float(sealed_at)
        index = _HistoryIndex(history)
        x_hist, meta = extract_history_window(index, self.spec, cutoff_ts)
        candidates, constraint_rejections = generate_v1_candidates(self.site, self.spec, meta)
        ranking = evaluate_and_rank_candidates(
            self.artifact,
            self.site,
            self.spec,
            x_hist,
            meta,
            candidates,
            policy=self.cfg.scoring_policy,
        )

        pred_id = sha256(
            f"{cutoff_ts:.3f}|{self.artifact.model_id}|{self.site.config_fingerprint}|{self.spec.feature_version}".encode()
        ).hexdigest()[:24]

        true_report_times = {
            "zones": dict(meta["zone_report_at"]),
            "units_power": dict(meta["unit_power_report_at"]),
            "environment": dict(meta["env_report_at"]),
        }

        payload = {
            "prediction_id": pred_id,
            "sealed_at": now_ts,
            "sealed_at_utc": _iso_utc(now_ts),
            "cutoff_ts": cutoff_ts,
            "cutoff_utc": _iso_utc(cutoff_ts),
            "true_report_times": true_report_times,
            "model_id": self.artifact.model_id,
            "model_state": self.artifact.model_state.value,
            "model_arch_version": MODEL_ARCH_VERSION,
            "feature_version": self.spec.feature_version,
            "constraint_version": CONSTRAINT_VERSION,
            "config_fingerprint": self.site.config_fingerprint,
            "observation_version": self.site.observation_version,
            "training_cutoff_ts": self.artifact.training_cutoff_ts,
            "training_cutoff_utc": _iso_utc(self.artifact.training_cutoff_ts),
            "data_origin": history.data_origin,
            "model_data_origin": self.artifact.metadata.get("data_origin"),
            "applicability_scope": {
                **ranking["applicability_scope"],
                "zones": list(self.spec.zone_names),
                "units": list(self.spec.unit_names),
                "horizons_minutes": list(self.spec.horizons_minutes),
                "observation_version": self.site.observation_version,
            },
            "t0_state": {
                "zone_temps": meta["zone_temps"],
                "zone_targets": meta["zone_targets"],
                "zone_age_s": meta["zone_age_s"],
                "env_temps": meta["env_temps"],
                "unit_modes": meta["unit_modes"],
                "unit_setpoints": meta["unit_setpoints"],
                "unit_watts": meta["unit_watts"],
            },
            "candidates": ranking["candidate_scores"],
            "constraint_rejections": constraint_rejections,
            "peer_rejected": ranking.get("peer_rejected", []),
            "recommended_candidate_id": ranking["recommended_candidate_id"],
            "declined": bool(ranking["declined"]),
            "decline_reasons": list(ranking["decline_reasons"]),
            "selection_reason": ranking["selection_reason"],
        }
        seal_digest = canonical_hash(payload)
        payload["seal_sha256"] = seal_digest

        with closing(_init_shadow_db(self.db_path)) as db, db:
            existing = db.execute(
                "SELECT payload_json FROM sealed_predictions WHERE prediction_id=?", (pred_id,)
            ).fetchone()
            if existing is not None:
                # Immutability: never overwrite an already sealed prediction!
                return json.loads(existing["payload_json"])
            db.execute(
                """
                INSERT INTO sealed_predictions (
                    prediction_id, sealed_at, cutoff_ts, model_id, model_state,
                    model_arch_version, feature_version, config_fingerprint,
                    training_cutoff_ts, recommended_candidate_id, declined,
                    seal_sha256, payload_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    pred_id,
                    now_ts,
                    cutoff_ts,
                    self.artifact.model_id,
                    self.artifact.model_state.value,
                    MODEL_ARCH_VERSION,
                    self.spec.feature_version,
                    self.site.config_fingerprint,
                    self.artifact.training_cutoff_ts,
                    ranking["recommended_candidate_id"],
                    int(bool(ranking["declined"])),
                    seal_digest,
                    json.dumps(payload, sort_keys=True, ensure_ascii=False),
                ),
            )
        return payload

    def settle_predictions(
        self,
        history: CanonicalHistory,
        *,
        now_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """Settle all eligible horizons of previously sealed predictions without mutating sealed rows.

        Strictly separates factual hold scoring from counterfactual recommendations and
        mid-horizon intervened intervals.
        """
        index = _HistoryIndex(history)
        if now_ts is None:
            # Infer latest causally available timestamp in history
            cand_ts = [
                *(r.report_at for reps in history.zone_temp_reports.values() for r in reps),
                *(r.report_at for reps in history.power_reports.values() for r in reps),
                *(e.at for evs in history.proxy_states.values() for e in evs),
            ]
            now_ts = max(cand_ts) if cand_ts else time.time()

        newly_settled: list[dict[str, Any]] = []
        with closing(_init_shadow_db(self.db_path)) as db:
            sealed_rows = db.execute(
                "SELECT prediction_id, sealed_at, cutoff_ts, seal_sha256, payload_json FROM sealed_predictions ORDER BY cutoff_ts"
            ).fetchall()
            settled_keys = {
                (r["prediction_id"], int(r["horizon_minutes"]))
                for r in db.execute("SELECT prediction_id, horizon_minutes FROM settlements").fetchall()
            }

            for row in sealed_rows:
                pred_id = row["prediction_id"]
                payload = json.loads(row["payload_json"])
                # Verify seal integrity before settling
                saved_digest = payload.get("seal_sha256")
                verify_copy = {k: v for k, v in payload.items() if k != "seal_sha256"}
                if saved_digest != row["seal_sha256"] or canonical_hash(verify_copy) != saved_digest:
                    raise ValueError(f"Sealed prediction integrity check failed for {pred_id}")

                cutoff_ts = float(row["cutoff_ts"])
                t0_state = payload["t0_state"]
                t0_rep_times = payload["true_report_times"]["zones"]
                cand_map = {c["candidate"]["candidate_id"]: c for c in payload.get("candidates", [])}
                hold_entry = cand_map.get("hold")
                rec_id = payload.get("recommended_candidate_id")

                for h_min in self.spec.horizons_minutes:
                    if (pred_id, int(h_min)) in settled_keys:
                        continue
                    h_s = float(h_min * 60)
                    horizon_end = cutoff_ts + h_s
                    if now_ts < horizon_end + self.cfg.settle_window_seconds:
                        continue

                    # 1. Detect mid-horizon interventions (action, proxy mode/setpoint, or target change)
                    interventions: list[dict[str, Any]] = []
                    for act in index.actions_in_window(cutoff_ts, horizon_end):
                        interventions.append(
                            {
                                "type": "action",
                                "at": act.at,
                                "unit": act.unit,
                                "kind": act.kind,
                                "after_mode": act.after_mode,
                                "after_setpoint": act.after_setpoint,
                                "source": act.source,
                                "receipt_status": act.receipt_status,
                            }
                        )
                    for u in self.spec.unit_names:
                        m0 = t0_state["unit_modes"].get(u)
                        sp0 = t0_state["unit_setpoints"].get(u)
                        for ev in index.proxy_states[u]:
                            if cutoff_ts < ev.at <= horizon_end:
                                if ev.mode not in {None, "unavailable", "unknown"} and ev.mode != m0:
                                    interventions.append(
                                        {"type": "proxy_mode_change", "at": ev.at, "unit": u, "from": m0, "to": ev.mode}
                                    )
                                    break
                                if (
                                    ev.mode in {"heat", "cool"}
                                    and ev.setpoint is not None
                                    and sp0 is not None
                                    and abs(ev.setpoint - sp0) > 1e-4
                                ):
                                    interventions.append(
                                        {
                                            "type": "proxy_setpoint_change",
                                            "at": ev.at,
                                            "unit": u,
                                            "from": sp0,
                                            "to": ev.setpoint,
                                        }
                                    )
                                    break
                    for z in self.spec.zone_names:
                        tgt0 = t0_state["zone_targets"].get(z)
                        tgt_end = index.zone_target_at(z, horizon_end)
                        if tgt0 is not None and tgt_end is not None and abs(tgt_end - tgt0) > 1e-4:
                            interventions.append(
                                {"type": "target_change", "at": horizon_end, "zone": z, "from": tgt0, "to": tgt_end}
                            )

                    intervened = len(interventions) > 0

                    # 2. Classify factual vs counterfactual execution status
                    if rec_id is None:
                        rec_exec_status = "declined"
                    elif rec_id == "hold":
                        rec_exec_status = "intervened_hold_unscored" if intervened else "factual_hold"
                    else:
                        # Non-hold recommendation in shadow mode is NOT executed by shadow.
                        # Check if live controller coincidentally executed exact same action right at t0 with no other changes.
                        rec_cand = cand_map.get(rec_id, {}).get("candidate")
                        coincident = False
                        if rec_cand and len(interventions) >= 1:
                            action_events = [iv for iv in interventions if iv["type"] == "action"]
                            target_events = [iv for iv in interventions if iv["type"] == "target_change"]
                            if (
                                len(action_events) == 1
                                and not target_events
                                and action_events[0]["unit"] == rec_cand["unit"]
                                and action_events[0]["after_mode"] == rec_cand["mode"]
                                and action_events[0]["after_setpoint"] is not None
                                and abs(action_events[0]["after_setpoint"] - rec_cand["to_setpoint"]) <= 1e-4
                                and action_events[0]["at"] <= cutoff_ts + 120.0
                            ):
                                # Also check no other unit changed mode/setpoint
                                other_changes = [
                                    iv
                                    for iv in interventions
                                    if iv.get("unit") not in {None, rec_cand["unit"]}
                                ]
                                coincident = len(other_changes) == 0
                        rec_exec_status = (
                            "factual_coincident_recommendation" if coincident else "counterfactual_unexecuted"
                        )

                    factual_kind = "intervened_hold_unscored" if intervened else "factual_hold"

                    # 3. Collect actual future observations (strictly newer temp report + exact power coverage)
                    h_key = f"{h_min}m"
                    actual_zones: dict[str, Any] = {}
                    hold_temp_errors: dict[str, Any] = {}
                    for z in self.spec.zone_names:
                        t0_temp = t0_state["zone_temps"].get(z)
                        t0_rep = t0_rep_times.get(z)
                        fut = index.settle_future_temp(
                            z, cutoff_ts, t0_rep, h_s, settle_window_s=self.cfg.settle_window_seconds
                        )
                        actual_delta = (fut.value_c - t0_temp) if (fut is not None and t0_temp is not None) else None
                        actual_zones[z] = {
                            "t0_temp_c": t0_temp,
                            "t0_report_at": t0_rep,
                            "future_temp_c": fut.value_c if fut is not None else None,
                            "future_report_at": fut.report_at if fut is not None else None,
                            "actual_delta_c": round(actual_delta, 4) if actual_delta is not None else None,
                            "label_valid": actual_delta is not None,
                        }
                        # ONLY score factual hold prediction when NOT intervened!
                        if not intervened and hold_entry is not None and actual_delta is not None:
                            zpred = (hold_entry["prediction"]["zones"] or {}).get(z, {})
                            p_delta = (zpred.get("pred_delta_c") or {}).get(h_key)
                            p_sigma = (zpred.get("pred_sigma_c") or {}).get(h_key)
                            if p_delta is not None and p_sigma is not None:
                                err = float(p_delta) - float(actual_delta)
                                hold_temp_errors[z] = {
                                    "pred_delta_c": round(float(p_delta), 4),
                                    "actual_delta_c": round(float(actual_delta), 4),
                                    "signed_error_c": round(err, 4),
                                    "abs_error_c": round(abs(err), 4),
                                    "sigma_c": round(float(p_sigma), 4),
                                    "within_pi80": bool(abs(err) <= Z_80 * float(p_sigma) + 1e-9),
                                }

                    # Power integration across [cutoff_ts, horizon_end]: require complete coverage
                    # (has_gap == False) so missing metering intervals are NEVER scored as 0 Wh.
                    effective_min_cov = max(float(self.cfg.min_wh_label_coverage), 1.0 - 1e-6)
                    actual_units_wh: dict[str, Any] = {}
                    all_units_valid = True
                    tot_wh_sum = 0.0
                    min_cov = 1.0
                    for u in self.spec.unit_names:
                        act_map = self.site.actuator_by_name(u)
                        pslice = index.power_slice(u, cutoff_ts, horizon_end)
                        pint = integrate_power_interval(
                            pslice,
                            cutoff_ts,
                            horizon_end,
                            primary_entity=act_map.power_entity,
                            fallback_entity=act_map.power_fallback,
                        )
                        full_wh = pint.full_interval_wh
                        u_valid = full_wh is not None and pint.coverage_ratio >= effective_min_cov
                        min_cov = min(min_cov, pint.coverage_ratio)
                        if u_valid:
                            assert full_wh is not None
                            tot_wh_sum += full_wh
                        else:
                            all_units_valid = False
                        actual_units_wh[u] = {
                            "wh": round(full_wh, 3) if u_valid and full_wh is not None else None,
                            "covered_wh_raw": round(pint.wh, 3) if pint.wh is not None else None,
                            "coverage_ratio": round(pint.coverage_ratio, 4),
                            "gap_seconds": round(pint.gap_seconds, 1),
                            "has_gap": pint.has_gap,
                            "label_valid": u_valid,
                            "covered_seconds_by_source": {k: round(v, 1) for k, v in pint.source_covered_seconds},
                        }

                    actual_total_wh = round(tot_wh_sum, 3) if all_units_valid else None
                    hold_wh_error: dict[str, Any] | None = None
                    if not intervened and hold_entry is not None and actual_total_wh is not None:
                        p_wh = (hold_entry["prediction"]["total_wh"] or {}).get(h_key)
                        p_wh_sig = (hold_entry["prediction"]["total_wh_sigma"] or {}).get(h_key)
                        if p_wh is not None and p_wh_sig is not None:
                            w_err = float(p_wh) - float(actual_total_wh)
                            hold_wh_error = {
                                "pred_total_wh": round(float(p_wh), 2),
                                "actual_total_wh": actual_total_wh,
                                "signed_error_wh": round(w_err, 2),
                                "abs_error_wh": round(abs(w_err), 2),
                                "sigma_wh": round(float(p_wh_sig), 2),
                                "within_pi80": bool(abs(w_err) <= Z_80 * float(p_wh_sig) + 1e-9),
                            }

                    settlement_record = {
                        "prediction_id": pred_id,
                        "horizon_minutes": int(h_min),
                        "settled_at": now_ts,
                        "settled_at_utc": _iso_utc(now_ts),
                        "cutoff_ts": cutoff_ts,
                        "horizon_end_ts": horizon_end,
                        "model_id": payload["model_id"],
                        "model_state": payload["model_state"],
                        "model_arch_version": payload["model_arch_version"],
                        "feature_version": payload["feature_version"],
                        "config_fingerprint": payload["config_fingerprint"],
                        "training_cutoff_ts": payload["training_cutoff_ts"],
                        "intervened": intervened,
                        "intervention_events": interventions,
                        "factual_settlement_kind": factual_kind,
                        "recommended_candidate_id": rec_id,
                        "recommended_execution_status": rec_exec_status,
                        "actual_zones": actual_zones,
                        "factual_hold_temp_errors": hold_temp_errors,
                        "actual_units_wh": actual_units_wh,
                        "actual_total_wh": actual_total_wh,
                        "joint_wh_coverage_ratio": round(min_cov, 4),
                        "joint_wh_has_gap": min_cov < 1.0 - 1e-6,
                        "joint_wh_label_valid": all_units_valid,
                        "factual_hold_wh_error": hold_wh_error,
                    }
                    with db:
                        db.execute(
                            """
                            INSERT OR IGNORE INTO settlements (
                                prediction_id, horizon_minutes, settled_at, cutoff_ts,
                                horizon_end_ts, intervened, factual_settlement_kind,
                                recommended_execution_status, payload_json
                            ) VALUES (?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                pred_id,
                                int(h_min),
                                now_ts,
                                cutoff_ts,
                                horizon_end,
                                int(intervened),
                                factual_kind,
                                rec_exec_status,
                                json.dumps(settlement_record, sort_keys=True, ensure_ascii=False),
                            ),
                        )
                    newly_settled.append(settlement_record)
        return newly_settled

    def build_status_and_reports(self, *, now_ts: float | None = None, last_error: dict[str, Any] | None = None) -> dict[str, Any]:
        """Aggregate sealed predictions & settled factual scores and write independent outputs atomically."""
        now_ts = time.time() if now_ts is None else float(now_ts)
        with closing(_init_shadow_db(self.db_path)) as db:
            sealed = [
                json.loads(r["payload_json"])
                for r in db.execute("SELECT payload_json FROM sealed_predictions ORDER BY cutoff_ts").fetchall()
            ]
            settled = [
                json.loads(r["payload_json"])
                for r in db.execute("SELECT payload_json FROM settlements ORDER BY horizon_end_ts, horizon_minutes").fetchall()
            ]

        decline_counts = Counter()
        for p in sealed:
            if p.get("declined"):
                for r in p.get("decline_reasons", []):
                    decline_counts[r] += 1

        kind_counts = Counter(s["factual_settlement_kind"] for s in settled)
        exec_counts = Counter(s["recommended_execution_status"] for s in settled)

        # Aggregate strictly FACTUAL (non-intervened) hold scores by zone and horizon
        factual_by_horizon: dict[str, Any] = {}
        for h_min in self.spec.horizons_minutes:
            h_rows = [s for s in settled if s["horizon_minutes"] == h_min]
            factual_rows = [s for s in h_rows if s["factual_settlement_kind"] == "factual_hold"]
            intervened_rows = [s for s in h_rows if s["intervened"]]
            zone_stats: dict[str, Any] = {}
            for z in self.spec.zone_names:
                z_errs = [
                    r["factual_hold_temp_errors"][z]
                    for r in factual_rows
                    if z in r.get("factual_hold_temp_errors", {})
                ]
                if z_errs:
                    abs_e = [e["abs_error_c"] for e in z_errs]
                    sgn_e = [e["signed_error_c"] for e in z_errs]
                    cov_e = [1.0 if e["within_pi80"] else 0.0 for e in z_errs]
                    zone_stats[z] = {
                        "n_factual": len(z_errs),
                        "mae_c": round(float(np.mean(abs_e)), 4),
                        "bias_c": round(float(np.mean(sgn_e)), 4),
                        "p90_abs_err_c": round(float(np.quantile(abs_e, 0.90)), 4),
                        "pi80_coverage": round(float(np.mean(cov_e)), 4),
                    }
                else:
                    zone_stats[z] = {"n_factual": 0, "mae_c": None, "bias_c": None, "p90_abs_err_c": None, "pi80_coverage": None}

            wh_errs = [r["factual_hold_wh_error"] for r in factual_rows if r.get("factual_hold_wh_error") is not None]
            gap_count = sum(1 for r in h_rows if r.get("joint_wh_has_gap"))
            cov_ratios = [r["joint_wh_coverage_ratio"] for r in h_rows if r.get("joint_wh_coverage_ratio") is not None]
            factual_by_horizon[f"{h_min}m"] = {
                "n_settled_total": len(h_rows),
                "n_factual_hold": len(factual_rows),
                "n_intervened_excluded": len(intervened_rows),
                "zones": zone_stats,
                "total_wh": {
                    "n_factual": len(wh_errs),
                    "mae_wh": round(float(np.mean([e["abs_error_wh"] for e in wh_errs])), 2) if wh_errs else None,
                    "bias_wh": round(float(np.mean([e["signed_error_wh"] for e in wh_errs])), 2) if wh_errs else None,
                    "pi80_coverage": round(float(np.mean([1.0 if e["within_pi80"] else 0.0 for e in wh_errs])), 4)
                    if wh_errs
                    else None,
                    "mean_joint_coverage_ratio": round(float(np.mean(cov_ratios)), 4) if cov_ratios else None,
                    "coverage_gap_count": gap_count,
                },
            }

        latest_sealed = sealed[-1] if sealed else None
        status = {
            "schema_version": 1,
            "shadow_only": True,
            "actuation_capable": False,
            "generated_at": now_ts,
            "generated_at_utc": _iso_utc(now_ts),
            "poll_interval_seconds": float(self.cfg.poll_interval_seconds),
            "state": "degraded" if last_error is not None else "healthy",
            "declined": bool(latest_sealed["declined"]) if latest_sealed is not None else True,
            "site": self.site.summary_dict(),
            "model": {
                "model_id": self.artifact.model_id,
                "model_state": self.artifact.model_state.value,
                "deployment_eligible": bool(self.artifact.metadata.get("deployment_eligible", False)),
                "model_arch_version": MODEL_ARCH_VERSION,
                "feature_version": self.spec.feature_version,
                "constraint_version": CONSTRAINT_VERSION,
                "config_fingerprint": self.site.config_fingerprint,
                "training_cutoff_ts": self.artifact.training_cutoff_ts,
                "training_cutoff_utc": _iso_utc(self.artifact.training_cutoff_ts),
                "parameter_count": self.artifact.model.parameter_count(),
                "data_origin": self.artifact.metadata.get("data_origin"),
                "readiness_reasons": self.artifact.metadata.get("readiness_reasons", []),
            },
            "counts": {
                "sealed_predictions": len(sealed),
                "declined_predictions": sum(1 for p in sealed if p.get("declined")),
                "settled_horizons": len(settled),
                "factual_settlement_kinds": dict(kind_counts),
                "recommended_execution_statuses": dict(exec_counts),
                "decline_reasons": dict(decline_counts),
            },
            "factual_scores": factual_by_horizon,
            "latest_sealed": {
                "prediction_id": latest_sealed["prediction_id"],
                "cutoff_utc": latest_sealed["cutoff_utc"],
                "sealed_at_utc": latest_sealed["sealed_at_utc"],
                "true_report_times": latest_sealed["true_report_times"],
                "recommended_candidate_id": latest_sealed["recommended_candidate_id"],
                "declined": latest_sealed["declined"],
                "decline_reasons": latest_sealed["decline_reasons"],
                "selection_reason": latest_sealed["selection_reason"],
                "seal_sha256": latest_sealed["seal_sha256"],
            }
            if latest_sealed
            else None,
            "last_error": last_error,
        }

        _atomic_write_text(
            self.cfg.output_dir / "neural-shadow-status.json",
            json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        )
        scores_lines = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in settled)
        _atomic_write_text(self.cfg.output_dir / "neural-shadow-scores.jsonl", scores_lines)

        health = {
            "at": now_ts,
            "at_utc": _iso_utc(now_ts),
            "state": "degraded" if last_error is not None else "healthy",
            "model_state": self.artifact.model_state.value,
            "declined": bool(latest_sealed["declined"]) if latest_sealed is not None else True,
            "poll_interval_seconds": float(self.cfg.poll_interval_seconds),
            "sealed_predictions": len(sealed),
            "settled_horizons": len(settled),
            "latest_cutoff_utc": latest_sealed["cutoff_utc"] if latest_sealed else None,
            "last_error": last_error,
        }
        _atomic_write_text(
            self.cfg.output_dir / "neural-shadow-health.json",
            json.dumps(health, indent=2, ensure_ascii=False) + "\n",
        )
        _atomic_write_text(self.cfg.output_dir / "neural-shadow-report.md", self._render_markdown(status))
        return status

    def _render_markdown(self, status: dict[str, Any]) -> str:
        m = status["model"]
        c = status["counts"]
        lines = [
            "# Neural-Network Thermostat Shadow Report",
            "",
            f"- Generated (UTC): `{status['generated_at_utc']}`",
            f"- Mode: **Shadow-only (zero actuation path)**",
            f"- Model ID: `{m['model_id']}` (`{m['model_state']}`, params: `{m['parameter_count']}`)",
            f"- Versions: arch=`{m['model_arch_version']}`, feature=`{m['feature_version']}`, constraint=`{m['constraint_version']}`",
            f"- Training cutoff (UTC): `{m['training_cutoff_utc']}` | Data origin: `{m['data_origin']}`",
            f"- Sealed predictions: `{c['sealed_predictions']}` (declined: `{c['declined_predictions']}`) | Settled horizons: `{c['settled_horizons']}`",
            "",
            "## Factual Hold Scores (Non-Intervened Intervals Only)",
            "",
            "| Horizon | Settled | Factual | Intervened (Excluded) | Zone Metrics (MAE / Bias / PI80) | Total Wh MAE (Coverage / Gaps) |",
            "|---|---:|---:|---:|---|---|",
        ]
        for h_key, h_val in status["factual_scores"].items():
            z_parts = []
            for z, zs in h_val["zones"].items():
                if zs["n_factual"] > 0:
                    z_parts.append(f"{z}: {zs['mae_c']}C / {zs['bias_c']:+.3f}C / {zs['pi80_coverage']:.0%}")
                else:
                    z_parts.append(f"{z}: n/a")
            wh = h_val["total_wh"]
            wh_str = (
                f"{wh['mae_wh']} Wh (cov={wh['mean_joint_coverage_ratio']}, gaps={wh['coverage_gap_count']})"
                if wh["mae_wh"] is not None
                else f"n/a (cov={wh['mean_joint_coverage_ratio']}, gaps={wh['coverage_gap_count']})"
            )
            lines.append(
                f"| {h_key} | {h_val['n_settled_total']} | {h_val['n_factual_hold']} | "
                f"{h_val['n_intervened_excluded']} | {'; '.join(z_parts)} | {wh_str} |"
            )
        lines.extend(
            [
                "",
                "Counterfactual recommendations (actions not executed on physical equipment) are tracked separately and never scored as factual hold trajectories.",
                "",
            ]
        )
        return "\n".join(lines)

    def run_isolated_cycle(
        self,
        history_loader,
        *,
        cutoff_ts: float | None = None,
        now_ts: float | None = None,
    ) -> dict[str, Any]:
        """Run one seal -> settle -> report cycle inside a strict timeout and exception boundary.

        Failures are captured in neural-shadow-health.json / neural-shadow-status.json and
        NEVER propagate to crash or disturb the existing thermostat controller.
        """
        wall_now = time.time() if now_ts is None else float(now_ts)
        use_alarm = (
            hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread()
            and self.cfg.cycle_timeout_seconds > 0
        )
        prev_handler = None
        if use_alarm:
            def _on_timeout(*_):
                raise ShadowCycleTimeout(f"Shadow cycle exceeded {self.cfg.cycle_timeout_seconds}s")

            prev_handler = signal.signal(signal.SIGALRM, _on_timeout)
            signal.setitimer(signal.ITIMER_REAL, float(self.cfg.cycle_timeout_seconds))

        try:
            history: CanonicalHistory = history_loader()
            if cutoff_ts is None:
                step_s = float(self.spec.step_minutes * 60)
                cutoff_ts = math.floor(wall_now / step_s) * step_s
            sealed = self.seal_prediction(history, cutoff_ts, sealed_at=wall_now)
            settled = self.settle_predictions(history, now_ts=wall_now)
            status = self.build_status_and_reports(now_ts=wall_now, last_error=None)
            return {
                "ok": True,
                "sealed_prediction_id": sealed["prediction_id"],
                "newly_settled_count": len(settled),
                "status": status,
            }
        except Exception as exc:
            frames = [
                {"file": Path(f.filename).name, "function": f.name, "line": f.lineno}
                for f in extract_tb(exc.__traceback__)
            ]
            err_info = {
                "at": wall_now,
                "at_utc": _iso_utc(wall_now),
                "type": type(exc).__name__,
                "stack": frames[-6:],
            }
            if isinstance(exc, ModelVersionMismatchError):
                try:
                    written = write_fail_closed_outputs(
                        self.cfg.output_dir,
                        health_state="version_mismatch",
                        model_state="version_mismatch",
                        error_type=type(exc).__name__,
                        error_detail=str(exc),
                        poll_interval_seconds=self.cfg.poll_interval_seconds,
                        now_ts=wall_now,
                    )
                    status = written["status"]
                except OSError:
                    status = None
                return {"ok": False, "error": err_info, "status": status}
            try:
                status = self.build_status_and_reports(now_ts=wall_now, last_error=err_info)
            except Exception:
                # Even if SQLite/report fails, fail closed in BOTH status.json and health.json.
                status = None
                try:
                    written = write_fail_closed_outputs(
                        self.cfg.output_dir,
                        health_state="error",
                        model_state="unavailable",
                        error_type=type(exc).__name__,
                        error_detail=None,
                        poll_interval_seconds=self.cfg.poll_interval_seconds,
                        now_ts=wall_now,
                    )
                    status = written["status"]
                except OSError:
                    pass
            return {"ok": False, "error": err_info, "status": status}
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                if prev_handler is not None:
                    signal.signal(signal.SIGALRM, prev_handler)
