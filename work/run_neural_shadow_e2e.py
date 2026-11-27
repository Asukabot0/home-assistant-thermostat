"""Runnable local end-to-end example for the neural-network thermostat shadow system.

Pipeline:
  data build -> train/load model -> seal predictions -> settle labels -> report

NOTE:
- When invoked with no external data path, this script generates SYNTHETIC verification
  data (data_origin="synthetic_verification_only") strictly to prove the pipeline runs
  end to end. Synthetic metrics MUST NEVER be claimed as real-house predictive accuracy
  or measured energy savings.
- Pass --production <dir> or --archives <zip...> to run against real read-only history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "home-assistant"))

from neural_shadow import DataOrigin  # noqa: E402
from neural_shadow.config import load_default_example_config  # noqa: E402
from neural_shadow.data_pipeline import (  # noqa: E402
    build_dataset,
    ingest_evidence_archives,
    ingest_production_readonly,
    save_dataset,
)
from neural_shadow.nn_model import load_checkpoint  # noqa: E402
from neural_shadow.runtime import ShadowRuntime, ShadowRuntimeConfig  # noqa: E402
from neural_shadow.synthetic import generate_synthetic_history  # noqa: E402
from neural_shadow.train import ReadinessGates, forward_chain_split_dataset, train_and_evaluate  # noqa: E402


def run_e2e(
    work_dir: Path,
    *,
    production: Path | None = None,
    archives: list[Path] | None = None,
    synthetic_hours: float = 96.0,
    epochs: int = 25,
    seed: int = 42,
) -> dict:
    # 1. Data ingestion & repeatable data build
    if production is not None:
        history = ingest_production_readonly(production, data_origin=DataOrigin.REAL.value)
        mode_note = f"REAL production read-only history from {production}"
    elif archives:
        history = ingest_evidence_archives(archives, data_origin=DataOrigin.REAL_SHORT_PACK.value)
        mode_note = f"REAL short evidence archive(s): {[p.name for p in archives]} (NOT complete history)"
    else:
        site = load_default_example_config()
        history = generate_synthetic_history(site, hours=synthetic_hours, seed=seed)
        mode_note = f"SYNTHETIC verification data ({synthetic_hours}h, seed={seed}) — verifies code only, NOT real accuracy"

    dataset = build_dataset(history)
    dataset_dir = save_dataset(dataset, work_dir / "dataset")

    # 2. Forward-chaining train, calibrate on validation, evaluate against all 3 baselines
    model_dir = work_dir / "model"
    gates = ReadinessGates(allow_synthetic_ready=False)
    artifact, eval_report = train_and_evaluate(
        dataset,
        output_dir=model_dir,
        seed=seed,
        epochs=epochs,
        gates=gates,
    )

    # 3. Reload checkpoint cleanly with strict version checks
    ckpt_path = model_dir / "thermostat_shadow_gru.pt"
    loaded = load_checkpoint(ckpt_path, expected_site=dataset.site, expected_spec=dataset.spec, strict=True)

    # 4. Seal predictions BEFORE settlement on held-out test window, then settle labels & write report
    rt_cfg = ShadowRuntimeConfig(state_dir=work_dir / "state", output_dir=work_dir / "output")
    runtime = ShadowRuntime(dataset.site, rt_cfg, loaded)

    split = forward_chain_split_dataset(dataset)
    seal_indices = split.test_idx[::3][:12] if split.test_idx.size > 0 else split.val_idx[:6]
    sealed_ids = []
    for idx in seal_indices:
        cutoff = float(dataset.cutoff_ts[int(idx)])
        # Seal at cutoff + 1s (strictly before future horizon settlement!)
        sealed = runtime.seal_prediction(history, cutoff, sealed_at=cutoff + 1.0)
        sealed_ids.append(sealed["prediction_id"])

    # Settle after test window closes
    final_ts = float(dataset.manifest["time_range_ts"]["to"]) + 600.0
    settled = runtime.settle_predictions(history, now_ts=final_ts)
    status = runtime.build_status_and_reports(now_ts=final_ts)

    test_replay = (eval_report.get("test") or {}).get("historical_replay_action_conditioned") or {}
    comparison_60m = {
        name: (metrics.get("temp_by_horizon") or {}).get("60m", {}).get("mae")
        for name, metrics in test_replay.items()
    }
    wh_comparison_60m = {
        name: (metrics.get("wh_total") or {}).get("60m", {}).get("mae")
        for name, metrics in test_replay.items()
    }

    summary = {
        "data_mode_disclosure": mode_note,
        "data_origin": dataset.manifest["data_origin"],
        "dataset_dir": str(dataset_dir),
        "model_dir": str(model_dir),
        "output_dir": str(work_dir / "output"),
        "parameter_count": loaded.model.parameter_count(),
        "model_state": loaded.model_state.value,
        "deployment_eligible": loaded.metadata.get("deployment_eligible", False),
        "readiness_reasons": loaded.metadata.get("readiness_reasons"),
        "split_sizes": eval_report.get("split_sizes"),
        "test_replay_60m_temp_mae_c": comparison_60m,
        "test_replay_60m_total_wh_mae": wh_comparison_60m,
        "action_response_audit": {
            "action_coverage_sufficient": (eval_report.get("action_response_audit") or {}).get("action_coverage_sufficient"),
            "action_response_passed": (eval_report.get("action_response_audit") or {}).get("action_response_passed"),
        },
        "sealed_count": len(sealed_ids),
        "settled_horizon_count": len(settled),
        "factual_settlement_kinds": status["counts"]["factual_settlement_kinds"],
        "recommended_execution_statuses": status["counts"]["recommended_execution_statuses"],
        "report_files": sorted(p.name for p in (work_dir / "output").iterdir()),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=None, help="Directory for E2E outputs (default: temp dir)")
    parser.add_argument("--production", type=Path, default=None, help="Optional real production root (read-only)")
    parser.add_argument("--archives", nargs="+", type=Path, default=None, help="Optional real evidence ZIP archive(s)")
    parser.add_argument("--synthetic-hours", type=float, default=96.0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        res = run_e2e(
            args.work_dir,
            production=args.production,
            archives=args.archives,
            synthetic_hours=args.synthetic_hours,
            epochs=args.epochs,
            seed=args.seed,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    with TemporaryDirectory() as tmp:
        res = run_e2e(
            Path(tmp),
            production=args.production,
            archives=args.archives,
            synthetic_hours=args.synthetic_hours,
            epochs=args.epochs,
            seed=args.seed,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
