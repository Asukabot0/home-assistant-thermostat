"""Command-line interface for the independent neural-network thermostat shadow system.

Commands:
- export-history: Read-only export from production SQLite/journals or evidence archives -> canonical snapshot.
- data-build:     Repeatable causal dataset + manifest build from production, archives, snapshot, or synthetic.
- train:          Forward-chaining train + validation calibration + baseline evaluation + readiness gating.
- evaluate:       Offline evaluation of an existing checkpoint against baselines and action-response probes.
- seal:           Seal predictions at one or more cutoff timestamps (before label settlement).
- settle:         Settle eligible horizons of previously sealed predictions (factual vs counterfactual).
- report:         Render neural-shadow-status.json, neural-shadow-scores.jsonl, health, and markdown report.
- run-once:       Execute one isolated seal -> settle -> report cycle (uses config JSON or CLI paths).
- serve:          Continuous isolated daemon loop (intended for compose.neural-shadow.yaml).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import signal
from threading import Event
import time
from typing import Any

from . import (
    DEFAULT_HISTORY_MINUTES,
    DataOrigin,
    HORIZONS_MINUTES,
    STEP_MINUTES,
)
from .config import load_default_example_config, load_site_config_from_files
from .data_pipeline import (
    CanonicalHistory,
    build_dataset,
    ingest_canonical_snapshot,
    ingest_evidence_archives,
    ingest_production_readonly,
    load_dataset,
    save_canonical_snapshot,
    save_dataset,
)
from .evaluate import evaluate_all_models
from .nn_model import ModelVersionMismatchError, load_checkpoint
from .runtime import ShadowRuntime, ShadowRuntimeConfig
from .scoring import ScoringPolicy
from .synthetic import generate_synthetic_history
from .train import ReadinessGates, forward_chain_split, train_and_evaluate

LOG = logging.getLogger("neural-shadow")


def _load_history_from_args(args: argparse.Namespace) -> CanonicalHistory:
    if getattr(args, "production", None):
        return ingest_production_readonly(
            Path(args.production),
            rooms_file=Path(args.rooms_file) if getattr(args, "rooms_file", None) else None,
            topology_file=Path(args.topology_file) if getattr(args, "topology_file", None) else None,
            policy_file=Path(args.policy_file) if getattr(args, "policy_file", None) else None,
            start_ts=getattr(args, "start_ts", None),
            end_ts=getattr(args, "end_ts", None),
            data_origin=DataOrigin.REAL.value,
        )
    if getattr(args, "archives", None):
        paths = [Path(p) for p in args.archives]
        return ingest_evidence_archives(paths, data_origin=DataOrigin.REAL_SHORT_PACK.value)
    if getattr(args, "snapshot", None):
        return ingest_canonical_snapshot(Path(args.snapshot))
    if getattr(args, "synthetic_hours", None):
        site = (
            load_site_config_from_files(Path(args.rooms_file), Path(args.topology_file), Path(args.policy_file) if args.policy_file else None)
            if getattr(args, "rooms_file", None) and getattr(args, "topology_file", None)
            else load_default_example_config()
        )
        return generate_synthetic_history(site, hours=float(args.synthetic_hours), seed=int(args.seed))
    raise ValueError("Specify one input source: --production, --archives, --snapshot, or --synthetic-hours")


def _add_source_args(parser: argparse.ArgumentParser, *, allow_synthetic: bool = False) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--production", type=Path, help="Read-only production home-assistant root directory")
    group.add_argument("--archives", nargs="+", type=Path, help="Read-only telemetry or acceptance ZIP archive(s)")
    group.add_argument("--snapshot", type=Path, help="Read-only exported canonical snapshot directory")
    if allow_synthetic:
        group.add_argument(
            "--synthetic-hours",
            type=float,
            help="Generate synthetic verification data ONLY (never for production deployment claims)",
        )
    parser.add_argument("--rooms-file", type=Path, default=None, help="Optional override path to rooms.json")
    parser.add_argument("--topology-file", type=Path, default=None, help="Optional override path to topology.json")
    parser.add_argument("--policy-file", type=Path, default=None, help="Optional override path to house_policy.json")
    parser.add_argument("--start-ts", type=float, default=None, help="Optional UTC epoch lower bound")
    parser.add_argument("--end-ts", type=float, default=None, help="Optional UTC epoch upper bound")


def _cmd_export_history(args: argparse.Namespace) -> int:
    history = _load_history_from_args(args)
    out = save_canonical_snapshot(history, Path(args.output))
    summary = {
        "exported_dir": str(out),
        "data_origin": history.data_origin,
        "zones": list(history.site.zone_names),
        "units": list(history.site.unit_names),
        "unique_temp_reports": {z: len(r) for z, r in history.zone_temp_reports.items()},
        "unique_power_reports": {u: len(r) for u, r in history.power_reports.items()},
        "actions": len(history.actions),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_data_build(args: argparse.Namespace) -> int:
    history = _load_history_from_args(args)
    ds = build_dataset(
        history,
        history_minutes=int(args.history_minutes),
        step_minutes=int(args.step_minutes),
        horizons_minutes=tuple(int(h) for h in args.horizons),
        min_wh_coverage=float(args.min_wh_coverage),
    )
    out = save_dataset(ds, Path(args.output))
    print(json.dumps({"dataset_dir": str(out), "manifest": ds.manifest}, indent=2, ensure_ascii=False))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    ds = load_dataset(Path(args.dataset))
    gates = ReadinessGates(
        min_real_history_hours=float(args.min_real_hours),
        min_train_samples=int(args.min_train_samples),
        min_val_samples=int(args.min_val_samples),
        min_test_samples=int(args.min_test_samples),
        allow_synthetic_ready=False,
    )
    artifact, report = train_and_evaluate(
        ds,
        output_dir=Path(args.output),
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        patience=int(args.patience),
        gates=gates,
    )
    summary = {
        "model_dir": str(args.output),
        "model_id": artifact.model_id,
        "model_state": artifact.model_state.value,
        "deployment_eligible": artifact.metadata.get("deployment_eligible", False),
        "parameter_count": artifact.model.parameter_count(),
        "data_origin": ds.manifest.get("data_origin"),
        "training_cutoff_utc": artifact.metadata.get("training_cutoff_utc"),
        "readiness_reasons": artifact.metadata.get("readiness_reasons"),
        "split_sizes": report.get("split_sizes"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    ds = load_dataset(Path(args.dataset))
    artifact = load_checkpoint(Path(args.checkpoint), expected_site=ds.site, expected_spec=ds.spec, strict=True)
    split = forward_chain_split(ds.cutoff_ts, max_horizon_seconds=float(max(ds.spec.horizons_minutes) * 60))
    report = evaluate_all_models(artifact, ds, split.train_idx, split.val_idx, split.test_idx)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _build_runtime_from_config(cfg_path: Path) -> tuple[ShadowRuntime, Any]:
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    inp = raw["inputs"]
    mod = raw["model"]
    out = raw["outputs"]
    rt = raw.get("runtime", {})
    site = load_site_config_from_files(
        Path(inp["rooms_file"]),
        Path(inp["topology_file"]),
        Path(inp["policy_file"]) if inp.get("policy_file") else None,
    )
    scoring_policy = ScoringPolicy(
        max_uncertainty_c=float(rt.get("max_uncertainty_c", 0.80)),
        ood_temp_margin_c=float(rt.get("ood_temp_margin_c", 3.0)),
        ood_outdoor_margin_c=float(rt.get("ood_outdoor_margin_c", 5.0)),
    )
    rcfg = ShadowRuntimeConfig(
        state_dir=Path(out["state_dir"]),
        output_dir=Path(out["output_dir"]),
        history_minutes=int(rt.get("history_minutes", DEFAULT_HISTORY_MINUTES)),
        step_minutes=int(rt.get("step_minutes", STEP_MINUTES)),
        horizons_minutes=tuple(int(h) for h in rt.get("horizons_minutes", HORIZONS_MINUTES)),
        settle_window_seconds=float(rt.get("settle_window_seconds", 300.0)),
        min_wh_label_coverage=float(rt.get("min_wh_label_coverage", 0.85)),
        cycle_timeout_seconds=float(rt.get("cycle_timeout_seconds", 25.0)),
        scoring_policy=scoring_policy,
    )
    ckpt = Path(mod["artifact_dir"]) / mod["checkpoint_name"]
    runtime = ShadowRuntime.from_checkpoint_path(site, rcfg, ckpt if ckpt.is_file() else None)
    return runtime, raw


def _make_production_loader(raw_cfg: dict[str, Any]):
    inp = raw_cfg["inputs"]
    ha_dir = Path(inp["ha_config_dir"])
    shadow_data_dir = Path(inp["shadow_data_dir"])
    pilot_data_dir = Path(inp["pilot_data_dir"])
    rooms_file = Path(inp["rooms_file"])
    topology_file = Path(inp["topology_file"])
    policy_file = Path(inp["policy_file"]) if inp.get("policy_file") else None

    def _loader() -> CanonicalHistory:
        # Support either standard root layout or container mount layout (/ha-readonly, /shadow-data-ro, /pilot-data-ro)
        from .data_pipeline import ingest_production_readonly

        # If all mounts sit under a single production root, use it directly; else read explicit mount paths.
        if (ha_dir.parent / "shadow-data").resolve() == shadow_data_dir.resolve() and (
            ha_dir.parent / "pilot-data"
        ).resolve() == pilot_data_dir.resolve():
            return ingest_production_readonly(
                ha_dir.parent,
                rooms_file=rooms_file,
                topology_file=topology_file,
                policy_file=policy_file,
                start_ts=time.time() - 6 * 3600,
                end_ts=time.time() + 60,
            )
        # Assemble virtual read-only view via symlinks in a temp dir inside /tmp (zero writes to mounted inputs!)
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            vroot = Path(tmp)
            (vroot / "config").symlink_to(ha_dir)
            (vroot / "shadow-data").symlink_to(shadow_data_dir)
            (vroot / "pilot-data").symlink_to(pilot_data_dir)
            return ingest_production_readonly(
                vroot,
                rooms_file=rooms_file,
                topology_file=topology_file,
                policy_file=policy_file,
                start_ts=time.time() - 6 * 3600,
                end_ts=time.time() + 60,
            )

    return _loader


def _cmd_seal_settle_report(args: argparse.Namespace) -> int:
    history = _load_history_from_args(args)
    rcfg = ShadowRuntimeConfig(state_dir=Path(args.state_dir), output_dir=Path(args.output_dir))
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    runtime = ShadowRuntime.from_checkpoint_path(history.site, rcfg, ckpt)

    if args.command == "seal":
        if args.cutoff_ts is not None:
            cutoffs = [float(args.cutoff_ts)]
        else:
            ts_all = [r.report_at for reps in history.zone_temp_reports.values() for r in reps]
            cutoffs = [max(ts_all)] if ts_all else [time.time()]
        sealed_ids = [runtime.seal_prediction(history, c)["prediction_id"] for c in cutoffs]
        print(json.dumps({"sealed_prediction_ids": sealed_ids, "model_state": runtime.artifact.model_state.value}, indent=2))
        return 0

    if args.command == "settle":
        settled = runtime.settle_predictions(history, now_ts=args.now_ts)
        print(json.dumps({"newly_settled_count": len(settled)}, indent=2))
        return 0

    status = runtime.build_status_and_reports(now_ts=args.now_ts)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


def _cmd_run_once(args: argparse.Namespace) -> int:
    try:
        runtime, raw_cfg = _build_runtime_from_config(Path(args.config))
    except ModelVersionMismatchError as exc:
        # Fail closed cleanly without crashing container supervisor or disturbing live controller
        raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        out_dir = Path(raw["outputs"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        err_payload = {
            "at": time.time(),
            "state": "version_mismatch",
            "declined": True,
            "error": type(exc).__name__,
            "detail": str(exc),
        }
        (out_dir / "neural-shadow-health.json").write_text(json.dumps(err_payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": False, **err_payload}, indent=2))
        return 2

    loader = _make_production_loader(raw_cfg)
    result = runtime.run_isolated_cycle(loader)
    print(json.dumps({"ok": result["ok"], "error": result.get("error"), "sealed_prediction_id": result.get("sealed_prediction_id"), "newly_settled_count": result.get("newly_settled_count")}, indent=2))
    return 0 if result["ok"] else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    cfg_path = Path(args.config)
    LOG.info("Neural thermostat shadow service starting (read-only, zero actuation path)")
    while not stop.is_set():
        poll_s = 300.0
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            poll_s = float(raw.get("runtime", {}).get("poll_interval_seconds", 300.0))
            runtime, raw_cfg = _build_runtime_from_config(cfg_path)
            loader = _make_production_loader(raw_cfg)
            result = runtime.run_isolated_cycle(loader)
            if not result["ok"]:
                LOG.warning("Neural shadow cycle failed isolated: %s", (result.get("error") or {}).get("type"))
        except Exception as exc:
            LOG.warning("Neural shadow outer loop isolated error: %s", type(exc).__name__)
        stop.wait(max(10.0, poll_s))
    LOG.info("Neural thermostat shadow service stopped cleanly")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neural_shadow", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_exp = sub.add_parser("export-history", help="Export read-only history into canonical snapshot")
    _add_source_args(p_exp, allow_synthetic=False)
    p_exp.add_argument("--output", type=Path, required=True)
    p_exp.set_defaults(func=_cmd_export_history)

    p_build = sub.add_parser("data-build", help="Build causally aligned dataset and manifest")
    _add_source_args(p_build, allow_synthetic=True)
    p_build.add_argument("--output", type=Path, required=True)
    p_build.add_argument("--history-minutes", type=int, default=DEFAULT_HISTORY_MINUTES)
    p_build.add_argument("--step-minutes", type=int, default=STEP_MINUTES)
    p_build.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS_MINUTES))
    p_build.add_argument("--min-wh-coverage", type=float, default=0.85)
    p_build.add_argument("--seed", type=int, default=42)
    p_build.set_defaults(func=_cmd_data_build)

    p_train = sub.add_parser("train", help="Train, calibrate, evaluate, and gate neural shadow model")
    p_train.add_argument("--dataset", type=Path, required=True)
    p_train.add_argument("--output", type=Path, required=True)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--epochs", type=int, default=45)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1.5e-3)
    p_train.add_argument("--patience", type=int, default=10)
    p_train.add_argument("--min-real-hours", type=float, default=72.0)
    p_train.add_argument("--min-train-samples", type=int, default=400)
    p_train.add_argument("--min-val-samples", type=int, default=120)
    p_train.add_argument("--min-test-samples", type=int, default=120)
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate checkpoint against baselines and action response")
    p_eval.add_argument("--dataset", type=Path, required=True)
    p_eval.add_argument("--checkpoint", type=Path, required=True)
    p_eval.add_argument("--output", type=Path, default=None)
    p_eval.set_defaults(func=_cmd_evaluate)

    for cmd_name in ("seal", "settle", "report"):
        p_rt = sub.add_parser(cmd_name, help=f"Run shadow {cmd_name} stage")
        _add_source_args(p_rt, allow_synthetic=True)
        p_rt.add_argument("--state-dir", type=Path, required=True)
        p_rt.add_argument("--output-dir", type=Path, required=True)
        p_rt.add_argument("--checkpoint", type=Path, default=None)
        p_rt.add_argument("--cutoff-ts", type=float, default=None)
        p_rt.add_argument("--now-ts", type=float, default=None)
        p_rt.add_argument("--seed", type=int, default=42)
        p_rt.set_defaults(func=_cmd_seal_settle_report)

    p_once = sub.add_parser("run-once", help="Run one isolated shadow cycle using a config JSON")
    p_once.add_argument("--config", type=Path, required=True)
    p_once.set_defaults(func=_cmd_run_once)

    p_srv = sub.add_parser("serve", help="Run continuous isolated shadow service daemon")
    p_srv.add_argument("--config", type=Path, required=True)
    p_srv.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
