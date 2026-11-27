# Neural-Network Thermostat Shadow System — Deployment Handoff

This document is the operator and deploying-agent handoff for the independent v1 neural-network thermostat shadow system (`home-assistant/neural_shadow/`).

> **Safety & Status Summary**
> - **Shadow-only, zero actuation path**: The neural shadow package has no HTTP client, no Home Assistant service calls, and runs with `network_mode: none` and read-only input mounts.
> - **Existing live controller untouched**: No files under `home-assistant/shadow/`, `home-assistant/config/`, `home-assistant/compose.pilot.yaml`, or `home-assistant/compose.shadow.yaml` were modified. Production permits, pause locks, command ledgers, `code_sha256`, `config_sha256`, and live control-rule parameters remain unchanged.
> - **Not deployed / no real-house training claims**: This delivery has **not** been deployed to production, has **no** live runtime results, and claims **no** measured energy savings or real-house predictive accuracy. Full multi-day production history is only available in the deployment environment; short public evidence packs (`telemetry-20261127` 3h, `acceptance-20261127T121613-jst` ~10h) are verified as `insufficient_data`, and synthetic runs verify software correctness only.

---

## 1. Git Branch and Commit to Check Out

- **Pullable branch**: `cursor/neural-thermostat-shadow-8300`
- **Verified code + offline test commit SHA**: `ebab1087f206aa26784f3c268c2843947a959204`
  - Core neural shadow package, Compose/Dockerfile, `.gitignore`, and optional telemetry export hook: `e321015bf419f68134cdc8c07e61e90a435c06b7`
  - Offline regression tests (`work/test_neural_shadow.py`) and local E2E example (`work/run_neural_shadow_e2e.py`): `ebab1087f206aa26784f3c268c2843947a959204`
- **Tip commit**: Check out the branch tip (`cursor/neural-thermostat-shadow-8300`, whose full commit SHA including this `DEPLOYMENT_HANDOFF.md` is reported in the handoff message and PR):

```sh
git fetch origin cursor/neural-thermostat-shadow-8300
git checkout cursor/neural-thermostat-shadow-8300
git rev-parse HEAD
```

Verify that `git log -n 3 --oneline` includes `e321015`, `ebab108`, and the handoff documentation commit.

---

## 2. Dependencies, Build, Config, and Data Input Contracts

### 2.1 Runtime & Training Dependencies

Pinned in `home-assistant/neural_shadow/requirements.txt`:
- Python `>= 3.11` (Docker image uses `python:3.12-slim`)
- `numpy>=1.26,<3.0`
- CPU-only PyTorch `torch>=2.5,<2.8`

Install on host (for offline training/evaluation outside Docker):

```sh
python3 -m venv .venv-neural-shadow
source .venv-neural-shadow/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5,<2.8"
pip install "numpy>=1.26,<3.0"
```

Note: The existing hourly telemetry worker (`home-assistant/telemetry/evidence.py`, `worker.py`) stays **pure Python stdlib** with zero PyTorch/NumPy dependency.

### 2.2 Container Build

Build the standalone read-only container image from the `home-assistant/` directory (Docker context is the repo root `..`):

```sh
cd home-assistant
docker compose -f compose.neural-shadow.yaml build
```

### 2.3 Configuration Contract

1. Copy `home-assistant/neural_shadow/neural_shadow.example.json` to `home-assistant/neural_shadow/config.json` (gitignored, keep owner-only permissions `chmod 600`).
2. Default container paths inside `neural_shadow/config.json`:
   - `inputs.rooms_file`: `/opt/shadow-config/rooms.json`
   - `inputs.topology_file`: `/opt/shadow-config/topology.json`
   - `inputs.policy_file`: `/opt/shadow-config/house_policy.json`
   - `inputs.ha_config_dir`: `/ha-readonly`
   - `inputs.shadow_data_dir`: `/shadow-data-ro`
   - `inputs.pilot_data_dir`: `/pilot-data-ro`
   - `model.artifact_dir`: `/models`
   - `model.checkpoint_name`: `thermostat_shadow_gru.pt`
   - `outputs.state_dir`: `/state`
   - `outputs.output_dir`: `/output`
3. **Version lock contract**:
   - `MODEL_ARCH_VERSION = "ns-gru-v1"` (2-layer 128-d GRU encoder + action-conditioned decoder, **240,471 parameters** for the default 2-zone / 3-unit topology)
   - `FEATURE_VERSION = "ns-feat-v1"`
   - `CONSTRAINT_VERSION = "ns-v1-hold-pm1c"` (V1 candidates strictly limited to `hold` and same-mode `±1°C` after unified hard-constraint checks)
   - `config_fingerprint`: SHA-256 of active zones, sensors, actuators, soft limits, and observation version. Any topology/feature/architecture mismatch fails closed (`ModelVersionMismatchError`) instead of silently running incompatible weights.
4. **Future sensor extensibility**:
   - Currently: 2 independent temperature label zones (`sleep` -> bedroom primary, `lounge` -> living primary; `study` shares the living sensor and is NEVER double-counted as a third temperature label) and 3 independent actuators (`bedroom`, `living`, `study`).
   - When a future study thermometer (`office_primary`) or outdoor balcony thermometer (`balcony`) comes online, enable it in `topology.json` and bump `observation_version`. Rebuild the dataset and retrain; the pipeline and model automatically adapt dimensions and update `config_fingerprint`.

### 2.4 Data Input Contracts (Strictly Read-Only)

The data pipeline (`home-assistant/neural_shadow/data_pipeline.py`) supports three read-only sources:
1. **Production directory (`--production /path/to/home-assistant`)** — primary deployment mode:
   - `shadow/rooms.json`, `shadow/topology.json`, `shadow/house_policy.json`
   - `config/home-assistant_v2.db` (opened `?mode=ro` + `PRAGMA query_only=ON`, WAL-safe)
   - `config/.storage/room_climate_control.sqlite3` (opened read-only with quiescent snapshot fallback)
   - `shadow-data/shadow.sqlite3` (`power_reports`, `decisions`)
   - `pilot-data/continuous-*.json`, `pilot-data/live-audit.jsonl*`, `pilot-data/receipt-*.json`
2. **Evidence ZIP archives (`--archives <zip1> [zip2 ...]`)**:
   - Supports anonymized hourly telemetry ZIPs (`effective-config.json` + `raw/*.jsonl`) and acceptance ZIPs (`effective-config.redacted.json` + `raw/*.jsonl` + `context/*.jsonl`).
   - Short archives are tagged `data_origin="real_short_evidence_pack"` and must not be treated as complete site history.
3. **Exported canonical snapshot (`--snapshot /path/to/snapshot`)**:
   - Produced by `export-history` (`snapshot_meta.json`, `temp_reports.jsonl`, `power_reports.jsonl`, `proxy_states.jsonl`, `actions.jsonl`).

### 2.5 Artifact Storage Convention

Large data, checkpoints, and runtime outputs are excluded via `.gitignore` and stored locally (matching existing repo conventions):
- Private config: `home-assistant/neural_shadow/config.json`
- Model checkpoints & training reports: `home-assistant/neural-shadow-models/`
- Independent shadow SQLite state (sealed predictions + settlements): `home-assistant/neural-shadow-data/`
- Independent shadow outputs/reports: `outputs/neural-shadow/`

---

## 3. Mount Layout (Read-Only Inputs, Model Dir, Independent Outputs)

Defined in `home-assistant/compose.neural-shadow.yaml`:

| Host Path (relative to `home-assistant/`) | Container Path | Mode | Purpose |
|---|---|---|---|
| `./neural_shadow/config.json` | `/config/neural_shadow.json` | `:ro` | Neural shadow runtime config |
| `./shadow` | `/opt/shadow-config` | `:ro` | `rooms.json`, `topology.json`, `house_policy.json` |
| `./config` | `/ha-readonly` | `:ro` | HA Recorder SQLite & `.storage/room_climate_control.sqlite3` |
| `./shadow-data` | `/shadow-data-ro` | `:ro` | Existing `shadow.sqlite3` (durable power ledger & decisions) |
| `./pilot-data` | `/pilot-data-ro` | `:ro` | Live controller journals, receipts, audit logs |
| `./neural-shadow-models` | `/models` | `:ro` | Trained checkpoint (`thermostat_shadow_gru.pt` + `.meta.json`) |
| `./neural-shadow-data` | `/state` | `rw` | Independent SQLite (`neural_shadow.sqlite3`: immutable sealed predictions + settlements) |
| `../outputs/neural-shadow` | `/output` | `rw` | Independent reports (`neural-shadow-status.json`, `neural-shadow-scores.jsonl`, `neural-shadow-health.json`, `neural-shadow-report.md`) |

Container hardening:
- `network_mode: none` (zero network access to Home Assistant or external endpoints)
- `read_only: true`, `tmpfs: ["/tmp:size=64m,mode=1777"]`
- `cap_drop: [ALL]`, `security_opt: [no-new-privileges:true]`
- `mem_limit: 512m`, `pids_limit: 64`, non-root user (`uid 10001`)

Create writable directories before first start:

```sh
mkdir -p home-assistant/neural-shadow-models \
         home-assistant/neural-shadow-data \
         outputs/neural-shadow
```

---

## 4. Operational Commands (Export, Build, Train, Evaluate, Start, Stop)

Run commands from the repository root with `PYTHONPATH="$PWD/home-assistant"` (using the host Python environment with `torch` and `numpy` installed, or via `docker compose -f home-assistant/compose.neural-shadow.yaml run --rm ...`).

### 4.1 Real-History Read-Only Export & Dataset Build

Export a self-contained read-only canonical snapshot from production (optional archival step) and build the aligned dataset + manifest:

```sh
export PYTHONPATH="$PWD/home-assistant"
PROD_DIR="/path/to/production/home-assistant"
WORK_DIR="/path/to/local-training-work"

# 1. Read-only export from production SQLite + pilot journals
python3 -B -m neural_shadow export-history \
  --production "$PROD_DIR" \
  --output "$WORK_DIR/canonical-snapshot"

# 2. Repeatable causal dataset build (3h history @ 5m step, 15/30/60m horizons)
python3 -B -m neural_shadow data-build \
  --snapshot "$WORK_DIR/canonical-snapshot" \
  --history-minutes 180 \
  --step-minutes 5 \
  --horizons 15 30 60 \
  --min-wh-coverage 0.85 \
  --output "$WORK_DIR/dataset"
```

(You can also run `data-build --production "$PROD_DIR" --output "$WORK_DIR/dataset"` directly.)

Inspect `$WORK_DIR/dataset/manifest.json` to verify `observed_span_hours`, `unique_temp_reports_by_zone`, `power_coverage_by_unit`, `action_type_counts`, and `data_sha256`.

### 4.2 Train, Calibrate, and Evaluate Against Baselines

```sh
export PYTHONPATH="$PWD/home-assistant"
python3 -B -m neural_shadow train \
  --dataset "$WORK_DIR/dataset" \
  --output home-assistant/neural-shadow-models \
  --seed 42 \
  --epochs 45 \
  --batch-size 64 \
  --lr 0.0015 \
  --patience 10
```

Outputs in `home-assistant/neural-shadow-models/`:
- `thermostat_shadow_gru.pt` (weights, train-fitted normalizer, OOD envelope, calibration scales, version contract)
- `thermostat_shadow_gru.meta.json` (human-readable provenance, `model_state`, `training_cutoff_utc`, `readiness_reasons`, `checkpoint_sha256`)
- `evaluation_report.json` (forward-chaining validation & test comparisons against `persistence`, `existing_trend`, and `thermal_inertia_linear`; per-zone & per-horizon MAE/RMSE/bias/p90/p95/PI80; Wh MAE & metering coverage; slices for `mode_heating`/`mode_cooling`/`mode_off`/`post_action`/`over_cool`/`over_heat`; action-response sensitivity audit)

Re-run standalone evaluation at any time:

```sh
python3 -B -m neural_shadow evaluate \
  --dataset "$WORK_DIR/dataset" \
  --checkpoint home-assistant/neural-shadow-models/thermostat_shadow_gru.pt \
  --output "$WORK_DIR/eval-recheck.json"
```

**Model lifecycle states**:
- `untrained`: random weights / missing checkpoint — shadow declines recommendations.
- `insufficient_data`: fails data span (`>= 72h`), split sample count, unique temp reports, power coverage, or action coverage gates — shadow declines recommendations.
- `uncalibrated`: trained on sufficient data, but failed validation PI calibration, action-response sensitivity check, or baseline sanity check — shadow logs diagnostic predictions and declines recommendations.
- `ready_to_use`: passed all data, calibration, baseline, and action-response gates on real history — shadow still remains **observation/scoring only** (zero actuation path).

### 4.3 Start, One-Shot Smoke, and Stop the Shadow Service

Copy and inspect config first:

```sh
cp home-assistant/neural_shadow/neural_shadow.example.json home-assistant/neural_shadow/config.json
chmod 600 home-assistant/neural_shadow/config.json
```

One-shot cycle test before starting daemon:

```sh
cd home-assistant
docker compose -f compose.neural-shadow.yaml run --rm neural-climate-shadow \
  run-once --config /config/neural_shadow.json
```

Start the continuous shadow sidecar:

```sh
cd home-assistant
docker compose -f compose.neural-shadow.yaml up -d --build
```

Stop the shadow sidecar (leaves existing `room-climate-shadow` and `heating-pilot` running):

```sh
cd home-assistant
docker compose -f compose.neural-shadow.yaml stop
```

---

## 5. Post-Deploy Verification: Data Freshness, Model Status, and First Scores

Check container health and zero network attachment:

```sh
docker ps --filter name=neural-climate-shadow
docker inspect -f '{{.HostConfig.NetworkMode}} {{.HostConfig.ReadonlyRootfs}}' \
  $(docker compose -f home-assistant/compose.neural-shadow.yaml ps -q)
# Expected: none true
```

Check health heartbeat, model status, versions, and latest sealed prediction freshness:

```sh
cat outputs/neural-shadow/neural-shadow-health.json
python3 -c '
import json
s = json.load(open("outputs/neural-shadow/neural-shadow-status.json"))
print("model:", json.dumps(s["model"], indent=2))
print("counts:", json.dumps(s["counts"], indent=2))
print("latest_sealed:", json.dumps(s["latest_sealed"], indent=2))
'
```

What to confirm immediately (first 5 minutes):
- `neural-shadow-health.json` has `"state": "healthy"` and `"last_error": null`.
- `model.model_state` matches `thermostat_shadow_gru.meta.json` (`untrained`, `insufficient_data`, `uncalibrated`, or `ready_to_use`), and `feature_version == "ns-feat-v1"`, `model_arch_version == "ns-gru-v1"`.
- `latest_sealed.true_report_times` shows fresh independent `sleep` / `lounge` report timestamps (not stale) and a non-empty `seal_sha256`.

What to confirm after 20–70 minutes (first 15m / 30m / 60m label settlements):
- `counts.settled_horizons > 0`
- `outputs/neural-shadow/neural-shadow-scores.jsonl` contains settled JSON lines with explicit `factual_settlement_kind` (`factual_hold` vs `intervened_hold_unscored`), `recommended_execution_status` (`factual_hold`, `counterfactual_unexecuted`, `factual_coincident_recommendation`, or `declined`), and `joint_wh_coverage_ratio` / `joint_wh_has_gap`.
- Inspect the human-readable summary:

```sh
cat outputs/neural-shadow/neural-shadow-report.md
```

---

## 6. Hooking Into the Existing Hourly Evidence Export

> **Note**: No production cron/LaunchAgent/timer is created or modified by this delivery. Below is the opt-in integration procedure once the shadow sidecar is running cleanly.

`home-assistant/telemetry/evidence.py` and `home-assistant/telemetry/worker.py` include a backward-compatible, exception-isolated optional hook:
- When `neural_shadow_output` is **omitted** from the telemetry `config.json`, hourly evidence export behaves **100% identically** to before.
- When `neural_shadow_output` is **set** to the host path of `outputs/neural-shadow`:
  1. Deploy the updated stdlib-only `home-assistant/telemetry/evidence.py` and `home-assistant/telemetry/worker.py` into `~/Library/Application Support/HomeAssistantThermostat/telemetry/` (with mode `0600`, keeping the existing `anonymization_key` intact).
  2. Add one optional key to `~/Library/Application Support/HomeAssistantThermostat/telemetry/config.json`:
     ```json
     "neural_shadow_output": "/absolute/path/to/repo/outputs/neural-shadow"
     ```
  3. On the next hourly run (at minute `:05`), each exported ZIP includes:
     - `raw/neural-shadow-scores.jsonl` (windowed settled scores with all zone/unit/model/prediction identifiers HMAC-anonymized through `PublicSchema`),
     - `summary.json -> neural_shadow` (model state, versions, training cutoff, factual vs intervened vs counterfactual counts, per-horizon MAE/bias/PI80, Wh MAE, and `has_coverage_gap` flags),
     - `manifest.json -> sources.neural_shadow` (watermark and freshness).
  4. **Fault isolation**: If `outputs/neural-shadow` is missing, stopped, or corrupted, `evidence.collect` logs `"status": "unavailable"` under `neural_shadow` and still completes the normal hourly evidence upload uninterrupted.

---

## 7. Stop and Remove Without Affecting the Existing Controller

Stopping or completely removing the neural shadow system never restarts, stops, or alters `room-climate-shadow` or `heating-pilot`:

```sh
# 1. Stop and remove only the neural shadow container
cd home-assistant
docker compose -f compose.neural-shadow.yaml down

# 2. (Optional) Remove the opt-in telemetry hook key if you enabled it:
#    Delete "neural_shadow_output" from
#    ~/Library/Application Support/HomeAssistantThermostat/telemetry/config.json

# 3. (Optional) Remove local shadow state, outputs, config, and models
rm -rf neural-shadow-data neural-shadow-models neural_shadow/config.json ../outputs/neural-shadow

# 4. Confirm existing live controller & shadow containers are still healthy and untouched
docker ps --filter name=heating-pilot --filter name=room-climate-shadow
```

---

## 8. Verification Matrix

### Already Verified Offline (in this delivery)
- **Full unit test suite (`125 tests`, all passing)** via `ROOM_CLIMATE_ROOMS_FILE="$PWD/home-assistant/shadow/rooms.example.json" python3 -B -m unittest discover -s work -p 'test_*.py'`, including 12 dedicated regressions in `work/test_neural_shadow.py`:
  - Time leakage (future mutation bit-for-bit invariance, 60m forward-chaining embargo, train-only normalization).
  - Shared sensors (`study` sharing `living` sensor produces strictly 2 temperature labels; enabling future `office_primary` + `balcony` works via config only).
  - Duplicate cache vs same-value new reports (temperature + power, int-truncated sample reconciliation).
  - Missing measurements (power gaps never treated as 0 Wh; missing zone temp triggers abstention).
  - Mid-horizon action/target changes & strict factual vs counterfactual separation.
  - Model version mismatch (`ModelVersionMismatchError`) & explicit lifecycle states (`untrained`, `insufficient_data`, `uncalibrated`, `ready_to_use`).
  - V1 candidate hard constraints (prohibiting start/stop/reversal) & worst-zone comfort-first scoring (no whole-house average cancellation).
  - `proxy_echo` unverified flag, failed calls, manual operations.
  - Exception & timeout isolation in both `ShadowRuntime` and the hourly `telemetry/evidence.py` export hook (including HMAC anonymization check).
  - Static AST + Compose audit verifying zero actuation path, `network_mode: none`, `read_only: true`, and `:ro` mounts.
  - Read-only mock production SQLite + journal ingestion (verified zero input modification).
- **Existing verification scripts (all passing)**: `verify_continuous_pilot.py`, `verify_reversal_transactions.py`, `verify_climate_cadence.py`, `verify_climate_shadow.py`, `verify_climate_strategy.py`, `verify_steady_learning.py`.
- **Local end-to-end synthetic example**: `python3 -B work/run_neural_shadow_e2e.py --synthetic-hours 96 --epochs 25 --seed 42` (`data build -> train -> reload -> seal -> settle -> report`; verified `240,471` parameters, baseline comparisons, action-response check, and `deployment_eligible=false` on synthetic data).
- **Read-only CLI smoke on short public evidence archives**: verified ingestion and honest `insufficient_data` gating on `telemetry-20261127` (3h) and `acceptance-20261127T121613-jst` (~10h).

### Must Be Verified in the Deployment Environment
1. **Host Docker build & runtime**: Build `compose.neural-shadow.yaml` on the production host architecture and verify `network_mode=none`, `ReadonlyRootfs=true`, and read-only access to live WAL-mode SQLite databases.
2. **Real multi-day history export & training**: Export full site history (`>= 72h` with sufficient heating/cooling/action coverage), run `data-build` and `train`, and review `evaluation_report.json` (per-zone/horizon errors, bias, quantiles, PI80 calibration, operating slices, Wh metering coverage, baseline comparisons, and action-response audit).
3. **Model readiness gate sign-off**: Confirm whether real site history reaches `model_state == "ready_to_use"` or remains `insufficient_data` / `uncalibrated` (e.g., if seasonal action coverage is narrow). Never bypass readiness gates.
4. **Live shadow observation**: Run `compose.neural-shadow.yaml`, verify fresh `true_report_times` and immutable seals in `neural-shadow-status.json`, and wait `>= 70 minutes` to inspect the first real settled `factual_hold` vs `intervened_hold_unscored` vs `counterfactual_unexecuted` scores.
5. **Non-interference check**: Confirm existing `room-climate-shadow` and `heating-pilot` containers keep identical `code_sha256`, `config_sha256`, normal 20s control cycles, and intact permits/ledgers.
6. **Optional hourly telemetry hook**: If enabled in telemetry `config.json`, download and inspect one generated hourly ZIP to confirm `raw/neural-shadow-scores.jsonl` and `summary.json -> neural_shadow` pass public redaction and checksum verification.
