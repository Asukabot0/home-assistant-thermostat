# Home Assistant Three-Room Thermostat

Coordinated thermostat software for three connected rooms with separate heating and cooling equipment. The default offline example (`rooms.example.json` + `topology.example.json`) models two independent temperature zones ("bedroom" and "living") where "study" shares the living-room sensor; when the study has its own temperature sensor, bind `study.temperature` to that independent entity in `rooms.json` and use `topology.three-zone.example.json`. The connection model and the initial calibration must be verified for each installation.

- The Home Assistant integration in `home-assistant/config/custom_components/room_climate_control` provides observations and virtual room thermostats.
- `home-assistant/shadow/service.py` reads Recorder data, maintains power evidence and estimates kWh for covered intervals.
- `home-assistant/shadow/continuous_pilot.py` schedules equipment changes through the configured climate entities and persists command and feedback state.
- `home-assistant/shadow/house_policy.json` contains policy parameters. The example target is 24°C with a configurable ±0.3°C tracking band. The ±0.5°C and ±0.7°C bands remain available for reporting.

## Offline tests

Run from the repository root using Python 3.11 or newer:

```sh
export ROOM_CLIMATE_ROOMS_FILE="$PWD/home-assistant/shadow/rooms.example.json"
python3 -B -m unittest discover -s work -p 'test_*.py'
python3 -B work/verify_continuous_pilot.py
python3 -B work/verify_reversal_transactions.py
```

These tests use simulated clocks, equipment and HA calls.

## Site setup

The Compose files are sidecar templates for an existing Home Assistant Container deployment with SQLite Recorder. They reference the HA image used for offline verification. Install the custom component alongside the referenced Versatile Thermostat integration. Add a copy of the example package YAML to your HA configuration and replace its sample entity IDs. Point the shadow Compose configuration mount at that same HA config directory.

1. Copy `home-assistant/shadow/rooms.example.json` to `home-assistant/shadow/rooms.json`. Use the same temperature, power, and adapter entity IDs as the HA package. Remove optional `power_fallback` entries if the corresponding historical power entity is unavailable. For the two-zone layout (`topology.example.json`), share the living-room sensor for `study`; for the three-zone star layout (`topology.three-zone.example.json`), set `study.temperature` to the study's own independent temperature entity before starting (otherwise `topology.configure()` rejects duplicate primary sensors).
2. Copy `examples/home-assistant.env.example` to `home-assistant/private/home-assistant.env` with private file permissions. Supply a Home Assistant access token locally. Set the local Compose variables `HA_NETWORK` and `ROOM_CLIMATE_HA_URL` for your Home Assistant instance.
3. Start the shadow service and check that fresh temperatures and power reports reach Recorder. Configure each `witness_trial_id` with a completed and physically witnessed trial record. `record_trial.py` records evidence for an action that already occurred. It does not actuate equipment. Use the command's `--help` and pass the genuine observation details.
4. Copy the disabled example permit into `home-assistant/pilot-data/live-permit.json`. Use a current Unix timestamp and check the baseline tuple for each room against the actual companion state. A mismatch no longer pauses a unit permanently: the controller adopts the reported state and yields that unit for `external_override_seconds` (one hour by default), then resumes it. Enable the permit after checking the read-only data, feedback, equipment protections, and safe temperature limits. `isolated_rooms` excludes rooms from automatic control and the connected-room schedule. To keep a unit or area out of automatic control for longer, switch its virtual thermostat out of `auto`, turn on the zone `pause_entity`, or list it in `isolated_rooms` or `paused_zones`.

Automatic stops and subsequent starts require an independently verified local off-power reference for each device. Supply `verified_off_power` with the corresponding `entity` and an exclusive `ceiling_w` from actual confirmed stopped measurements. The examples omit that value so that powering off and restarting equipment is gated until commissioning. Mode and setting echoes alone are proxy evidence.

Use `topology.example.json` (two zones with shared living/study temperature) or `topology.three-zone.example.json` (three independent zones in a bedroom–living–study star layout with independent lock groups and `opposite_operation.enabled: false`; requires `study.temperature` in `rooms.json` to point to an independent sensor). When changing `lock_group` assignments (for example splitting `living_study` into `living` and `study`), verify and clear any unresolved `holds` or switch to a new permit (`start` timestamp): any leftover `holds` key in `continuous-<start>.json` that does not match a current `lock_group` fails closed and pauses dispatch until resolved. See [unified control and evidence scheduling](docs/unified-control.md) for the state machine, production deployment, rollback and telemetry installation.

## Web dashboard (read-only)

`home-assistant/dashboard/` provides a zero-dependency, strictly read-only web frontend and HTTP/SSE server for inspecting the controller's state machines (zone closed-loop, bidirectional handoff, command ledger/transactions, and safety locks), live zone & actuator telemetry, and the predicted or queued next command with live gate countdowns. It mounts all controller data read-only and has no Home Assistant credentials or actuation path.

Run against local runtime files (default live read-only mode, `http://127.0.0.1:8780`):

```sh
PYTHONPATH="$PWD/home-assistant" python3 -B -m dashboard
```

Preview with built-in simulated state-machine scenarios (explicit opt-in only):

```sh
PYTHONPATH="$PWD/home-assistant" python3 -B -m dashboard --demo
```

Or run as a hardened read-only Docker Compose sidecar from `home-assistant/` (uses `read_only: true`, `:ro` mounts, `user: 0:0`, and `cap_drop: [ALL]` with no added capabilities — matching `compose.pilot.yaml` so it can read the same `0600` journals and site configs without weakening container isolation):

```sh
cd home-assistant
docker compose -f compose.dashboard.yaml up -d
```

Security note: the dashboard has no built-in authentication and binds `127.0.0.1:8780` by default. Do **not** expose it directly to a LAN via `DASHBOARD_BIND=0.0.0.0` unless placed behind an authenticating reverse proxy (or accessed over SSH port forwarding / Tailscale).

Production credentials, equipment identifiers, unreduced telemetry, command journals and backups belong in the local runtime files excluded by `.gitignore`.

## Decisions

Decisions are recorded entry by entry in the [decision ledger](docs/decision-ledger.md). `AGENTS.md` defines what to record and how.
