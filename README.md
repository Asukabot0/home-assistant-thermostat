# Home Assistant Three-Room Thermostat

Coordinated thermostat software for three connected rooms with separate heating and cooling equipment. "Bedroom" and "living" are independent temperature zones, and "study" shares the living-room sensor. The connection model and the initial calibration must be verified for each installation.

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

1. Copy `home-assistant/shadow/rooms.example.json` to `home-assistant/shadow/rooms.json`. Use the same temperature, power, and adapter entity IDs as the HA package. Remove optional `power_fallback` entries if the corresponding historical power entity is unavailable. Use two independent temperature sensors and share the living-room sensor for the study.
2. Copy `examples/home-assistant.env.example` to `home-assistant/private/home-assistant.env` with private file permissions. Supply a Home Assistant access token locally. Set the local Compose variables `HA_NETWORK` and `ROOM_CLIMATE_HA_URL` for your Home Assistant instance.
3. Start the shadow service and check that fresh temperatures and power reports reach Recorder. Configure each `witness_trial_id` with a completed and physically witnessed trial record. `record_trial.py` records evidence for an action that already occurred. It does not actuate equipment. Use the command's `--help` and pass the genuine observation details.
4. Copy the disabled example permit into `home-assistant/pilot-data/live-permit.json`. Use a current Unix timestamp and check the baseline tuple for each room against the actual companion state. A mismatch no longer pauses a unit permanently: the controller adopts the reported state and yields that unit for `external_override_seconds` (one hour by default), then resumes it. Enable the permit after checking the read-only data, feedback, equipment protections, and safe temperature limits. `isolated_rooms` excludes rooms from automatic control and the connected-room schedule. To keep a unit or area out of automatic control for longer, switch its virtual thermostat out of `auto`, turn on the zone `pause_entity`, or list it in `isolated_rooms` or `paused_zones`.

Automatic stops and subsequent starts require an independently verified local off-power reference for each device. Supply `verified_off_power` with the corresponding `entity` and an exclusive `ceiling_w` from actual confirmed stopped measurements. The examples omit that value so that powering off and restarting equipment is gated until commissioning. Mode and setting echoes alone are proxy evidence.

Use `topology.example.json` to configure sensors, zones and equipment influence. See [unified control and evidence scheduling](docs/unified-control.md) for the state machine, production deployment, rollback and telemetry installation.

Production credentials, equipment identifiers, unreduced telemetry, command journals and backups belong in the local runtime files excluded by `.gitignore`.
