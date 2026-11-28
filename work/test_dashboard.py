"""Unit, integration, HTTP/SSE, and read-only isolation tests for the web dashboard."""
from __future__ import annotations

import ast
import http.client
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parent
HA_DIR = REPO_ROOT / "home-assistant"
for _p in (str(WORK_DIR), str(HA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.collector import DashboardPaths, collect_snapshot  # noqa: E402
from dashboard.demo import SCENARIOS, build_demo_snapshot  # noqa: E402
from dashboard.server import ServerConfig, create_server  # noqa: E402

import test_p0_p1_dispatch as fixture  # noqa: E402
from topology import configure  # noqa: E402

pilot = fixture.pilot
EXAMPLE_TOPOLOGY = Path(pilot.__file__).with_name("topology.example.json")


class DashboardCollectorTests(unittest.TestCase):
    def _empty_paths(self, root: Path) -> DashboardPaths:
        return DashboardPaths(
            shadow_config_dir=HA_DIR / "shadow",
            pilot_data_dir=root / "pilot-data",
            shadow_data_dir=root / "shadow-data",
            shadow_output_dir=root / "climate-shadow",
            neural_output_dir=root / "neural-shadow",
        )

    def test_empty_live_dir_never_fakes_demo_data(self):
        with TemporaryDirectory() as tmp:
            snap = collect_snapshot(self._empty_paths(Path(tmp)), now=1_700_000_000.0)
            self.assertEqual(snap["mode"], "live")
            self.assertIsNone(snap["scenario"])
            self.assertFalse(snap["controller"]["data_present"])
            self.assertEqual(snap["controller"]["health"], "offline")
            self.assertEqual(snap["next_command"]["stage"], "paused_or_locked")
            # Strictly JSON serializable without NaN
            json.dumps(snap, allow_nan=False)

    def test_malformed_json_files_degrade_gracefully(self):
        with TemporaryDirectory() as tmp:
            paths = self._empty_paths(Path(tmp))
            paths.pilot_data_dir.mkdir(parents=True)
            (paths.pilot_data_dir / "status.json").write_text("{broken json", encoding="utf-8")
            (paths.pilot_data_dir / "live-permit.json").write_text("not-json", encoding="utf-8")
            (paths.pilot_data_dir / "live-audit.jsonl").write_text("{bad\n{\"event\":\"startup\",\"at\":100}\n", encoding="utf-8")

            snap = collect_snapshot(paths, now=1_700_000_000.0)
            self.assertEqual(snap["mode"], "live")
            self.assertEqual(snap["controller"]["status_error"], "JSONDecodeError")
            self.assertFalse(snap["controller"]["permit"]["active"])
            self.assertEqual(len(snap["audit_recent"]), 1)

    def test_real_controller_cycle_integration_and_next_command_forecast(self):
        import heating_pilot

        h = fixture.DispatchTests()
        self.addCleanup(h.doCleanups)
        h.setUp()
        topology = configure(json.loads(EXAMPLE_TOPOLOGY.read_bytes()), pilot.ROOMS)
        self.enterContext(patch.object(pilot, "TOPOLOGY", topology))
        self.enterContext(patch.object(heating_pilot, "DATA_DIR", h.root))
        self.enterContext(patch.object(pilot, "publish", side_effect=heating_pilot.publish))
        live_permit_path = h.root / "live-permit.json"
        h.permit = live_permit_path
        self.enterContext(patch.object(pilot, "LIVE_PERMIT", live_permit_path))

        h.whole_scene(("heat", 22, 23.35, 180), ("heat", 23, 23.95, 140), ("off", 25, 23.95, 8))
        permit = json.loads(h.permit.read_bytes())
        for r, sig in permit["rooms"].items():
            if sig[0] == "off":
                permit["rooms"][r] = ["cool", 25, "auto"]
        h.permit.write_text(json.dumps(permit), encoding="utf-8")
        h.park("study", ago=800)

        def step(seconds=0, bed=None, lounge=None):
            h.now += seconds
            for room, deg in (("bedroom", bed), ("living", lounge), ("study", lounge)):
                if deg is not None:
                    h.live[room]["temperature"] = deg
                h.observe(room, new_report=True)
                with h.db:
                    h.db.execute(
                        "UPDATE decisions SET value=json_set(value,'$.power_evidence.bands.low.seconds',600) WHERE room=?",
                        (room,),
                    )
            return h.snapshot()

        # Accumulate partial dwell (10 minutes < 20 minutes) -> Observe + projected next command
        step(bed=23.35, lounge=23.95)
        step(600, bed=23.35, lounge=23.95)

        paths = DashboardPaths(
            shadow_config_dir=HA_DIR / "shadow",
            pilot_data_dir=h.root,
            shadow_data_dir=h.root,
            shadow_output_dir=h.root / "no-shadow-out",
            neural_output_dir=h.root / "no-neural-out",
        )
        snap_dwell = collect_snapshot(paths, now=h.now + 5)
        self.assertEqual(snap_dwell["controller"]["mode"], "unified_topology")
        self.assertTrue(snap_dwell["controller"]["permit"]["active"])
        sleep_zone = next(z for z in snap_dwell["state_machines"]["zone_states"] if z["zone"] == "sleep")
        self.assertEqual(sleep_zone["state"], "observe")
        self.assertEqual(sleep_zone["reason"], "waiting_reports_or_dwell")
        self.assertEqual(snap_dwell["next_command"]["stage"], "projected")
        self.assertEqual(snap_dwell["next_command"]["action"]["room"], "bedroom")
        self.assertEqual(snap_dwell["next_command"]["action"]["to_value"], 23)
        self.assertGreater(snap_dwell["next_command"]["eta_seconds"], 0)
        # shadow.sqlite3 series should also load read-only
        self.assertIn("bedroom", snap_dwell["shadow_series"])

        # Continue until dwell qualifies (20m across independent reports) -> command dispatched -> in_flight accepted
        step(600, bed=23.32, lounge=23.95)
        step(600, bed=23.30, lounge=23.95)
        self.assertEqual(len(h.calls), 1)
        snap_inflight = collect_snapshot(paths, now=h.now + 10)
        self.assertEqual(snap_inflight["next_command"]["stage"], "in_flight")
        self.assertEqual(snap_inflight["state_machines"]["ledger"]["active_phase"], "accepted")
        self.assertEqual(snap_inflight["next_command"]["action"]["room"], "bedroom")
        self.assertEqual(snap_inflight["next_command"]["action"]["to_value"], 23)
        self.assertEqual(snap_inflight["commands"][0]["status"], "accepted")

    def test_queued_transaction_initial_safety_easing_and_reversal_initial(self):
        with TemporaryDirectory() as tmp:
            paths = self._empty_paths(Path(tmp))
            paths.pilot_data_dir.mkdir(parents=True)
            now = 1_764_300_000.0
            (paths.pilot_data_dir / "live-permit.json").write_text(
                json.dumps({"enabled": True, "start": int(now - 3600), "end": None, "rooms": {"bedroom": ["heat", 28, "auto"], "living": ["heat", 23, "auto"], "study": ["heat", 23, "auto"]}}),
                encoding="utf-8",
            )
            (paths.pilot_data_dir / f"continuous-{int(now - 3600)}.json").write_text(
                json.dumps({
                    "start": int(now - 3600),
                    "baselines": {"bedroom": ["heat", 28, "auto"], "living": ["heat", 23, "auto"], "study": ["heat", 23, "auto"]},
                    "holds": {},
                    "manual_holds": {},
                    "parked": {},
                    "commands": [],
                    "transitions": {
                        "bedroom": {
                            "id": now - 10,
                            "mode": "heat",
                            "ready_at": now + 4,
                            "members": {"bedroom": {"phase": "initial", "target": 24.0}},
                        }
                    },
                    "closed_loop": {},
                }),
                encoding="utf-8",
            )
            (paths.pilot_data_dir / "status.json").write_text(
                json.dumps({
                    "at": "2026-11-28T06:00:00+00:00",
                    "status": "持续实机控制运行中",
                    "rooms": {
                        "bedroom": {"room_c": 23.2, "target_c": 24.0, "mode_reported": "heat", "setpoint_reported": 28, "power_w": 300, "ready": True, "power_age_s": 5, "temperature_age_s": 20},
                    },
                    "control": {"at": now - 2, "topology": "site-1", "need": "heat", "owner": "sleep", "zones": {}},
                }),
                encoding="utf-8",
            )
            snap = collect_snapshot(paths, now=now)
            self.assertEqual(snap["next_command"]["stage"], "queued_transaction")
            self.assertEqual(snap["next_command"]["action"]["purpose"], "initial_safety_easing")
            self.assertEqual(snap["next_command"]["action"]["to_value"], 24)

    def test_inactive_permit_blocks_queued_transaction_forecast_but_keeps_inflight(self):
        with TemporaryDirectory() as tmp:
            paths = self._empty_paths(Path(tmp))
            paths.pilot_data_dir.mkdir(parents=True)
            now = 1_764_300_000.0
            # Permit is disabled while an `initial` transaction remains in the journal
            (paths.pilot_data_dir / "live-permit.json").write_text(
                json.dumps({
                    "enabled": False,
                    "start": int(now - 3600),
                    "end": None,
                    "rooms": {"bedroom": ["heat", 24, "auto"], "living": ["heat", 23, "auto"], "study": ["heat", 23, "auto"]},
                }),
                encoding="utf-8",
            )
            journal_path = paths.pilot_data_dir / f"continuous-{int(now - 3600)}.json"
            journal_data = {
                "start": int(now - 3600),
                "baselines": {"bedroom": ["heat", 24, "auto"], "living": ["heat", 23, "auto"], "study": ["heat", 23, "auto"]},
                "holds": {},
                "manual_holds": {},
                "parked": {},
                "commands": [],
                "transitions": {
                    "bedroom": {
                        "id": now - 10,
                        "mode": "heat",
                        "ready_at": now + 4,
                        "members": {"bedroom": {"phase": "initial", "target": 24.0}},
                    }
                },
                "closed_loop": {},
            }
            journal_path.write_text(json.dumps(journal_data), encoding="utf-8")
            (paths.pilot_data_dir / "status.json").write_text(
                json.dumps({
                    "at": "2026-11-28T06:00:00+00:00",
                    "status": "实机控制已暂停或验证窗口已结束",
                    "rooms": {
                        "bedroom": {"room_c": 23.2, "target_c": 24.0, "mode_reported": "heat", "setpoint_reported": 28, "power_w": 300, "ready": True, "power_age_s": 5, "temperature_age_s": 20},
                    },
                    "control": {"at": now - 2, "topology": "site-1", "need": "heat", "owner": "sleep", "zones": {}},
                }),
                encoding="utf-8",
            )

            snap = collect_snapshot(paths, now=now)
            self.assertEqual(snap["next_command"]["stage"], "paused_or_locked")
            self.assertIsNone(snap["next_command"]["action"])

            # Already-sent accepted command still renders as in_flight even if permit was just disabled
            journal_data["commands"].append({
                "room": "bedroom",
                "kind": "mode",
                "value": "heat",
                "before": ["off", None, "auto"],
                "sent_at": now - 20,
                "status": "accepted",
                "purpose": "energy_start",
                "expected": ["heat"],
            })
            journal_path.write_text(json.dumps(journal_data), encoding="utf-8")
            snap_inflight = collect_snapshot(paths, now=now)
            self.assertEqual(snap_inflight["next_command"]["stage"], "in_flight")

    def test_handoff_countdown_anchors_to_current_handoff_newest_first(self):
        with TemporaryDirectory() as tmp:
            paths = self._empty_paths(Path(tmp))
            paths.pilot_data_dir.mkdir(parents=True)
            now = 1_764_300_000.0
            (paths.pilot_data_dir / "live-permit.json").write_text(
                json.dumps({
                    "enabled": True,
                    "start": int(now - 7200),
                    "end": None,
                    "rooms": {"bedroom": ["heat", 22, "auto"], "living": ["cool", 25, "auto"], "study": ["cool", 25, "auto"]},
                }),
                encoding="utf-8",
            )
            old_handoff_at = now - 3600
            cur_handoff_at = now - 80
            journal_path = paths.pilot_data_dir / f"continuous-{int(now - 7200)}.json"
            journal_data = {
                "start": int(now - 7200),
                "baselines": {"bedroom": ["heat", 22, "auto"], "living": ["cool", 25, "auto"], "study": ["off", None, "auto"]},
                "holds": {},
                "manual_holds": {},
                "parked": {},
                "transitions": {},
                "commands": [
                    # Older completed handoff for the same blocker ("living")
                    {
                        "room": "living",
                        "kind": "temperature",
                        "value": 25,
                        "before": ["cool", 24, "auto"],
                        "sent_at": old_handoff_at + 2,
                        "proxy_at": old_handoff_at + 5,
                        "status": "proxy_echo",
                        "purpose": "handoff_reduce",
                        "control": {"zone": "lounge", "mode": "cool", "handoff_at": old_handoff_at},
                    },
                    {
                        "room": "living",
                        "kind": "mode",
                        "value": "off",
                        "before": ["cool", 25, "auto"],
                        "sent_at": old_handoff_at + 200,
                        "proxy_at": old_handoff_at + 205,
                        "status": "proxy_echo",
                        "purpose": "handoff_off",
                        "control": {"zone": "lounge", "mode": "cool", "handoff_at": old_handoff_at},
                    },
                    # Current handoff's fresh reduction echoed 40s ago -> 140s remaining of 180s window
                    {
                        "room": "living",
                        "kind": "temperature",
                        "value": 26,
                        "before": ["cool", 25, "auto"],
                        "sent_at": now - 44,
                        "proxy_at": now - 40,
                        "status": "proxy_echo",
                        "purpose": "handoff_reduce",
                        "control": {"zone": "lounge", "mode": "cool", "handoff_at": cur_handoff_at},
                    },
                ],
                "closed_loop": {},
            }
            journal_path.write_text(json.dumps(journal_data), encoding="utf-8")

            def write_status(phase: str):
                (paths.pilot_data_dir / "status.json").write_text(
                    json.dumps({
                        "at": "2026-11-28T06:00:00+00:00",
                        "status": "持续实机控制运行中",
                        "rooms": {
                            "bedroom": {"room_c": 23.1, "target_c": 24.0, "mode_reported": "heat", "setpoint_reported": 22, "power_w": 200, "ready": True, "power_age_s": 5, "temperature_age_s": 20, "zone": "sleep"},
                            "living": {"room_c": 24.7, "target_c": 24.0, "mode_reported": "cool", "setpoint_reported": 26, "power_w": 210, "ready": True, "power_age_s": 5, "temperature_age_s": 20, "zone": "lounge"},
                        },
                        "control": {
                            "at": now - 2,
                            "topology": "site-1",
                            "need": "heat",
                            "owner": "sleep",
                            "conflict": True,
                            "elected": "bedroom",
                            "feasible": {"sleep": ["bedroom"]},
                            "handoff": {
                                "owner": "sleep",
                                "receiver": "bedroom",
                                "blocker": "living",
                                "direction": "heat",
                                "phase": phase,
                                "reason": "observe_both_sensors_after_reduction",
                                "selected_at": cur_handoff_at,
                            },
                            "zones": {
                                "sleep": {"key": "sleep", "measured": 23.1, "target": 24.0, "error": 0.9, "at": now - 20, "quality": "fresh", "paused": False, "intent": "recover", "reason": "confirmed_bias", "direction": "heat", "qualified": True},
                                "lounge": {"key": "lounge", "measured": 24.7, "target": 24.0, "error": -0.7, "at": now - 20, "quality": "fresh", "paused": False, "intent": "limited", "reason": "conflicting_priority", "direction": "cool", "qualified": True},
                            },
                        },
                    }),
                    encoding="utf-8",
                )

            write_status("verify_reduction")
            snap_red = collect_snapshot(paths, now=now)
            self.assertEqual(snap_red["next_command"]["stage"], "handoff_waiting")
            # Must anchor to current handoff's reduction (40s ago => 140s remaining), NOT the 3600s-old one (0s)
            self.assertAlmostEqual(snap_red["next_command"]["eta_seconds"], 140.0, places=1)

            # In verify_exit with NO handoff_off for cur_handoff_at yet, must NOT borrow old_handoff_at's handoff_off
            write_status("verify_exit")
            snap_no_off = collect_snapshot(paths, now=now)
            self.assertIsNone(snap_no_off["next_command"]["eta_seconds"])

            # Once current handoff's handoff_off echoes 30s ago -> 150s remaining
            journal_data["commands"].append({
                "room": "living",
                "kind": "mode",
                "value": "off",
                "before": ["cool", 26, "auto"],
                "sent_at": now - 33,
                "proxy_at": now - 30,
                "status": "proxy_echo",
                "purpose": "handoff_off",
                "control": {"zone": "lounge", "mode": "cool", "handoff_at": cur_handoff_at},
            })
            journal_path.write_text(json.dumps(journal_data), encoding="utf-8")
            snap_exit = collect_snapshot(paths, now=now)
            self.assertAlmostEqual(snap_exit["next_command"]["eta_seconds"], 150.0, places=1)

    def test_unreadable_private_config_reports_error_instead_of_silent_example_fallback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_dir = root / "shadow-cfg"
            cfg_dir.mkdir()
            (cfg_dir / "rooms.example.json").write_text(json.dumps({"bedroom": {"name": "Example"}}), encoding="utf-8")
            # Existing rooms.json that fails to read/parse must NOT silently fall back to rooms.example.json
            (cfg_dir / "rooms.json").write_text("{corrupt", encoding="utf-8")
            paths = DashboardPaths(
                shadow_config_dir=cfg_dir,
                pilot_data_dir=root / "pilot-data",
                shadow_data_dir=root / "shadow-data",
                shadow_output_dir=root / "climate-shadow",
                neural_output_dir=root / "neural-shadow",
            )
            snap = collect_snapshot(paths, now=1_764_300_000.0)
            self.assertTrue(any(e["file"] == "rooms.json" for e in snap["controller"]["read_errors"]))
            self.assertEqual(snap["actuators"], [])

    def test_permit_validation_matches_controller_and_respects_permit_inactive(self):
        with TemporaryDirectory() as tmp:
            paths = self._empty_paths(Path(tmp))
            paths.pilot_data_dir.mkdir(parents=True)
            now = 1_764_300_000.0
            (paths.pilot_data_dir / "status.json").write_text(
                json.dumps({"at": "2026-11-28T06:00:00+00:00", "status": "持续实机控制运行中", "rooms": {}, "control": {"at": now - 5, "topology": "site-1", "zones": {}}}),
                encoding="utf-8",
            )

            # 1. Invalid baseline tuple ("off", None, "auto") must be rejected like continuous_pilot.read_permit()
            (paths.pilot_data_dir / "live-permit.json").write_text(
                json.dumps({
                    "enabled": True, "start": int(now - 3600), "end": None,
                    "rooms": {"bedroom": ["heat", 22, "auto"], "living": ["cool", 25, "auto"], "study": ["off", None, "auto"]},
                }),
                encoding="utf-8",
            )
            snap_bad_sig = collect_snapshot(paths, now=now)
            self.assertFalse(snap_bad_sig["controller"]["permit"]["active"])
            self.assertEqual(snap_bad_sig["controller"]["health"], "paused")

            # 2. Invalid isolated_rooms entry must be rejected
            (paths.pilot_data_dir / "live-permit.json").write_text(
                json.dumps({
                    "enabled": True, "start": int(now - 3600), "end": None,
                    "isolated_rooms": ["nonexistent_room"],
                    "rooms": {"bedroom": ["heat", 22, "auto"], "living": ["cool", 25, "auto"], "study": ["cool", 25, "auto"]},
                }),
                encoding="utf-8",
            )
            snap_bad_iso = collect_snapshot(paths, now=now)
            self.assertFalse(snap_bad_iso["controller"]["permit"]["active"])

            # 3. Valid permit file, but controller itself emitted `permit_inactive` in live-audit.jsonl -> trust controller
            (paths.pilot_data_dir / "live-permit.json").write_text(
                json.dumps({
                    "enabled": True, "start": int(now - 3600), "end": None,
                    "rooms": {"bedroom": ["heat", 22, "auto"], "living": ["cool", 25, "auto"], "study": ["cool", 25, "auto"]},
                }),
                encoding="utf-8",
            )
            (paths.pilot_data_dir / "live-audit.jsonl").write_text(
                json.dumps({"at": now - 3, "event": "cycle", "outcome": "permit_inactive"}) + "\n",
                encoding="utf-8",
            )
            snap_ctrl_paused = collect_snapshot(paths, now=now)
            self.assertFalse(snap_ctrl_paused["controller"]["permit"]["active"])
            self.assertEqual(snap_ctrl_paused["controller"]["health"], "paused")
            self.assertEqual(snap_ctrl_paused["next_command"]["stage"], "paused_or_locked")

    def test_predict_zone_action_respects_capable_peer_and_energy_scores(self):
        with TemporaryDirectory() as tmp:
            paths = self._empty_paths(Path(tmp))
            paths.pilot_data_dir.mkdir(parents=True)
            now = 1_764_300_000.0
            (paths.pilot_data_dir / "live-permit.json").write_text(
                json.dumps({
                    "enabled": True, "start": int(now - 3600), "end": None,
                    "rooms": {"bedroom": ["heat", 22, "auto"], "living": ["heat", 23, "auto"], "study": ["heat", 22, "auto"]},
                }),
                encoding="utf-8",
            )
            # Study is parked (off), living is running heat@23 (still has room up to 25).
            # Even if study had a lower energy score, living is capable so dashboard must predict strengthening living, not starting study.
            (paths.pilot_data_dir / f"continuous-{int(now - 3600)}.json").write_text(
                json.dumps({
                    "start": int(now - 3600),
                    "baselines": {"bedroom": ["heat", 22, "auto"], "living": ["heat", 23, "auto"], "study": ["off", None, "auto"]},
                    "holds": {}, "manual_holds": {}, "transitions": {}, "commands": [],
                    "parked": {"study": {"off_at": now - 900, "mode": "heat", "started_at": now - 5000}},
                    "closed_loop": {},
                }),
                encoding="utf-8",
            )
            (paths.pilot_data_dir / "status.json").write_text(
                json.dumps({
                    "at": "2026-11-28T06:00:00+00:00",
                    "status": "持续实机控制运行中",
                    "rooms": {
                        "bedroom": {"room_c": 24.0, "target_c": 24.0, "mode_reported": "heat", "setpoint_reported": 22, "power_w": 120, "ready": True, "power_age_s": 5, "temperature_age_s": 20, "zone": "sleep"},
                        "living": {"room_c": 23.3, "target_c": 24.0, "mode_reported": "heat", "setpoint_reported": 23, "power_w": 180, "ready": True, "power_age_s": 5, "temperature_age_s": 20, "zone": "lounge"},
                        "study": {"room_c": 23.3, "target_c": 24.0, "mode_reported": "off", "setpoint_reported": None, "power_w": 8, "ready": True, "power_age_s": 5, "temperature_age_s": 20, "zone": "lounge"},
                    },
                    "control": {
                        "at": now - 5, "topology": "site-1", "need": "heat", "owner": "lounge",
                        "feasible": {"lounge": ["study", "living"]},
                        "energy_scores": {"study": 100.0, "living": 200.0},
                        "zones": {
                            "sleep": {"key": "sleep", "measured": 24.0, "target": 24.0, "error": 0.0, "at": now - 20, "quality": "fresh", "paused": False, "intent": "maintain", "reason": "near_target", "qualified": False},
                            "lounge": {"key": "lounge", "measured": 23.3, "target": 24.0, "error": 0.7, "at": now - 20, "quality": "fresh", "paused": False, "intent": "recover", "reason": "confirmed_bias", "direction": "heat", "qualified": True},
                        },
                    },
                }),
                encoding="utf-8",
            )
            snap = collect_snapshot(paths, now=now)
            self.assertEqual(snap["next_command"]["action"]["room"], "living")
            self.assertEqual(snap["next_command"]["action"]["purpose"], "zone_strengthen")
            self.assertEqual(snap["next_command"]["action"]["to_value"], 24)

    def test_legacy_pre_topology_mode_produces_zone_views(self):
        with TemporaryDirectory() as tmp:
            paths = self._empty_paths(Path(tmp))
            paths.pilot_data_dir.mkdir(parents=True)
            now = 1_764_300_000.0
            (paths.pilot_data_dir / "live-permit.json").write_text(
                json.dumps({"enabled": True, "start": int(now - 1000), "end": None, "rooms": {"bedroom": ["heat", 22, "auto"], "living": ["heat", 23, "auto"], "study": ["heat", 23, "auto"]}}),
                encoding="utf-8",
            )
            (paths.pilot_data_dir / "status.json").write_text(
                json.dumps({
                    "at": "2026-11-28T06:00:00+00:00",
                    "status": "持续实机控制运行中",
                    "rooms": {
                        "bedroom": {"room_c": 23.1, "target_c": 24.0, "mode_reported": "heat", "setpoint_reported": 22, "power_w": 180, "ready": True, "temperature_age_s": 30, "power_age_s": 10, "cold_seconds": 1320, "cold_confirmed": True, "warm_seconds": 0, "house_need": "heat", "status": "持续需求已确认"},
                        "living": {"room_c": 24.0, "target_c": 24.0, "mode_reported": "heat", "setpoint_reported": 23, "power_w": 120, "ready": True, "temperature_age_s": 25, "power_age_s": 10, "cold_seconds": 0, "warm_seconds": 0, "house_need": "heat", "status": "维持"},
                    },
                    "control": None,
                }),
                encoding="utf-8",
            )
            snap = collect_snapshot(paths, now=now)
            self.assertEqual(snap["controller"]["mode"], "legacy_gentle")
            zones = {z["zone"]: z for z in snap["state_machines"]["zone_states"]}
            self.assertEqual(zones["bedroom"]["state"], "recover")
            self.assertEqual(zones["living"]["state"], "maintain")

    def test_all_demo_scenarios_valid_and_distinct(self):
        stages = set()
        for item in SCENARIOS:
            snap = build_demo_snapshot(item["id"], now=1_764_300_000.0)
            self.assertEqual(snap["mode"], "demo")
            self.assertEqual(snap["scenario"], item["id"])
            self.assertTrue(snap["controller"]["data_present"])
            self.assertGreaterEqual(len(snap["state_machines"]["zone_states"]), 2)
            self.assertGreaterEqual(len(snap["actuators"]), 3)
            stages.add(snap["next_command"]["stage"])
            json.dumps(snap, allow_nan=False)
        self.assertGreaterEqual(len(stages), 5)

    def test_three_zone_star_topology_labels_and_privacy_safe_summary(self):
        three_zone_topo = json.loads((HA_DIR / "shadow" / "topology.three-zone.example.json").read_text(encoding="utf-8"))
        rooms_cfg = json.loads((HA_DIR / "shadow" / "rooms.example.json").read_text(encoding="utf-8"))
        rooms_cfg["study"]["temperature"] = "sensor.example_office_temperature"

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_dir = root / "shadow-cfg"
            cfg_dir.mkdir()
            (cfg_dir / "house_policy.json").write_text((HA_DIR / "shadow" / "house_policy.json").read_text(encoding="utf-8"), encoding="utf-8")
            (cfg_dir / "rooms.json").write_text(json.dumps(rooms_cfg), encoding="utf-8")
            (cfg_dir / "topology.json").write_text(json.dumps(three_zone_topo), encoding="utf-8")
            pilot_dir = root / "pilot-data"
            pilot_dir.mkdir()
            now = 1_764_300_000.0
            (pilot_dir / "live-permit.json").write_text(
                json.dumps({
                    "enabled": True, "start": int(now - 3600), "end": None,
                    "rooms": {"bedroom": ["off", 24, "auto"], "living": ["cool", 25, "auto"], "study": ["cool", 24, "auto"]},
                }),
                encoding="utf-8",
            )
            (pilot_dir / "status.json").write_text(
                json.dumps({
                    "at": "2026-11-28T09:00:00+00:00",
                    "status": "持续实机控制运行中",
                    "rooms": {
                        "bedroom": {"room_c": 24.0, "target_c": 24.0, "mode_reported": "off", "setpoint_reported": None, "power_w": 5, "ready": True, "power_age_s": 10, "temperature_age_s": 20, "zone": "sleep"},
                        "living": {"room_c": 24.4, "target_c": 24.0, "mode_reported": "cool", "setpoint_reported": 25, "power_w": 150, "ready": True, "power_age_s": 10, "temperature_age_s": 20, "zone": "lounge"},
                        "study": {"room_c": 25.4, "target_c": 24.0, "mode_reported": "cool", "setpoint_reported": 24, "power_w": 220, "ready": True, "power_age_s": 10, "temperature_age_s": 20, "zone": "office"},
                    },
                    "control": {
                        "at": now - 5, "topology": "site-star-1", "need": "cool", "owner": "office", "lead": "study",
                        "zones": {
                            "sleep": {"key": "sleep", "measured": 24.0, "target": 24.0, "error": 0.0, "at": now - 20, "quality": "fresh", "paused": False, "intent": "maintain", "reason": "near_target", "qualified": False},
                            "lounge": {"key": "lounge", "measured": 24.4, "target": 24.0, "error": -0.4, "at": now - 20, "quality": "fresh", "paused": False, "intent": "observe", "reason": "waiting_reports_or_dwell", "qualified": False},
                            "office": {"key": "office", "measured": 25.4, "target": 24.0, "error": -1.4, "at": now - 20, "quality": "fresh", "paused": False, "intent": "recover", "reason": "confirmed_bias", "direction": "cool", "qualified": True},
                        },
                    },
                }),
                encoding="utf-8",
            )
            paths = DashboardPaths(
                shadow_config_dir=cfg_dir,
                pilot_data_dir=pilot_dir,
                shadow_data_dir=root / "shadow-data",
                shadow_output_dir=root / "climate-shadow",
                neural_output_dir=root / "neural-shadow",
            )
            snap = collect_snapshot(paths, now=now)
            zone_map = {z["zone"]: z for z in snap["state_machines"]["zone_states"]}
            self.assertEqual(zone_map["sleep"]["label"], "卧室区 (sleep)")
            self.assertEqual(zone_map["lounge"]["label"], "客厅区 (lounge)")
            self.assertEqual(zone_map["office"]["label"], "书房区 (office)")
            self.assertTrue(zone_map["office"]["is_owner"])

            act_map = {a["room"]: a for a in snap["actuators"]}
            self.assertEqual(act_map["bedroom"]["lock_group"], "bedroom")
            self.assertEqual(act_map["living"]["lock_group"], "living")
            self.assertEqual(act_map["study"]["lock_group"], "study")
            self.assertEqual(set(act_map["bedroom"]["influences"]), {"sleep", "lounge"})
            self.assertEqual(set(act_map["living"]["influences"]), {"lounge", "sleep", "office"})
            self.assertEqual(set(act_map["study"]["influences"]), {"office", "lounge"})

            topo = snap["topology"]
            self.assertTrue(topo["present"])
            self.assertFalse(topo["from_example"])
            self.assertFalse(topo["opposite_operation"]["enabled"])
            pairs = {tuple(sorted((c["from"], c["to"]))) for c in topo["connections"]}
            self.assertEqual(pairs, {("lounge", "sleep"), ("lounge", "office")})
            self.assertNotIn("sensor.example_", json.dumps(topo))

            # Two-zone fallback must still label lounge as shared living/study when office is absent
            demo_snap = build_demo_snapshot("recover_strengthen", now=now)
            demo_zones = {z["zone"]: z for z in demo_snap["state_machines"]["zone_states"]}
            self.assertEqual(demo_zones["lounge"]["label"], "客厅 / 书房区 (lounge)")

            # Live 3-zone status without mounted topology.json must not falsely adopt 2-zone topology.example.json
            (cfg_dir / "topology.json").unlink()
            (cfg_dir / "topology.example.json").write_text(EXAMPLE_TOPOLOGY.read_text(encoding="utf-8"), encoding="utf-8")
            snap_no_topo = collect_snapshot(paths, now=now)
            self.assertFalse(snap_no_topo["topology"]["present"])
            no_topo_zones = {z["zone"]: z for z in snap_no_topo["state_machines"]["zone_states"]}
            self.assertEqual(no_topo_zones["lounge"]["label"], "客厅区 (lounge)")
            self.assertEqual(no_topo_zones["office"]["label"], "书房区 (office)")


class DashboardHTTPServerTests(unittest.TestCase):
    def _start_server(self, config: ServerConfig):
        srv = create_server(config)
        port = srv.server_address[1]
        thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        thread.start()

        def cleanup():
            srv.shutdown_event.set()
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=3)

        self.addCleanup(cleanup)
        return port

    def _request(self, port: int, method: str, path: str):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request(method, path)
            resp = conn.getresponse()
            body = resp.read()
            headers = dict(resp.getheaders())
            return resp.status, headers, body
        finally:
            conn.close()

    def test_demo_server_routes_sse_and_read_only_enforcement(self):
        port = self._start_server(ServerConfig(host="127.0.0.1", port=0, demo=True, stream_interval_s=0.5))

        # Static assets & security headers
        for route, expected_ct in (
            ("/", "text/html"),
            ("/assets/styles.css", "text/css"),
            ("/assets/app.js", "application/javascript"),
            ("/assets/favicon.svg", "image/svg+xml"),
        ):
            status, headers, body = self._request(port, "GET", route)
            self.assertEqual(status, 200, route)
            self.assertIn(expected_ct, headers.get("Content-Type", ""))
            self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")
            self.assertEqual(headers.get("X-Frame-Options"), "SAMEORIGIN")
            self.assertIn("default-src 'self'", headers.get("Content-Security-Policy", ""))
            self.assertGreater(len(body), 20)

        # Snapshot API + scenario switcher
        status, _, body = self._request(port, "GET", "/api/snapshot?scenario=handoff_conflict")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["mode"], "demo")
        self.assertEqual(data["scenario"], "handoff_conflict")
        self.assertTrue(data["state_machines"]["handoff"]["active"])
        self.assertEqual(data["state_machines"]["handoff"]["phase"], "verify_reduction")

        # Health endpoint
        status, _, body = self._request(port, "GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ok"])

        # Mutations strictly rejected with 405
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            status, headers, _ = self._request(port, method, "/api/snapshot")
            self.assertEqual(status, 405, method)
            self.assertEqual(headers.get("Allow"), "GET, HEAD")

        # Path traversal rejected
        for bad_path in ("/assets/../server.py", "/assets/%2e%2e/server.py", "/etc/passwd"):
            status, _, _ = self._request(port, "GET", bad_path)
            self.assertEqual(status, 404, bad_path)

        # SSE stream returns snapshot event frame
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", "/api/stream?scenario=inflight_echo")
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            self.assertIn("text/event-stream", resp.getheader("Content-Type", ""))
            first_event = resp.readline().decode("utf-8").strip()
            first_data = resp.readline().decode("utf-8").strip()
            self.assertEqual(first_event, "event: snapshot")
            self.assertTrue(first_data.startswith("data: "))
            frame = json.loads(first_data.removeprefix("data: "))
            self.assertEqual(frame["scenario"], "inflight_echo")
            self.assertEqual(frame["next_command"]["stage"], "in_flight")
        finally:
            conn.close()

    def test_sse_client_limit_and_snapshot_cache(self):
        port = self._start_server(
            ServerConfig(host="127.0.0.1", port=0, demo=True, stream_interval_s=1.0, cache_ttl_s=5.0, max_sse_clients=1)
        )
        # Rapid /api/snapshot calls within cache_ttl_s share the same cached snapshot timestamp
        _, _, b1 = self._request(port, "GET", "/api/snapshot")
        _, _, b2 = self._request(port, "GET", "/api/snapshot")
        self.assertEqual(json.loads(b1)["generated_at"], json.loads(b2)["generated_at"])

        # Open 1st SSE stream (consumes the single slot), verify 2nd concurrent SSE stream receives 503
        c1 = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            c1.request("GET", "/api/stream")
            r1 = c1.getresponse()
            self.assertEqual(r1.status, 200)
            self.assertEqual(r1.readline().decode("utf-8").strip(), "event: snapshot")
            r1.readline()  # consume data line

            status2, headers2, _ = self._request(port, "GET", "/api/stream")
            self.assertEqual(status2, 503)
            self.assertEqual(headers2.get("Retry-After"), "5")
        finally:
            c1.close()

    def test_live_server_ignores_scenario_param(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = DashboardPaths(
                shadow_config_dir=HA_DIR / "shadow",
                pilot_data_dir=root / "pilot-data",
                shadow_data_dir=root / "shadow-data",
                shadow_output_dir=root / "climate-shadow",
                neural_output_dir=root / "neural-shadow",
            )
            port = self._start_server(ServerConfig(host="127.0.0.1", port=0, paths=paths, demo=False))
            status, _, body = self._request(port, "GET", "/api/snapshot?scenario=handoff_conflict")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(data["mode"], "live")
            self.assertIsNone(data["scenario"])
            self.assertFalse(data["controller"]["data_present"])


class DashboardIsolationAuditTests(unittest.TestCase):
    def test_ast_zero_actuation_and_no_controller_writes(self):
        dashboard_dir = HA_DIR / "dashboard"
        forbidden_imports = {"urllib.request", "http.client", "requests", "httpx", "aiohttp"}
        forbidden_literals = ("services/climate", "HOMEASSISTANT_TOKEN", "ha_client", "set_temperature", "set_hvac_mode")

        for py_file in sorted(dashboard_dir.glob("*.py")):
            src = py_file.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name, forbidden_imports, f"{py_file.name} imports {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    self.assertNotIn(mod, forbidden_imports, f"{py_file.name} imports from {mod}")
                if py_file.name in {"collector.py", "server.py"} and isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute) and node.func.attr in {"write_text", "write_bytes", "unlink", "replace"}:
                        self.fail(f"{py_file.name} must remain strictly read-only, found .{node.func.attr}()")
            # Collector/server must never contain HA actuation endpoints
            if py_file.name != "collector.py":
                for lit in forbidden_literals:
                    self.assertNotIn(lit, src, f"{py_file.name} contains forbidden actuation literal {lit}")
            else:
                for lit in ("services/climate", "HOMEASSISTANT_TOKEN", "ha_client"):
                    self.assertNotIn(lit, src, f"{py_file.name} contains forbidden actuation literal {lit}")

    def test_compose_dashboard_strictly_readonly_and_hardened(self):
        compose_path = HA_DIR / "compose.dashboard.yaml"
        self.assertTrue(compose_path.is_file())
        text = compose_path.read_text(encoding="utf-8")
        non_comment = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        self.assertIn("read_only: true", non_comment)
        self.assertIn("cap_drop: [ALL]", non_comment)
        self.assertNotIn("cap_add", non_comment)
        self.assertNotIn("DAC_READ_SEARCH", non_comment)
        self.assertNotIn("DAC_OVERRIDE", non_comment)
        self.assertIn("no-new-privileges:true", non_comment)
        self.assertIn("127.0.0.1", non_comment)
        self.assertNotIn("private", non_comment)
        self.assertNotIn("home-assistant.env", non_comment)
        self.assertNotIn("homeassistant_default", non_comment)

        in_volumes = False
        mounts = []
        for line in non_comment.splitlines():
            stripped = line.strip()
            if stripped == "volumes:":
                in_volumes = True
                continue
            if in_volumes:
                if stripped.startswith("- "):
                    mounts.append(stripped[2:].strip())
                elif stripped:
                    in_volumes = False
        self.assertGreaterEqual(len(mounts), 4)
        for m in mounts:
            self.assertTrue(m.endswith(":ro"), f"Dashboard volume mount must be read-only: {m}")

    def test_controller_codes_covered_by_dashboard_labels(self):
        from dashboard import collector

        def value_strings(node: ast.AST) -> set[str]:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return {node.value}
            if isinstance(node, ast.IfExp):
                return value_strings(node.body) | value_strings(node.orelse)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and len(node.args) >= 2:
                return value_strings(node.args[1])
            return set()

        purposes: set[str] = set()
        outcomes: set[str] = set()
        actuator_reasons: set[str] = set()
        handoff_phases: set[str] = set()
        handoff_reasons: set[str] = set()

        for filename in ("zone_control.py", "continuous_pilot.py", "house_dispatch.py"):
            tree = ast.parse((HA_DIR / "shadow" / filename).read_text(encoding="utf-8"), filename=filename)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "append":
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "candidates" and node.args:
                        tup = node.args[0]
                        if isinstance(tup, ast.Tuple) and len(tup.elts) == 6:
                            purposes |= {s for s in value_strings(tup.elts[5]) if s.isascii()}
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "audit":
                    for kw in node.keywords:
                        if kw.arg == "outcome":
                            outcomes |= value_strings(kw.value)
                if filename == "zone_control.py" and isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and tgt.value.id == "reasons":
                            actuator_reasons |= {s for s in value_strings(node.value) if s.isascii() and "_" in s}
                if filename == "zone_control.py" and isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Name) and node.func.id == "dict") or (
                        isinstance(node.func, ast.Attribute) and node.func.attr == "update"
                    ):
                        kw_map = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                        if "phase" in kw_map:
                            handoff_phases |= value_strings(kw_map["phase"])
                            if "reason" in kw_map:
                                handoff_reasons |= value_strings(kw_map["reason"])

        self.assertTrue(purposes)
        self.assertTrue(outcomes)
        self.assertTrue(actuator_reasons)
        self.assertTrue(handoff_phases)
        self.assertEqual(purposes - set(collector.PURPOSE_LABELS), set())
        self.assertEqual(outcomes - set(collector.OUTCOME_LABELS), set())
        all_reason_labels = set(collector.ACTUATOR_STATUS_LABELS) | set(collector.HANDOFF_REASON_LABELS) | set(collector.ZONE_REASON_LABELS)
        self.assertEqual(actuator_reasons - all_reason_labels, set())
        self.assertEqual(handoff_phases - set(collector.HANDOFF_PHASE_LABELS), set())
        self.assertEqual(handoff_reasons - set(collector.HANDOFF_REASON_LABELS), set())


if __name__ == "__main__":
    unittest.main()
