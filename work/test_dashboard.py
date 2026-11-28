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
        self.assertIn("read_only: true", text)
        self.assertIn("cap_drop: [ALL]", text)
        self.assertIn("no-new-privileges:true", text)
        self.assertNotIn("private", text)
        self.assertNotIn("home-assistant.env", text)
        self.assertNotIn("homeassistant_default", text)

        in_volumes = False
        mounts = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "volumes:":
                in_volumes = True
                continue
            if in_volumes:
                if stripped.startswith("- "):
                    mounts.append(stripped[2:].strip())
                elif stripped and not stripped.startswith("#"):
                    in_volumes = False
        self.assertGreaterEqual(len(mounts), 4)
        for m in mounts:
            self.assertTrue(m.endswith(":ro"), f"Dashboard volume mount must be read-only: {m}")


if __name__ == "__main__":
    unittest.main()
