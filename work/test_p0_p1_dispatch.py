"""Run policy, final gate and dispatch using an isolated journal and fake HA."""
from contextlib import ExitStack, closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "home-assistant/shadow"))
import continuous_pilot as pilot
import pilot_gate
from model import numeric
from house_policy import recent_trend, temperature_change
from service import init, metered_interval, power_evidence
from steady_learning import initial_setpoint


TICK = 12000
NOW = TICK + 10


class DispatchTests(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(TemporaryDirectory())
        self.root = Path(tmp)
        self.db = self.enterContext(closing(sqlite3.connect(self.root / "shadow.sqlite3")))
        self.db.row_factory = sqlite3.Row
        init(self.db, TICK)
        self.permit = self.root / "permit.json"
        self.calls = []
        self.live = {}
        self.now = NOW
        self.auto_echo = False
        stack = self.enterContext(ExitStack())
        for namespace, name, value in ((pilot, "DATA_DIR", self.root), (pilot, "LIVE_PERMIT", self.permit),
                                       (pilot_gate, "DATA", self.root)):
            stack.enter_context(patch.object(namespace, name, value))
        stack.enter_context(patch.object(pilot, "time", side_effect=lambda: self.now))
        stack.enter_context(patch.object(pilot_gate, "time", side_effect=lambda: self.now))
        stack.enter_context(patch.object(pilot, "publish"))
        stack.enter_context(patch.object(pilot, "room_feedback", side_effect=self.feedback))

    def feedback(self, call, room):
        return deepcopy(self.live[room])

    def request(self, path, fields):
        self.calls.append((path, fields))
        if self.auto_echo:
            room = next(r for r, c in pilot.ROOMS.items() if c["adapter"] == fields["entity_id"])
            journal = json.loads(self.journal.read_text())
            self.assertEqual(journal["commands"][-1]["status"], "prepared")
            v = self.live[room]
            mode, setting, fan, *rest = v["signature"]
            if "hvac_mode" in fields:
                mode = fields["hvac_mode"]
                setting = None if mode == "off" else 27
                v["adapter"]["last_mode_change"] = self.now
            else:
                setting = fields["temperature"]
            v["signature"] = (mode, setting, fan, *rest)
            v["adapter"].update(mode=mode, setpoint=setting, updated_at=self.now)
        return 200

    @property
    def journal(self):
        return self.root / f"continuous-{TICK-4000}.json"

    def scenario(self, mode, setting, temperature, watts, *, high=False, age=10, manual=False,
                 preset="标准", cooling=False, new_context=False):
        context = [pilot.ROOMS["bedroom"]["power"], "1970-01-01T00:00:00+00:00"]
        sig = (mode, setting, "auto")
        self.live = {room: {"signature": (*sig, "auto" if room == "bedroom" else "off", 24, preset, False),
                            "temperature": temperature, "temperature_age": 15, "native_authorized": False,
                            "offsets": {"heat": 0, "cool": 0}, "power_context": context,
                            "temperature_report_at": TICK, "temperature_source": ["sensor.fixture", context[1]],
                            "adapter": {"mode": mode, "setpoint": setting, "power": watts,
                                        "from_power_s": age, "last_mode_change": TICK-4000,
                                        "power_state_updated_at": TICK-30, "modes": ["heat", "cool", "off"]}}
                     for room in pilot.ROOMS}
        if new_context:
            self.live["bedroom"]["power_context"] = [context[0], "1970-01-01T00:00:01+00:00"]
        baseline = dict.fromkeys(self.live, sig)
        if manual:
            baseline["bedroom"] = (mode, setting+1, "auto")
        self.permit.write_text(json.dumps({"enabled": True, "start": TICK-4000, "end": None, "rooms": baseline,
                                           "isolated_rooms": ["living", "study"]}))
        with self.db:
            self.db.execute("DELETE FROM decisions")
            self.db.execute("DELETE FROM trials")
            self.db.execute("INSERT INTO trials VALUES (?,?,?,?)",
                            (pilot_gate.WITNESSES["bedroom"], "bedroom", TICK-1000,
                             json.dumps({"sent_at": TICK-1000, "physical": "matched", "completed_followup": True,
                                         "beeps_observed": 0})))
            if cooling:
                self.db.execute("INSERT INTO trials VALUES (?,?,?,?)",
                                ("beep", "bedroom", TICK-60, json.dumps({"sent_at": TICK-60, "beep_at": TICK-60})))
            evidence = power_evidence(TICK-180, [(TICK-180, numeric(watts))], context, {"high": 190}, None)
            evidence = power_evidence(TICK, [(t, numeric(watts)) for t in range(TICK-180, TICK+1, 60)],
                                      context, {"high": 190}, evidence)
            for stamp in range(TICK-1200, TICK+1, 60):
                measured = temperature - (TICK-stamp)*.4/600 if high else temperature
                decision = {"target": 24, "quiet": False, "reported_mode": mode, "from_setpoint": setting,
                            "measured": measured, "source_fresh": True, "temperature_report_at": stamp,
                            "temperature_source": self.live["bedroom"]["temperature_source"],
                            "outdoor": 14, "activity": "fixture", "confidence": "推测·中" if high else "未知",
                            "power_reason": "fixture", "model": {"edges_w": [190]}, "power_report_at": stamp,
                            "power_report_w": numeric(watts), "above_high_seconds": evidence["bands"]["high"]["seconds"],
                            "power_evidence": evidence, "low_power_cutoff": 30,
                            "power_state_updated_at": TICK-30}
                self.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?)",
                                ("bedroom", stamp, "1970-01-01", 0, json.dumps(decision)))

    def test_direct_high_advice_passes_real_gate_and_sends_once(self):
        self.scenario("heat", 24, 24.5, 230, high=True)
        live = self.live["bedroom"]
        shadow = pilot_gate.room_advice("bedroom", "heat", offsets={}, feedback=live)
        self.assertTrue(shadow["decision"]["direct_high_overshoot"])
        self.assertEqual(pilot_gate.eligible(NOW, live["signature"], live, shadow, setpoints=(23,))[1], 23)
        pilot.cycle(self.request)
        pilot.cycle(self.request)
        self.assertEqual(self.calls, [("services/climate/set_temperature",
                                      {"entity_id": pilot.ROOMS["bedroom"]["adapter"], "temperature": 23})])

    def test_high_advice_drops_when_live_power_falls_after_suggestion(self):
        self.scenario("heat", 24, 24.5, 230, high=True)
        reads = 0
        def lose_power(call, room):
            nonlocal reads
            data = self.feedback(call, room)
            if room == "bedroom":
                reads += 1
                if reads > 1:
                    data["adapter"]["power"] = 30
            return data
        with patch.object(pilot, "room_feedback", side_effect=lose_power):
            pilot.cycle(self.request)
        self.assertGreaterEqual(reads, 2)
        self.assertEqual(self.calls, [])

    def test_direct_high_rechecks_freshness_context_and_ignores_beep_budget(self):
        for settings in ({"age": 121}, {"new_context": True}):
            with self.subTest(settings=settings):
                self.scenario("heat", 24, 24.5, 230, high=True, **settings)
                pilot.cycle(self.request)
                self.assertEqual(self.calls, [])
        self.scenario("heat", 24, 24.5, 230, high=True, cooling=True)
        pilot.cycle(self.request)
        self.assertEqual(len(self.calls), 1)

    def test_low_power_can_increase_heating_and_cooling_through_dispatch(self):
        for args, expected in ((("heat", 20, 23, 20), 21), (("cool", 26, 25, 20), 25)):
            with self.subTest(mode=args[0]):
                self.scenario(*args)
                pilot.cycle(self.request)
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(self.calls.pop()[1]["temperature"], expected)
                (self.root / f"continuous-{TICK-4000}.json").unlink()

    def test_low_power_increase_retains_safety_gates(self):
        fixtures = [
            (("heat", 20, 23, 20), {"age": 601}),
            (("heat", 20, 23, "unavailable"), {}),
            (("heat", 20, 23, 20), {"manual": True}),
            (("heat", 20, 23, 20), {"preset": "睡眠静音"}),
            (("heat", 25, 23, 20), {}),
            (("heat", 24, 25, 20), {}),
        ]
        for args, settings in fixtures:
            with self.subTest(args=args, settings=settings):
                self.scenario(*args, **settings)
                pilot.cycle(self.request)
                self.assertEqual(self.calls, [])
                (self.root / f"continuous-{TICK-4000}.json").unlink()

    def test_initial_setting_follows_target_and_intersects_both_limits(self):
        for room in ("bedroom", "study"):
            for target, expected in ((22, 22), (24, 24), (26, 25)):
                self.assertEqual(initial_setpoint(self.db, room, "heat", NOW, target, (20, 25)), expected)
        self.assertEqual(initial_setpoint(self.db, "bedroom", "heat", NOW, 31, (14, 35)), 30)
        self.assertEqual(initial_setpoint(self.db, "bedroom", "cool", NOW, 15, (12, 26)), 16)
        self.assertIsNone(initial_setpoint(self.db, "bedroom", "heat", NOW, 24, (31, 35)))
        with self.db:
            self.db.execute("INSERT INTO stable_episodes VALUES ('bedroom','heat',?,?,?)", (TICK-1200, TICK, 4))
            self.assertEqual(initial_setpoint(self.db, "bedroom", "heat", NOW, 22, (20, 25)), 22)
            self.db.execute("INSERT INTO stable_episode_quality VALUES ('bedroom','heat',?,?,?)", (TICK, 22, 3))
        self.assertEqual(initial_setpoint(self.db, "bedroom", "heat", NOW, 22, (20, 25)), 23)

    def whole_scene(self, bedroom, living, study=None):
        self.scenario(*bedroom)
        self.auto_echo = True
        specs = {"bedroom": bedroom, "living": living, "study": study or ("heat", 24, living[2], 20)}
        data = json.loads(self.permit.read_text())
        data["isolated_rooms"] = [] if study else ["study"]
        for room, (mode, setting, temperature, power) in specs.items():
            v = self.live[room]
            v["signature"] = (mode, setting, "auto", "auto", 24, "标准", False)
            v["temperature"] = temperature
            v["adapter"].update(mode=mode, setpoint=setting, power=power)
            v["temperature_source"] = ["sensor." + ("living" if room == "study" else room), v["power_context"][1]]
            data["rooms"][room] = list(v["signature"][:3])
            with self.db:
                self.db.execute("INSERT OR IGNORE INTO trials VALUES (?,?,?,?)",
                                (pilot_gate.WITNESSES[room], room, TICK-1000,
                                 json.dumps({"sent_at": TICK-1000, "physical": "matched",
                                             "completed_followup": True, "beeps_observed": 0})))
            self.observe(room)
        self.permit.write_text(json.dumps(data))

    def observe(self, room, *, change=0, minutes=32, new_report=True, cadence=60, high_seconds=0, history=None):
        tick = self.now // 60 * 60
        v = self.live[room]
        adapter = v["adapter"]
        if new_report:
            v["temperature_report_at"] = tick
        v["temperature_age"] = self.now-v["temperature_report_at"]
        adapter["from_power_s"] = self.now-tick
        with self.db:
            self.db.execute("DELETE FROM decisions WHERE room=?", (room,))
            for stamp in range(tick-minutes*60, tick+1, 60):
                report = min(stamp, v["temperature_report_at"])
                report = tick - ((tick-report+cadence-1)//cadence)*cadence
                value = {"target": v["signature"][4], "quiet": False, "reported_mode": adapter["mode"],
                         "from_setpoint": adapter["setpoint"],
                         "measured": history.get(stamp, v["temperature"]) if history else
                                     v["temperature"]-change*(tick-report)/600,
                         "source_fresh": True, "temperature_report_at": report,
                         "temperature_source": v["temperature_source"], "outdoor": 14, "activity": "fixture",
                         "confidence": "推测·中", "power_reason": "fixture", "model": {"edges_w": [190]},
                         "power_report_at": stamp, "power_report_w": adapter["power"], "low_power_cutoff": 30,
                         "power_evidence": {"context": v["power_context"], "last_report": tick,
                                            "bands": {"low": {"since": tick-900},
                                                      "high": {"seconds": high_seconds, "threshold": 190}}},
                         "reversal_low_cutoff": 28, "power_state_updated_at": adapter["power_state_updated_at"]}
                self.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?)",
                                (room, stamp, "1970-01-01", 0, json.dumps(value)))

    def snapshot(self):
        pilot.cycle(self.request)
        return json.loads(self.journal.read_text())

    def test_mixed_modes_inside_comfort_exit_the_opposite_side_gradually(self):
        self.whole_scene(("heat", 24, 24, 240), ("cool", 24, 24, 240))
        state = self.snapshot()
        self.assertEqual((state["commands"][-1]["room"], state["commands"][-1]["value"]), ("living", 25))
        self.snapshot()
        self.assertEqual(len(self.calls), 1)
        self.now += 660
        self.observe("bedroom")
        self.observe("living")
        state = self.snapshot()
        self.assertEqual((state["commands"][-1]["purpose"], state["commands"][-1]["value"]), ("energy_off", "off"))
        parked = self.snapshot()
        self.assertEqual(parked["baselines"]["living"][0], "off")
        self.assertEqual(parked["parked"]["living"]["mode"], "cool")

    def test_conflicting_zones_are_not_averaged_and_study_isolation_is_manual(self):
        self.whole_scene(("heat", 23, 22.8, 20), ("cool", 24, 25.2, 240), ("cool", 24, 25.2, 240))
        self.snapshot()
        self.assertEqual(self.calls, [])
        permit = json.loads(self.permit.read_text())
        permit["isolated_rooms"] = ["living", "study"]
        self.permit.write_text(json.dumps(permit))
        state = self.snapshot()
        self.assertEqual(state["commands"][-1]["room"], "bedroom")
        self.assertEqual(len(self.calls), 1)

    def test_cold_room_already_warming_does_not_increase(self):
        self.whole_scene(("heat", 23, 23.1, 20), ("heat", 24, 24, 240))
        self.observe("bedroom", change=.4)
        self.snapshot()
        self.assertEqual(self.calls, [])

    def test_new_feedback_blocks_cross_unit_escalation_and_allows_assist_later(self):
        self.whole_scene(("heat", 24, 22.8, 20), ("heat", 23, 23.0, 20))
        sent = self.snapshot()
        self.assertEqual((sent["commands"][-1]["room"], sent["commands"][-1]["value"]), ("bedroom", 25))
        self.snapshot()
        self.now += 660
        self.observe("bedroom", minutes=9)
        self.observe("living", new_report=False)
        self.snapshot()
        self.assertEqual(len(self.calls), 1)
        self.observe("living")
        # Supporting the same established demand is now based on a new report from both references.
        state = self.snapshot()
        self.assertEqual((state["commands"][-1]["room"], state["commands"][-1]["value"]), ("living", 24))

    def test_energy_off_persists_and_recovers_for_new_need_not_old_plan(self):
        self.whole_scene(("heat", 24, 24, 200), ("heat", 25, 24, 200))
        sent = self.snapshot()
        self.assertEqual(sent["commands"][-1]["purpose"], "coordinated_easing")
        self.snapshot()
        self.now += 660
        self.observe("bedroom", minutes=9)
        self.observe("living", minutes=9)
        self.snapshot()
        state = self.snapshot()
        self.assertIn("living", state["parked"])
        self.live["living"]["adapter"]["power"] = 15
        self.now += 2000
        self.live["bedroom"]["temperature"] = 25.4
        self.live["living"]["temperature"] = 25.3
        self.observe("bedroom")
        self.observe("living")
        self.snapshot()
        # The opposing active unit must first be reduced or stopped, never joined by cooling.
        self.assertFalse(any(fields.get("hvac_mode") == "cool" for _, fields in self.calls))
        self.live["bedroom"]["temperature"] = 24
        self.live["living"]["temperature"] = 22.9
        self.live["bedroom"]["adapter"]["power"] = 20
        self.now += 2100
        # Set up a now saturated, non-improving primary and retain the parked state.
        self.live["bedroom"]["signature"] = ("heat", 25, *self.live["bedroom"]["signature"][2:])
        self.live["bedroom"]["adapter"]["setpoint"] = 25
        state = json.loads(self.journal.read_text())
        state["baselines"]["bedroom"] = ["heat", 25, "auto"]
        self.journal.write_text(json.dumps(state))
        self.observe("bedroom")
        self.observe("living")
        state = self.snapshot()
        self.assertEqual((state["commands"][-1]["purpose"], state["commands"][-1]["value"]), ("energy_start", "heat"))
        # A paused continuation preserves the confirmed mode. The new target replaces the unsent initial setting.
        v = self.live["living"]
        v["signature"] = (*v["signature"][:4], 22, "睡眠静音", False)
        self.now += 60
        self.observe("living")
        self.observe("bedroom")
        state = self.snapshot()
        self.assertEqual(state["baselines"]["living"][0], "heat")
        self.assertNotIn("living", state["parked"])
        self.assertEqual(state["transitions"]["living_study"]["members"]["living"]["phase"], "initial")
        v["signature"] = (*v["signature"][:5], "标准", False)
        state = self.snapshot()
        self.assertEqual((state["commands"][-1]["purpose"], state["commands"][-1]["value"]),
                         ("initial_safety_easing", 22))
        self.assertTrue(state["transitions"]["living_study"]["members"]["living"]["initial_invalidated"])
        self.assertEqual(state["baselines"]["living"], ["heat", 27, "auto"])
        state = self.snapshot()
        self.assertNotIn("living_study", state["transitions"])
        self.assertEqual(state["baselines"]["living"], ["heat", 22, "auto"])

    def test_reduction_returns_out_of_range_setting_to_its_soft_boundary_only(self):
        self.whole_scene(("heat", 27, 24.3, 240), ("heat", 24, 24, 200))
        self.observe("bedroom", change=.6)
        state = self.snapshot()
        self.assertEqual((state["commands"][-1]["room"], state["commands"][-1]["value"]), ("bedroom", 25))

    def test_early_weakening_from_below_target_and_slow_recovery_does_not_wait_forever(self):
        early = {"trend": .9, "cold_seconds": 0, "warm_seconds": 0}
        self.assertEqual(temperature_change("heat", 24, 23.8, 24, early, (20, 25), has_output=True)[0], 23)
        cold = {"trend": .16, "cold_seconds": 1800, "warm_seconds": 0}
        self.assertEqual(temperature_change("heat", 22, 21.8, 24, cold, (20, 25), has_output=False)[0], 23)
        cold["trend"] = 0
        self.assertEqual(temperature_change("heat", 22, 21.8, 24, cold, (20, 25), has_output=False)[0], 24)
        cold["trend"] = None
        self.assertEqual(temperature_change("heat", 22, 21.8, 24, cold, (20, 25), has_output=False)[0], 23)

    def test_trend_does_not_cross_control_or_temperature_source_boundary(self):
        end = 900
        points = [{"tick": t, "temp_report": t, "measured": 24+t/6000, "adapter_mode": "heat",
                   "reported": 24, "desired": 24, "temperature_source": ["sensor.one", "boot"]}
                  for t in range(60, end+1, 60)]
        self.assertIsNotNone(recent_trend(points, points[-1], end, 0))
        self.assertIsNone(recent_trend(points, points[-1], end, 700))
        for change in ({"adapter_mode": "cool"}, {"desired": 25}, {"reported": 25},
                       {"temperature_source": ["sensor.two", "boot"]}):
            fresh = dict(points[-1], **change)
            self.assertIsNone(recent_trend(points, fresh, end, 0))

    def test_energy_observation_does_not_fill_power_gaps(self):
        power = [(0, 600), (60, 600), (300, 600), (360, 600)]
        self.assertEqual(metered_interval(power, 0, 360), (.02, 120))

    def test_mode_success_then_initial_http_500_leaves_confirmed_baseline(self):
        self.whole_scene(("heat", 25, 24, 20), ("heat", 24, 22.8, 15))
        v = self.live["living"]
        v["signature"] = ("off", None, *v["signature"][2:])
        v["adapter"].update(mode="off", setpoint=None)
        self.journal.write_text(json.dumps({"start": TICK-4000, "holds": {}, "commands": [],
                              "baselines": {r: list(x["signature"][:3]) for r, x in self.live.items()},
                              "parked": {"living": {"off_at": NOW-900, "mode": "heat", "started_at": NOW-4000}}}))
        self.observe("living")
        sent = self.snapshot()
        self.assertEqual(sent["commands"][-1]["purpose"], "energy_start")
        self.now += 60
        self.observe("living")
        self.observe("bedroom")
        def fail(path, fields):
            self.assertIn("temperature", fields)
            raise HTTPError("http://fixture", 500, "fixture", {}, None)
        with patch.object(self, "request", side_effect=fail) as failed:
            state = self.snapshot()
            self.snapshot()
            self.assertEqual(failed.call_count, 1)
        self.assertEqual(state["commands"][-1]["status"], "unconfirmed")
        self.assertEqual(state["baselines"]["living"], ["heat", 27, "auto"])
        self.assertIn("living_study", state["holds"])
        self.assertEqual(state["transitions"]["living_study"]["members"]["living"]["phase"], "initial")

    def test_unconfirmed_prepared_is_locked_after_restart_without_duplicate(self):
        self.scenario("heat", 20, 23, 20)
        first = self.snapshot()
        self.assertEqual(first["commands"][0]["status"], "accepted")
        first["commands"][0]["status"] = "prepared"  # Simulate crash before persisting HTTP result.
        self.journal.write_text(json.dumps(first))
        recovered = self.snapshot()
        self.assertEqual(recovered["commands"][0]["status"], "unconfirmed")
        self.assertIn("bedroom", recovered["holds"])
        self.assertEqual(recovered["baselines"]["bedroom"], ["heat", 20, "auto"])
        self.assertEqual(len(self.calls), 1)

    def test_mixed_mode_shutdown_cannot_bypass_minimum_runtime(self):
        self.whole_scene(("heat", 24, 24, 240), ("cool", 26, 24, 240))
        self.live["living"]["adapter"]["last_mode_change"] = NOW - 100
        self.snapshot()
        self.assertEqual(self.calls, [])

    def test_permission_removed_in_second_assessment_never_dispatches(self):
        self.scenario("heat", 20, 23, 20)
        read = pilot.read_permit
        first = read(NOW)
        with patch.object(pilot, "read_permit", side_effect=(first, None)):
            self.snapshot()
        self.assertEqual(self.calls, [])

    def test_normal_mode_change_needs_measured_post_shutdown_low_segment(self):
        self.whole_scene(("heat", 25, 24, 20), ("heat", 24, 22.8, 15))
        live = self.live["living"]
        live["signature"] = ("off", None, *live["signature"][2:])
        live["adapter"].update(mode="off", setpoint=None)
        state = {"start": TICK-4000, "holds": {}, "commands": [],
                 "baselines": {r: list(x["signature"][:3]) for r, x in self.live.items()},
                 "parked": {"living": {"off_at": NOW-290, "mode": "heat", "started_at": NOW-4000}}}
        self.journal.write_text(json.dumps(state))
        self.observe("living")
        self.snapshot()
        self.assertEqual(self.calls, [])
        self.now += 70
        self.observe("living")
        self.observe("bedroom")
        self.snapshot()
        self.assertEqual(self.calls[-1][1]["hvac_mode"], "heat")

    def update_state(self, **fields):
        state = json.loads(self.journal.read_text()) if self.journal.exists() else {
            "start": TICK-4000, "commands": [], "holds": {},
            "baselines": {room: list(v["signature"][:3]) for room, v in self.live.items()}}
        state.update(fields)
        self.journal.write_text(json.dumps(state))

    def park(self, *rooms, ago=600, previous="heat"):
        parked = {}
        for room in rooms:
            live = self.live[room]
            live["signature"] = ("off", None, *live["signature"][2:])
            live["adapter"].update(mode="off", setpoint=None, power=8, last_mode_change=self.now-ago)
            self.observe(room)
            parked[room] = {"off_at": self.now-ago, "mode": previous, "started_at": self.now-4400}
        self.update_state(parked=parked)

    def recent_adjustment(self, room, old, *, ago, value=None):
        v = self.live[room]
        mode, setting, fan = v["signature"][:3]
        self.update_state(commands=[{
            "room": room, "kind": "temperature", "value": setting if value is None else value,
            "before": [mode, old, fan], "sent_at": self.now-ago, "proxy_at": self.now-ago,
            "purpose": "temperature", "status": "proxy_echo", "followup": {}}])

    def test_sparse_bedroom_recovery_counts_reports_not_cached_minutes(self):
        self.whole_scene(("heat", 24, 22.9, 8), ("heat", 24, 24, 8))
        self.park("bedroom", "living", ago=650)
        self.observe("bedroom", minutes=30, cadence=3600)
        self.snapshot()
        self.assertEqual(len(self.calls), 0)
        self.observe("bedroom", minutes=32, cadence=900)
        sent = self.snapshot()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual((sent["commands"][-1]["room"], sent["commands"][-1]["value"]), ("bedroom", "heat"))
        self.snapshot()
        self.assertLessEqual(len(self.calls), 2)

    def test_comfortable_unsaturated_lead_does_not_block_local_start(self):
        self.whole_scene(("heat", 22, 24.0, 120), ("heat", 23, 22.9, 8))
        self.park("living")
        state = self.snapshot()
        self.assertEqual((state["commands"][-1]["room"], state["commands"][-1]["purpose"]), ("living", "energy_start"))

    def test_initial_21_heat_is_void_when_new_target_22_is_already_too_warm(self):
        self.whole_scene(("heat", 21, 23.4, 20), ("cool", 24, 23.4, 20))
        v = self.live["bedroom"]
        v["signature"] = (*v["signature"][:4], 22, *v["signature"][5:])
        self.observe("bedroom")
        v["adapter"]["last_mode_change"] = NOW-300
        self.update_state(transitions={"bedroom": {"id": NOW-300, "mode": "heat", "ready_at": NOW-60,
                                                   "members": {"bedroom": {"phase": "initial", "target": 24}}}})
        state = self.snapshot()
        self.assertEqual(state["baselines"]["bedroom"], ["heat", 21, "auto"])
        self.assertFalse(state["transitions"])
        self.assertFalse(self.calls)

    def test_direct_high_and_trend_actions_have_distinct_purposes(self):
        for mode, temp, slope in (("heat",24.5,.0), ("cool",23.5,.0), ("heat",24.3,.5), ("cool",23.7,-.5)):
            with self.subTest(mode=mode, temp=temp):
                self.calls.clear()
                self.journal.unlink(missing_ok=True)
                self.whole_scene((mode,24,temp,230),(mode,24,24,8))
                self.observe("bedroom", minutes=15, change=slope, high_seconds=180)
                state=self.snapshot()
                self.assertEqual(len(self.calls),1)
                self.assertEqual(state["commands"][-1]["purpose"], "trend_easing" if slope else "direct_high_easing")
                self.snapshot()
                self.assertEqual(len(self.calls),1)

    def test_direct_high_recheck_withdraws_fast_path_and_recovery_suppresses_it(self):
        self.whole_scene(("heat",24,24.5,240),("heat",24,24,8))
        self.observe("bedroom",high_seconds=180,change=-.4)
        self.snapshot()
        self.assertEqual(self.calls,[])
        self.observe("bedroom",high_seconds=179)
        self.snapshot()
        self.assertEqual(self.calls,[])
        self.observe("bedroom",high_seconds=180)
        def drop(call, room):
            v=self.feedback(call,room)
            if room=="bedroom" and pilot.room_feedback.call_count>3:
                v["adapter"]["power"]=20
            return v
        # Reset the invocation count for the two assessments of this cycle.
        pilot.room_feedback.reset_mock()
        pilot.room_feedback.side_effect=drop
        self.snapshot()
        self.assertEqual(self.calls,[])

    def test_conflict_allows_only_weakening_adverse_output(self):
        self.whole_scene(("cool",24,23.1,240),("cool",24,25.1,8))
        state=self.snapshot()
        self.assertEqual((state["commands"][-1]["room"],state["commands"][-1]["value"]),("bedroom",25))
        self.assertEqual(len(self.calls),1)

    def test_acceptance_reduction_exits_at_once_after_new_adverse_feedback(self):
        self.whole_scene(("heat",23,24.0,8),("heat",23,24.8,8),("heat",22,24.8,198))
        self.park("bedroom","living")
        self.recent_adjustment("study",23,ago=350)
        self.observe("living",change=.2)
        self.observe("study",change=.2)
        self.live["study"]["adapter"]["last_mode_change"]=self.now-3600
        state=self.snapshot()
        self.assertEqual((state["commands"][-1]["room"],state["commands"][-1]["purpose"]),("study","energy_off"))
        self.assertEqual(len(self.calls),1)

    def test_acceptance_parked_start_after_parallel_wait_and_mild_initial(self):
        self.whole_scene(("heat",24,23.4,8),("heat",24,24.8,8),("heat",22,24.8,8))
        self.park("bedroom","living","study",ago=290,previous="heat")
        state=self.snapshot()
        self.assertEqual(len(self.calls),0)  # The newest low report proves only 300 seconds minus the cycle offset.
        self.now+=60
        for room in self.live:
            self.observe(room)
        sent=self.snapshot()
        self.assertEqual((sent["commands"][-1]["room"],sent["commands"][-1]["value"]),("living","cool"))
        self.now+=60
        # In this replay use a less-conservative stored companion setting after mode acknowledgement.
        v=self.live["living"]
        v["signature"]=("cool",24,*v["signature"][2:])
        v["adapter"]["setpoint"]=24
        for room in self.live:
            self.observe(room)
        sent=self.snapshot()
        self.assertEqual((sent["commands"][-1]["value"],sent["commands"][-1]["purpose"]),(25,"reversal_initial"))
        self.assertEqual(len(self.calls),2)

    def test_acceptance_cooling_eases_before_full_escalation_window(self):
        self.whole_scene(("heat",23,23.4,8),("cool",24,24.1,260))
        self.park("bedroom")
        self.recent_adjustment("living",25,ago=420)
        self.observe("living",change=-.8,minutes=6,cadence=300)
        state=self.snapshot()
        self.assertEqual((state["commands"][-1]["purpose"],state["commands"][-1]["value"]),("trend_easing",25))
        self.snapshot()
        self.assertEqual(len(self.calls),1)

    def test_accepted_on_restart_waits_for_proxy_or_times_out_without_resending(self):
        self.scenario("heat",20,23,20)
        sent=self.snapshot()
        self.assertEqual(sent["commands"][-1]["status"],"accepted")
        self.live["bedroom"]["adapter"]["updated_at"]=self.now-60
        self.now+=60
        self.snapshot()
        self.now+=130
        state=self.snapshot()
        self.assertEqual(state["commands"][-1]["status"],"unconfirmed")
        self.assertIn("bedroom",state["holds"])
        self.assertEqual(len(self.calls),1)

    def start_for_initial(self, mode, inherited):
        measured = 23.2 if mode == "heat" else 24.8
        self.journal.unlink(missing_ok=True)
        self.calls.clear()
        self.now = NOW
        self.whole_scene((mode, 24, measured, 8), (mode, 24, 24, 8))
        self.park("bedroom", "living", ago=720, previous=mode)
        state = self.snapshot()
        self.assertEqual(state["commands"][-1]["purpose"], "energy_start")
        current = self.live["bedroom"]
        current["signature"] = (mode, inherited, *current["signature"][2:])
        current["adapter"]["setpoint"] = inherited
        return current

    def refresh_for_initial(self, current, target=24, temp=None, minutes=4):
        self.now = NOW + minutes*60
        current["signature"] = (*current["signature"][:4], target, *current["signature"][5:])
        if temp is not None:
            current["temperature"] = temp
        self.observe("bedroom")
        self.observe("living")

    def test_excessive_heat_and_cool_in_hold_are_repaired_after_real_start(self):
        for mode, inherited in (("heat", 27), ("cool", 20)):
            for minutes in (4, 12, 22):
                with self.subTest(mode=mode, minutes=minutes):
                    current = self.start_for_initial(mode, inherited)
                    self.refresh_for_initial(current, temp=24, minutes=minutes)
                    state = self.snapshot()
                    self.assertEqual(len(self.calls), 2)
                    mode_action, repair = state["commands"][-2:]
                    self.assertEqual((repair["purpose"], repair["value"]), ("initial_safety_easing", 24))
                    self.assertEqual(repair["transaction"], mode_action["transaction"])
                    self.assertEqual(state["baselines"]["bedroom"], [mode, inherited, "auto"])
                    self.assertEqual(mode_action["status"], "proxy_echo")
                    state = self.snapshot()
                    self.assertEqual(state["baselines"]["bedroom"], [mode, 24, "auto"])
                    self.assertFalse(state["transitions"])
                    self.assertEqual(len(self.calls), 2)

    def test_initial_rebuilds_from_new_target_or_closes_legal_setting_in_hold(self):
        current = self.start_for_initial("heat", 27)
        self.refresh_for_initial(current, target=22, temp=22.1)
        state = self.snapshot()
        self.assertEqual(state["commands"][-1]["value"], 22)
        self.assertEqual(len(self.calls), 2)
        for inherited in (20, 23):
            current = self.start_for_initial("heat", inherited)
            self.refresh_for_initial(current, target=22, temp=22)
            state = self.snapshot()
            self.assertFalse(state["transitions"])
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(state["baselines"]["bedroom"], ["heat", inherited, "auto"])

    def test_initial_safety_honors_stale_data_pause_and_unresolved_echo(self):
        for block in ("reading", "power", "permission", "failure", "manual", "accepted", "prepared"):
            with self.subTest(block=block):
                current = self.start_for_initial("cool", 20)
                self.refresh_for_initial(current, temp=24)
                if block=="reading":
                    current["temperature_age"] = 1201
                elif block=="power":
                    current["adapter"]["from_power_s"] = 121
                elif block=="permission":
                    permit = json.loads(self.permit.read_text())
                    permit["enabled"] = False
                    self.permit.write_text(json.dumps(permit))
                elif block=="failure":
                    self.update_state(holds={"bedroom":"test"})
                elif block=="manual":
                    self.update_state(manual_holds={"bedroom":"test"})
                else:
                    self.now = NOW+60
                    current["adapter"]["updated_at"] = NOW-5
                    if block=="prepared":
                        state = json.loads(self.journal.read_text())
                        state["commands"][-1]["status"] = "prepared"
                        self.update_state(commands=state["commands"])
                state = self.snapshot()
                self.assertEqual(len(self.calls), 1)
                self.assertIn("bedroom", state["transitions"])
                if block == "accepted":
                    self.assertEqual(state["commands"][-1]["status"], "accepted")
                if block=="prepared":
                    self.assertEqual(state["commands"][-1]["status"], "unconfirmed")
                    self.assertIn("bedroom", state["holds"])

    def test_protective_stop_allows_new_power_value_but_requires_fresh_power(self):
        self.whole_scene(("cool",24,24,240), ("heat",20,24.8,198))
        self.recent_adjustment("living",21,ago=660)
        v=self.live["living"]
        v["adapter"]["power_state_updated_at"] = NOW
        state = self.snapshot()
        self.assertEqual(state["commands"][-1]["purpose"], "energy_off")
        self.assertEqual(len(self.calls),1)
        self.journal.unlink()
        self.calls.clear()
        self.whole_scene(("cool",24,24,240), ("heat",20,24.8,198))
        self.recent_adjustment("living",21,ago=660)
        self.live["living"]["adapter"]["from_power_s"] = 121
        self.snapshot()
        self.assertEqual(self.calls,[])

    def test_mild_recovery_memory_cannot_reuse_extreme_shedding_setting(self):
        for mode, remembered, expected in (("heat", 20, 22), ("cool", 26, 26)):
            with self.subTest(mode=mode):
                current = self.start_for_initial(mode, remembered)
                self.refresh_for_initial(current, temp=23.2 if mode=="heat" else 24.8)
                state = self.snapshot()
                if mode=="heat":
                    self.assertEqual((state["commands"][-1]["purpose"], state["commands"][-1]["value"]),
                                     ("reversal_initial", expected))
                else:
                    self.assertEqual(len(self.calls), 1)

    def start_parked_living(self, *, failures=None):
        self.whole_scene(("heat", 25, 24, 20), ("heat", 24, 22.8, 15))
        live = self.live["living"]
        live["signature"] = ("off", None, *live["signature"][2:])
        live["adapter"].update(mode="off", setpoint=None, last_mode_change=self.now-900)
        state = {"start": TICK-4000, "holds": {}, "commands": [],
                 "baselines": {r: list(x["signature"][:3]) for r, x in self.live.items()},
                 "parked": {"living": {"off_at": self.now-900, "mode": "heat", "started_at": self.now-4000}}}
        if failures:
            state["start_failures"] = failures
        self.journal.write_text(json.dumps(state))
        self.observe("living")
        started = self.snapshot()
        self.assertEqual((started["commands"][-1]["purpose"], started["commands"][-1]["value"]),
                         ("energy_start", "heat"))
        for seconds in (20, 20, 60):
            self.now += seconds
            self.observe("living")
            echoed = self.snapshot()
        self.assertEqual((echoed["commands"][0]["purpose"], echoed["commands"][0]["status"]),
                         ("energy_start", "proxy_echo"))
        self.assertTrue(all(c["status"] == "proxy_echo" for c in echoed["commands"]))
        return echoed

    def revert_living_to_off(self, after=300):
        self.now += after
        live = self.live["living"]
        live["signature"] = ("off", None, *live["signature"][2:])
        live["adapter"].update(mode="off", setpoint=None, last_mode_change=self.now-10, updated_at=self.now-10)
        self.observe("living")
        self.observe("bedroom")
        return self.snapshot()

    def test_unpowered_start_that_reverts_is_retried_not_paused(self):
        self.start_parked_living()
        calls = len(self.calls)
        state = self.revert_living_to_off()
        self.assertEqual(state["manual_holds"], {})
        self.assertEqual(state["baselines"]["living"], ["off", None, "auto"])
        failure = state["start_failures"]["living"]
        self.assertEqual((failure["count"], failure["mode"], failure["retry_at"]), (1, "heat", self.now+900))
        self.assertEqual(state["parked"]["living"]["mode"], "heat")
        self.assertNotIn("living_study", state["transitions"])
        self.assertEqual(len(self.calls), calls)
        self.now += 600
        self.observe("living")
        self.observe("bedroom")
        self.snapshot()
        self.assertEqual(len(self.calls), calls)
        self.now += 320
        self.observe("living")
        self.observe("bedroom")
        retried = self.snapshot()
        self.assertEqual(len(self.calls), calls+1)
        self.assertEqual((retried["commands"][-1]["purpose"], retried["commands"][-1]["value"]),
                         ("energy_start", "heat"))

    def test_repeated_unpowered_starts_hold_retries_for_a_bounded_time(self):
        self.start_parked_living(failures={"living": {"count": 2, "at": NOW-2000, "mode": "heat",
                                                      "retry_at": NOW-100}})
        calls = len(self.calls)
        state = self.revert_living_to_off()
        failure = state["start_failures"]["living"]
        self.assertEqual((failure["count"], failure["hold_until"]), (3, self.now+7200))
        self.assertNotIn("retry_at", failure)
        self.assertEqual(state["manual_holds"], {})
        self.now += 3600
        self.observe("living")
        self.observe("bedroom")
        self.snapshot()
        self.assertEqual(len(self.calls), calls)

    def test_start_that_draws_power_clears_previous_failures(self):
        self.start_parked_living(failures={"living": {"count": 1, "at": NOW-2000, "mode": "heat",
                                                      "retry_at": NOW-100}})
        self.now += 60
        self.live["living"]["adapter"]["power"] = 600
        self.observe("living")
        state = self.snapshot()
        self.assertNotIn("living", state.get("start_failures", {}))
        self.assertEqual(state["commands"][0]["peak_watts"], 600)
        self.live["living"]["adapter"]["power"] = 15
        state = self.revert_living_to_off()
        self.assertNotIn("living", state.get("start_failures", {}))
        self.assertIn("living", state["external_overrides"])

    def test_external_change_yields_temporarily_then_resumes(self):
        self.scenario("heat", 20, 23, 20, manual=True)
        pilot.cycle(self.request)
        self.assertEqual(self.calls, [])
        state = json.loads(self.journal.read_text())
        self.assertEqual(state["manual_holds"], {})
        override = state["external_overrides"]["bedroom"]
        self.assertEqual((override["previous"], override["reported"], override["until"]),
                         (["heat", 21, "auto"], ["heat", 20, "auto"], NOW+3600))
        self.assertEqual(state["baselines"]["bedroom"], ["heat", 20, "auto"])
        self.now = NOW+3590
        self.observe("bedroom")
        pilot.cycle(self.request)
        self.assertEqual(self.calls, [])
        self.now = NOW+3610
        self.observe("bedroom")
        pilot.cycle(self.request)
        self.assertEqual(self.calls[-1][1]["temperature"], 21)
        self.assertNotIn("bedroom", json.loads(self.journal.read_text())["external_overrides"])

    def test_external_change_during_override_extends_it(self):
        self.scenario("heat", 20, 23, 20, manual=True)
        pilot.cycle(self.request)
        self.now = NOW+1800
        live = self.live["bedroom"]
        live["signature"] = ("heat", 19, *live["signature"][2:])
        live["adapter"]["setpoint"] = 19
        self.observe("bedroom")
        pilot.cycle(self.request)
        override = json.loads(self.journal.read_text())["external_overrides"]["bedroom"]
        self.assertEqual((override["reported"], override["until"]), (["heat", 19, "auto"], NOW+1800+3600))
        self.assertEqual(self.calls, [])

    def test_external_off_parks_unit_with_previous_mode(self):
        self.whole_scene(("heat", 25, 24, 20), ("heat", 24, 22.8, 240))
        self.snapshot()
        calls = len(self.calls)
        live = self.live["living"]
        live["signature"] = ("off", None, *live["signature"][2:])
        live["adapter"].update(mode="off", setpoint=None, power=8, last_mode_change=self.now)
        self.now += 20
        self.observe("living")
        state = self.snapshot()
        self.assertEqual(state["parked"]["living"], {"off_at": self.now-20, "mode": "heat", "started_at": self.now-20})
        self.assertIn("living", state["external_overrides"])
        self.assertEqual(len(self.calls), calls)

    def test_legacy_automatic_hold_is_released_but_operator_hold_remains(self):
        self.whole_scene(("heat", 25, 24, 20), ("heat", 24, 22.8, 15), ("heat", 24, 22.8, 15))
        for room in ("living", "study"):
            live = self.live[room]
            live["signature"] = ("off", None, *live["signature"][2:])
            live["adapter"].update(mode="off", setpoint=None, last_mode_change=self.now-3600)
        self.update_state(baselines={"bedroom": list(self.live["bedroom"]["signature"][:3]),
                                     "living": ["heat", 24, "auto"], "study": ["heat", 24, "auto"]},
                          manual_holds={"living": pilot.LEGACY_EXTERNAL_HOLD, "study": "operator"})
        state = self.snapshot()
        self.assertEqual(state["manual_holds"], {"study": "operator"})
        self.assertEqual(state["baselines"]["living"], ["off", None, "auto"])
        self.assertEqual(state["parked"]["living"], {"off_at": self.now-3600, "mode": "heat",
                                                     "started_at": self.now-3600})
        self.assertNotIn("living", state.get("external_overrides", {}))
        self.assertIn("study", state["external_overrides"])

    def test_attribute_drift_after_failed_start_is_not_counted_again(self):
        self.start_parked_living()
        self.revert_living_to_off()
        self.now += 120
        live = self.live["living"]
        live["signature"] = ("off", 24, *live["signature"][2:])
        self.observe("living")
        state = self.snapshot()
        failure = state["start_failures"]["living"]
        self.assertEqual(failure["count"], 1)
        self.assertNotIn("hold_until", failure)
        self.assertNotIn("living", state.get("external_overrides", {}))
        self.assertEqual(state["parked"]["living"]["mode"], "heat")
        self.assertEqual(state["baselines"]["living"], ["off", 24, "auto"])

    def test_stopped_unit_drift_keeps_parked_evidence_without_override(self):
        self.whole_scene(("heat", 25, 24, 20), ("heat", 24, 24, 15))
        self.park("living", ago=600)
        parked = {"off_at": self.now-600, "mode": "cool", "started_at": self.now-1200, "handoff": True}
        self.update_state(parked={"living": parked},
                          baselines={r: list(x["signature"][:3]) for r, x in self.live.items()})
        live = self.live["living"]
        live["signature"] = ("off", 24, "low", *live["signature"][3:])
        self.observe("living")
        state = self.snapshot()
        self.assertEqual(state["parked"]["living"], parked)
        self.assertNotIn("living", state.get("external_overrides", {}))
        self.assertEqual(state["baselines"]["living"], ["off", 24, "low"])

    def test_legacy_hold_is_released_only_after_an_available_report(self):
        self.whole_scene(("heat", 25, 24, 20), ("heat", 24, 22.8, 15))
        live = self.live["living"]
        live["signature"] = ("unavailable", None, None, *live["signature"][3:])
        live["adapter"].update(mode="unavailable", setpoint=None)
        self.update_state(baselines={"bedroom": list(self.live["bedroom"]["signature"][:3]),
                                     "living": ["heat", 24, "auto"], "study": ["heat", 24, "auto"]},
                          manual_holds={"living": pilot.LEGACY_EXTERNAL_HOLD})
        state = self.snapshot()
        self.assertEqual(state["manual_holds"], {"living": pilot.LEGACY_EXTERNAL_HOLD})
        self.assertEqual(state["baselines"]["living"], ["heat", 24, "auto"])
        live["signature"] = ("off", None, "auto", *live["signature"][3:])
        live["adapter"].update(mode="off", last_mode_change=self.now-3600)
        self.now += 20
        self.observe("living")
        state = self.snapshot()
        self.assertEqual(state["manual_holds"], {})
        self.assertEqual(state["parked"]["living"]["mode"], "heat")
        self.assertNotIn("living", state.get("external_overrides", {}))

    def test_manual_start_clears_start_failures(self):
        self.whole_scene(("heat", 25, 24, 20), ("heat", 24, 22.8, 15))
        self.park("living", ago=900)
        self.update_state(baselines={r: list(x["signature"][:3]) for r, x in self.live.items()},
                          start_failures={"living": {"count": 3, "at": NOW-100, "mode": "heat", "hold_until": NOW+7000}})
        live = self.live["living"]
        live["signature"] = ("heat", 24, *live["signature"][2:])
        live["adapter"].update(mode="heat", setpoint=24, power=600, last_mode_change=self.now)
        self.now += 20
        self.observe("living")
        state = self.snapshot()
        self.assertNotIn("living", state["start_failures"])
        self.assertIn("living", state["external_overrides"])
        self.assertNotIn("living", state["parked"])

    def test_revert_without_power_after_the_start_is_an_external_change(self):
        self.whole_scene(("heat", 25, 24, 20), ("heat", 24, 22.8, 15))
        live = self.live["living"]
        live["signature"] = ("off", None, *live["signature"][2:])
        live["adapter"].update(mode="off", setpoint=None, last_mode_change=self.now-900)
        self.update_state(baselines={r: list(x["signature"][:3]) for r, x in self.live.items()},
                          parked={"living": {"off_at": self.now-900, "mode": "heat", "started_at": self.now-4000}})
        self.observe("living")
        started = self.snapshot()
        self.assertEqual(started["commands"][-1]["purpose"], "energy_start")
        for seconds in (20, 20):
            self.now += seconds
            self.observe("living")
            echoed = self.snapshot()
        self.assertIsNone(echoed["commands"][0].get("peak_watts"))
        state = self.revert_living_to_off()
        self.assertNotIn("living", state.get("start_failures", {}))
        self.assertIn("living", state["external_overrides"])


if __name__ == "__main__":
    unittest.main()
