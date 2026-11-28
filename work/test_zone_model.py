"""Online zone model and learned influence against simulated rooms and the real controller."""
from copy import deepcopy
import json
from math import pi, sin
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import test_p0_p1_dispatch as fixture
import house_policy
from service import metered_interval
from topology import configure, fingerprint
from zone_control import observations
import zone_model


pilot = fixture.pilot
EXAMPLE = Path(pilot.__file__).with_name("topology.example.json")
POLICY = dict(house_policy.POLICY)
READY_STATS = {"n": 60, "unscored": 0, "model": 0.05, "persistence": 0.1, "rule": 0.1, "coverage": 0.7}


def mark_ready(zone):
    zone.update(ready=True, stats=dict(READY_STATS))


def site(**influences):
    """Two zones and two units; `influences` overrides the configured cross influence."""
    actuators = {"ua": {"zone": "a", "influences": {"a": 1, "b": influences.get("ua_b", 0.35)}, "lock_group": "ua"},
                 "ub": {"zone": "b", "influences": {"b": 1, "a": influences.get("ub_a", 0.35)}, "lock_group": "ub"}}
    sensors = {"pa": {"role": "primary"}, "pb": {"role": "primary"},
               "outdoor": {"role": "environment", "entity": "sensor.outdoor"}}
    zones = {"a": {"primary": "pa"}, "b": {"primary": "pb"}}
    return {"sensors": sensors, "zones": zones, "actuators": actuators,
            "sources": {key: "sensor." + key for key in sensors},
            "zone_versions": {key: fingerprint([key, zone]) for key, zone in zones.items()}}


class House:
    """Minute-step truth: dT/dt = a*(T_out-T) + c + sum of unit gains while cooling."""

    gains = {("ua", "a"): -10.0, ("ua", "b"): -5.0, ("ub", "b"): -8.0, ("ub", "a"): 0.0}

    def __init__(self, seed=7, cadences=(("a", 300), ("b", 600))):
        self.rng = random.Random(seed)
        self.cadences = cadences
        self.t, self.temps = 0.0, {"a": 25.0, "b": 25.0}
        self.reports, self.power = {"a": None, "b": None}, {"ua": [], "ub": []}
        self.modes, self.changed = {"ua": "off", "ub": "off"}, {"ua": -1e6, "ub": -1e6}

    def outdoor(self):
        return 31+3*sin(2*pi*self.t/86400)

    def schedule(self, unit):
        minute = int(self.t//60)
        both = minute % 360 < 15
        alone = minute % 120 < 20 if unit == "ua" else 60 <= minute % 120 < 80
        return "cool" if alone or both else "off"

    def step(self):
        for unit in self.modes:
            mode = self.schedule(unit)
            if mode != self.modes[unit]:
                self.modes[unit], self.changed[unit] = mode, self.t
        kw = {unit: (0.6+self.rng.uniform(-0.05, 0.05)) if mode == "cool" else 0.002 for unit, mode in self.modes.items()}
        for unit in kw:
            # A meter report holds until the next one.
            self.power[unit].append((self.t, kw[unit]*1000))
        for zone, drift in (("a", 0.4), ("b", 0.6)):
            rate = 0.08*(self.outdoor()-self.temps[zone])+drift
            rate += sum(self.gains[(unit, zone)]*kw[unit] for unit in kw if self.modes[unit] == "cool")
            self.temps[zone] += rate/60
        self.t += 60
        for zone, cadence in self.cadences:
            if int(self.t) % cadence == 0:
                self.reports[zone] = {"value": round(self.temps[zone]+self.rng.gauss(0, 0.03), 1),
                                      "at": self.t, "quality": "fresh", "source": ["sensor.p" + zone, "x"]}

    def units_now(self):
        return {unit: {"mode": self.modes[unit], "changed_at": self.changed[unit],
                       "kw": self.power[unit][-1][1]/1000 if self.power[unit] else None} for unit in self.modes}

    def interval(self, unit, start, end):
        # Like the controller, only the reports received so far are known.
        kwh, covered = metered_interval([p for p in self.power[unit] if p[0] <= self.t], start, end)
        return kwh*3600/covered if covered >= 0.8*(end-start) else None


def run(house, topology, local, policy, minutes, weather=True):
    outputs = {}
    for _ in range(minutes):
        house.step()
        context = zone_model.begin(local, topology, policy)
        environment = {"outdoor": {"value": round(house.outdoor(), 1), "at": house.t, "quality": "fresh"}} if weather else {}
        for zone in ("a", "b"):
            if house.reports[zone] is not None:
                outputs[zone] = zone_model.observe_zone(context, zone, house.reports[zone], None, environment,
                                                        house.units_now(), house.interval, house.t)
    return outputs


class ZoneModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.house, cls.topology, cls.local = House(), site(), {}
        cls.outputs = run(cls.house, cls.topology, cls.local, POLICY, 2*1440)

    def test_identifies_gains_and_beats_persistence_on_undisturbed_windows(self):
        zones = self.local["model"]["zones"]
        for key, home_gain in (("a", -10.0), ("b", -8.0)):
            zone = zones[key]
            unit = "ua" if key == "a" else "ub"
            learned = zone["theta"][zone["features"].index(f"{unit}:cool")]
            self.assertAlmostEqual(learned, home_gain, delta=1.5, msg=key)
            stats, running = zone["stats"], zone["running"]
            self.assertGreaterEqual(stats["n"], POLICY["model_min_samples"])
            self.assertGreater(stats["unscored"], 0)
            self.assertLess(stats["model"], 0.7*stats["persistence"])
            self.assertTrue(0.5 <= stats["coverage"] <= 0.95, stats["coverage"])
            self.assertGreaterEqual(running["n"], POLICY["model_min_running_samples"])
            self.assertLess(running["model"], 0.5*min(running["persistence"], running["rule"]))
            self.assertTrue(zone["ready"] and zone["ready_running"], key)
            self.assertFalse(self.outputs[key]["use"])  # Shadow by default.

    def test_learned_influence_separates_zero_from_real_cross_effect(self):
        learned = zone_model.influence(self.local["model"], self.topology, POLICY)
        self.assertEqual(learned[("ub", "cool", "a")]["status"], "negligible")
        cross = learned[("ua", "cool", "b")]
        self.assertAlmostEqual(cross["ratio"], 0.5, delta=0.15)
        self.assertEqual(cross["status"], "significant")
        self.assertNotIn(("ua", "heat", "b"), learned)  # Heating was never excited.
        self.assertEqual(zone_model.trusted(learned, self.local["model"], self.topology), learned)

    def test_risk_influences_follow_confident_learning_only(self):
        learned = {("ub", "cool", "a"): {"status": "negligible", "ratio": 0.01, "std": 0.02},
                   ("ub", "cool", "c"): {"status": "significant", "ratio": 0.6, "std": 0.05},
                   ("ub", "heat", "a"): {"status": "uncertain", "ratio": 0.0, "std": 0.5},
                   ("ub", "cool", "b"): {"status": "negligible", "ratio": 0.0, "std": 0.0}}
        self.assertEqual(zone_model.risk_influences("ub", "cool", self.topology, learned), {"b": 1, "c": 0.6})
        self.assertEqual(zone_model.risk_influences("ub", "heat", self.topology, learned), {"b": 1, "a": 0.35})
        self.assertEqual(zone_model.risk_influences("ub", "cool", self.topology, {}), {"b": 1, "a": 0.35})

    def test_active_policy_uses_a_ready_model_and_state_survives_json(self):
        local = json.loads(json.dumps(self.local))
        house = deepcopy(self.house)
        outputs = run(house, self.topology, local, dict(POLICY, model_active=1), 30)
        self.assertEqual((outputs["a"]["hold"], outputs["a"]["use"]), (None, True))
        self.assertEqual(outputs["a"]["forecast"]["source"], "model")
        zone = local["model"]["zones"]["a"]
        size = len(zone["theta"])
        for i in range(size):
            self.assertGreater(zone["covariance"][i][i], 0)
            for j in range(size):
                self.assertAlmostEqual(zone["covariance"][i][j], zone["covariance"][j][i], places=9)
        self.assertLess(len(json.dumps(local)), 20000)

    def test_output_from_an_unidentified_unit_holds_the_forecast(self):
        local = json.loads(json.dumps(self.local))
        context = zone_model.begin(local, self.topology, dict(POLICY, model_active=1))
        environment = {"outdoor": {"value": 31, "at": self.house.t, "quality": "fresh"}}
        report = local["model"]["zones"]["a"]["last"]
        cooling = {"ua": {"mode": "cool", "changed_at": 0, "kw": 0.6}, "ub": {"mode": "off", "changed_at": 0, "kw": 0.0}}
        heating = {"ua": {"mode": "off", "changed_at": 0, "kw": 0.0}, "ub": {"mode": "heat", "changed_at": 0, "kw": 0.6}}
        for units, hold in ((cooling, None), (heating, "unidentified_unit")):
            outcome = zone_model.observe_zone(context, "a", dict(report, quality="fresh"), None, environment, units,
                                              lambda *_: None, self.house.t)
            self.assertEqual((outcome["hold"], outcome["use"]), (hold, hold is None))


class FrequentReportTests(unittest.TestCase):
    def test_minute_reports_learn_gains_far_from_the_prior(self):
        policy = dict(POLICY, model_prior_gain=3)
        house, topology, local = House(seed=11, cadences=(("a", 60), ("b", 60))), site(), {}
        run(house, topology, local, policy, 1440)
        for key, unit, truth in (("a", "ua", -10.0), ("b", "ub", -8.0)):
            zone = local["model"]["zones"][key]
            self.assertGreater(zone["updates"], 150, key)
            self.assertGreaterEqual(zone["excited"][unit]["cool"], POLICY["influence_min_excitation"], key)
            self.assertAlmostEqual(zone["theta"][zone["features"].index(f"{unit}:cool")], truth, delta=2.0, msg=key)
            self.assertGreaterEqual(zone["stats"]["n"], POLICY["model_min_samples"], key)
            self.assertLessEqual(len(zone["pending"]),
                                 zone_model.SETTLE_SECONDS[1]//zone_model.PENDING_SPACING_SECONDS+1)
            self.assertLess(zone["stats"]["model"], zone["stats"]["persistence"])


class ZoneModelRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.topology, self.local, self.house = site(), {}, House(seed=3)
        run(self.house, self.topology, self.local, POLICY, 600)

    def context(self, policy=POLICY):
        return zone_model.begin(self.local, self.topology, policy)

    def test_glitch_report_is_downweighted(self):
        zone = self.local["model"]["zones"]["a"]
        before, noise = list(zone["theta"]), zone["noise"]
        last = zone["last"]
        glitch = {"value": last["value"]+3.0, "at": last["at"]+300, "quality": "fresh"}
        environment = {"outdoor": {"value": 31, "at": glitch["at"], "quality": "fresh"}}
        idle = {unit: {"mode": "off", "changed_at": 0, "kw": 0.0} for unit in ("ua", "ub")}
        zone_model.observe_zone(self.context(), "a", glitch, None, environment, idle, lambda *_: 0.002, glitch["at"])
        self.assertLess(max(abs(a-b) for a, b in zip(zone["theta"], before)), 1.5)
        self.assertLess(zone["noise"], max(noise, zone_model.NOISE_FLOOR_C**2)*3)

    def test_uncovered_output_accumulates_and_a_reversal_skips_learning(self):
        zone = self.local["model"]["zones"]["a"]
        environment = {"outdoor": {"value": 31, "at": self.house.t+300, "quality": "fresh"}}
        running = {"ua": {"mode": "cool", "changed_at": 0, "kw": 0.6}, "ub": {"mode": "off", "changed_at": 0, "kw": 0.0}}
        updates, anchor = zone["updates"], zone["anchor"]["at"]
        for step in (300, 600):
            report = {"value": zone["last"]["value"], "at": anchor+step, "quality": "fresh"}
            zone_model.observe_zone(self.context(), "a", report, None, environment, running, lambda *_: None, report["at"])
            self.assertEqual((zone["updates"], zone["reason"], zone["anchor"]["at"]), (updates, "accumulating", anchor))
        report = dict(report, at=anchor+POLICY["model_max_interval_seconds"]+60)
        zone_model.observe_zone(self.context(), "a", report, None, environment, running, lambda *_: None, report["at"])
        self.assertEqual((zone["reason"], zone["anchor"]["at"]), ("interval", report["at"]))
        zone["anchor"]["modes"]["ua"] = "heat"
        running["ua"]["changed_at"] = report["at"]+10
        report = dict(report, at=report["at"]+300)
        zone_model.observe_zone(self.context(), "a", report, None, environment, running, lambda *_: 0.5, report["at"])
        self.assertEqual((zone["updates"], zone["reason"]), (updates, "mode_reversal"))

    def test_missing_environment_readings_do_not_stop_learning_or_forecasts(self):
        updates = self.local["model"]["zones"]["a"]["updates"]
        outputs = run(self.house, self.topology, self.local, POLICY, 120, weather=False)
        self.assertGreater(self.local["model"]["zones"]["a"]["updates"], updates)
        self.assertIsNotNone(outputs["a"]["forecast"])

    def test_intervention_leaves_forecasts_unscored(self):
        zone = self.local["model"]["zones"]["a"]
        environment = {"outdoor": {"value": 31, "at": self.house.t, "quality": "fresh"}}
        idle = {unit: {"mode": "off", "changed_at": 0, "kw": 0.002} for unit in ("ua", "ub")}
        start = zone["last"]["at"]
        zone["pending"] = []
        settled, unscored = zone["stats"]["n"], zone["stats"]["unscored"]
        # The 300 s forecast settles at 900 s; the 600 s forecast meets an action at 650 s.
        for step, intervened in ((300, 0), (600, 0), (900, 0), (1200, start+650)):
            report = {"value": zone["last"]["value"], "at": start+step, "quality": "fresh"}
            zone_model.observe_zone(self.context(), "a", report, None, environment, idle, lambda *_: 0.002,
                                    report["at"], intervened)
        self.assertEqual(zone["stats"]["n"], settled+1)
        self.assertEqual(zone["stats"]["unscored"], unscored+1)

    def test_new_environment_sensor_migrates_coefficients_but_a_new_definition_resets(self):
        zone = self.local["model"]["zones"]["a"]
        gain = zone["theta"][zone["features"].index("ua:cool")]
        topology = deepcopy(self.topology)
        topology["sensors"]["balcony"] = {"role": "environment", "entity": "sensor.balcony"}
        topology["sensors"]["fixed"] = {"role": "environment", "from_actuator": "ua"}
        topology["sources"].update(balcony="sensor.balcony", fixed="sensor.fixed")
        migrated = zone_model.zone_state(zone_model.begin(self.local, topology, POLICY), "a")
        self.assertEqual(migrated["features"][:3], ["env:balcony", "env:outdoor", "mass"])
        self.assertEqual(migrated["theta"][migrated["features"].index("ua:cool")], gain)
        self.assertEqual((migrated["stats"]["n"], migrated["ready"], migrated["ready_running"]), (0, False, False))
        topology["zone_versions"]["a"] = "changed"
        reset = zone_model.zone_state(zone_model.begin(self.local, topology, POLICY), "a")
        self.assertEqual(reset["updates"], 0)
        self.assertEqual(reset["theta"][reset["features"].index("ua:cool")], -POLICY["model_prior_gain"])


class ControllerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.h = fixture.DispatchTests()
        self.addCleanup(self.h.doCleanups)
        self.h.setUp()
        self.topology = configure(json.loads(EXAMPLE.read_bytes()), pilot.ROOMS)
        self.enterContext(patch.object(pilot, "TOPOLOGY", self.topology))
        self.audit = self.enterContext(patch.object(pilot, "audit"))

    def trace(self):
        return next(x.kwargs["assessment"] for x in reversed(self.audit.call_args_list)
                    if x.args == ("cycle",) and "assessment" in x.kwargs)

    def learned_state(self, ready):
        """Bedroom cooling clearly serves its own zone and leaves the lounge unchanged."""
        local = {}
        context = zone_model.begin(local, self.topology, POLICY)
        for key in self.topology["zones"]:
            zone_model.zone_state(context, key)
        for key, gain, variance in (("sleep", -10.0, 0.25), ("lounge", -0.2, 0.04)):
            zone = local["model"]["zones"][key]
            index = zone["features"].index("bedroom:cool")
            zone["theta"][index] = gain
            zone["covariance"][index][index] = variance
            zone["excited"] = {"bedroom": {"cool": 12}}
            if ready:
                mark_ready(zone)
        return local["model"]

    def scene(self, *rooms):
        self.h.whole_scene(*rooms)
        permit = json.loads(self.h.permit.read_bytes())
        for room, state in permit["rooms"].items():
            if state[0] == "off":
                permit["rooms"][room] = ["cool", 25, "auto"]
        self.h.permit.write_text(json.dumps(permit))
        self.h.park("bedroom", "living", "study", ago=800)

    def assisted_start(self, active, ready=True):
        self.scene(("off", 25, 23.4, 8), ("off", 24, 24.6, 8), ("off", 24, 24.6, 8))
        bedroom = self.h.live["bedroom"]
        bedroom["signature"] = (*bedroom["signature"][:4], 23, *bedroom["signature"][5:])
        self.h.update_state(manual_holds={"living": "fixture", "study": "fixture"},
                            closed_loop={"model": self.learned_state(ready)})
        with patch.dict(house_policy.POLICY, {"model_active": active}):
            for seconds in (0, 600, 600, 600):
                self.h.now += seconds
                for room in self.h.live:
                    self.h.observe(room)
                state = self.h.snapshot()
        return state

    def test_shadow_model_records_but_keeps_cross_zone_assistance(self):
        started = self.assisted_start(0)
        trace = self.trace()
        self.assertEqual([(a["purpose"], a["control"]["zone"]) for a in started["commands"]], [("energy_start", "lounge")])
        self.assertFalse(trace["zone_model"]["active"])
        lounge = next(row for row in trace["zone_model"]["influence"] if row["zone"] == "lounge")
        self.assertEqual((lounge["unit"], lounge["mode"], lounge["ratio"], lounge["status"], lounge["trusted"]),
                         ("bedroom", "cool", 0.02, "negligible", True))

    def test_active_model_stops_assistance_that_cannot_reach_the_zone(self):
        started = self.assisted_start(1)
        trace = self.trace()
        self.assertTrue(trace["zone_model"]["active"])
        self.assertNotIn("bedroom", trace["feasible"].get("lounge", []))
        self.assertEqual([(a["purpose"], a["control"]["zone"]) for a in started["commands"]], [("energy_start", "sleep")])

    def test_active_model_ignores_influence_until_both_zones_are_ready(self):
        started = self.assisted_start(1, ready=False)
        trace = self.trace()
        self.assertIn("bedroom", trace["feasible"]["lounge"])
        self.assertEqual([(a["purpose"], a["control"]["zone"]) for a in started["commands"]], [("energy_start", "lounge")])
        self.assertFalse(next(row for row in trace["zone_model"]["influence"] if row["zone"] == "lounge")["trusted"])

    def test_example_topology_learns_without_actuator_environment_points(self):
        self.scene(("off", 25, 24.0, 8), ("off", 25, 24.1, 8), ("off", 25, 24.1, 8))
        for _ in range(12):
            self.h.now += 600
            for room in self.h.live:
                self.h.observe(room)
            state = self.h.snapshot()
        zones = state["closed_loop"]["model"]["zones"]
        self.assertEqual(zones["sleep"]["features"][0], "mass")
        for key in ("sleep", "lounge"):
            self.assertGreater(zones[key]["updates"], 5, key)
            self.assertGreater(zones[key]["stats"]["n"], 0, key)
        self.assertEqual(self.h.calls, [])


class ObservationTests(unittest.TestCase):
    def setUp(self):
        harness = fixture.DispatchTests()
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        harness.whole_scene(("off", 25, 24.0, 8), ("off", 25, 24.2, 8), ("off", 25, 24.2, 8))
        self.harness, self.topology = harness, configure(json.loads(EXAMPLE.read_bytes()), pilot.ROOMS)

    def observe(self, memory, policy=POLICY, seconds=600, change=0.0):
        live = self.harness.live
        self.harness.now += seconds
        for room in live:
            live[room]["temperature"] += change
            live[room]["temperature_report_at"] = self.harness.now
        return observations(self.topology, deepcopy(live), {}, memory, [], set(), self.harness.now, policy)[0]

    def test_active_model_replaces_only_the_response_trend(self):
        memory = {}
        for _ in range(4):
            self.observe(memory, change=-0.1)
        for zone in memory["model"]["zones"].values():
            mark_ready(zone)
        zones = self.observe(memory, dict(POLICY, model_active=1), change=-0.1)
        sleep = zones["sleep"]
        self.assertIsNone(sleep["model_hold"])
        self.assertEqual(sleep["response_trend"]["source"], "model")
        self.assertNotEqual((sleep["trend"] or {}).get("source"), "model")

    def test_damaged_model_state_falls_back_to_rules(self):
        memory = {}
        self.observe(memory)
        memory["model"]["zones"]["sleep"]["covariance"] = "damaged"
        zones = self.observe(memory, dict(POLICY, model_active=1))
        self.assertEqual(zones["sleep"]["model_hold"], "model_error")
        self.assertEqual(memory["model"]["errors"]["sleep"]["error"], "TypeError")
        self.assertNotIn("sleep", memory["model"]["zones"])
        self.observe(memory)
        self.assertIn("sleep", memory["model"]["zones"])


if __name__ == "__main__":
    unittest.main()
