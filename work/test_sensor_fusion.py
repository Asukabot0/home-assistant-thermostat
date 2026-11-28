"""Multi-sensor zone fusion with learned offsets, bounded failover and stable zone identity."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import test_p0_p1_dispatch as fixture
from fusion import fuse
import house_policy
from topology import configure, fingerprint
from zone_control import observations


pilot = fixture.pilot
EXAMPLE = Path(pilot.__file__).with_name("topology.example.json")
POLICY = dict(house_policy.POLICY)


def reading(value, at, source="sensor.x", quality="fresh"):
    return {"value": value, "at": at, "source": [source, "ha_report"], "quality": quality}


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.topology = {"sensors": {"p": {"role": "primary"}, "x": {"role": "auxiliary", "weight": 0.5},
                                     "c": {"role": "constraint"}},
                         "sources": {"p": "sensor.p", "x": "sensor.x", "c": "sensor.c"}}
        self.policy = dict(POLICY, fusion_offset_min_pairs=3)

    def zone(self, fusion):
        return {"primary": "p", "sensors": ["x", "c"], "fusion": fusion}

    def learn(self, zone, memory, pairs=4):
        for i in range(pairs):
            readings = {"p": reading(24.0, 100*i, "sensor.p"), "x": reading(23.4, 100*i+30), "c": reading(26, 100*i)}
            fuse(zone, self.topology, readings, memory, 100*i+40, self.policy)
        return 100*pairs

    def test_primary_mode_keeps_the_reading_and_identity_while_learning_offsets(self):
        memory = {}
        self.learn(self.zone("primary"), memory)
        primary = reading(24.2, 500, "sensor.p")
        ref, detail = fuse(self.zone("primary"), self.topology, {"p": primary, "x": reading(23.6, 480)}, memory, 510,
                           self.policy)
        self.assertEqual(ref, primary)
        self.assertAlmostEqual(detail["sensors"]["x"]["offset"], -0.6, places=3)
        self.assertEqual(detail["sensors"]["x"]["pairs"], 5)

    def test_weighted_dropout_holds_the_learned_offset_without_changing_identity(self):
        memory, zone = {}, self.zone("weighted")
        now = self.learn(zone, memory)
        fresh, detail = fuse(zone, self.topology, {"p": reading(24.0, now, "sensor.p"), "x": reading(23.4, now)}, memory,
                             now+10, self.policy)
        stale, held = fuse(zone, self.topology, {"p": reading(24.0, now+600, "sensor.p"),
                                                 "x": reading(23.4, now, quality="stale")}, memory, now+610, self.policy)
        self.assertAlmostEqual(fresh["value"], (24.0+0.5*23.4)/1.5)
        self.assertAlmostEqual(stale["value"], fresh["value"])
        self.assertEqual(stale["source"], fresh["source"])
        self.assertEqual(stale["at"], now+600)
        self.assertEqual((detail["sensors"]["x"]["estimated"], held["sensors"]["x"]["estimated"]), (False, True))

    def test_weighted_dropout_before_offsets_are_learned_keeps_identity(self):
        zone = self.zone("weighted")
        fresh, _ = fuse(zone, self.topology, {"p": reading(24.0, 0, "sensor.p"), "x": reading(23.4, 0)}, {}, 10, self.policy)
        memory = {}
        stale, _ = fuse(zone, self.topology, {"p": reading(24.0, 600, "sensor.p"), "x": reading(23.4, 0, quality="stale")},
                        memory, 610, self.policy)
        self.assertEqual(stale["value"], 24.0)
        self.assertEqual(stale["source"], fresh["source"])

    def test_estimated_dates_by_latest_report_and_fails_over_for_a_bounded_time(self):
        memory, zone = {}, self.zone("estimated")
        now = self.learn(zone, memory)
        newer, _ = fuse(zone, self.topology, {"p": reading(24.0, now, "sensor.p"), "x": reading(23.5, now+200)}, memory,
                        now+210, self.policy)
        self.assertEqual(newer["at"], now+200)
        stale_primary = reading(24.0, now, "sensor.p", quality="stale")
        failover, detail = fuse(zone, self.topology, {"p": stale_primary, "x": reading(23.2, now+1500)}, memory,
                                now+1510, self.policy)
        offset = memory["x"]["offset"]
        self.assertEqual((failover["quality"], failover["at"], detail["failover"]), ("fresh", now+1500, True))
        self.assertAlmostEqual(failover["value"], ((23.2-offset)+0.5*23.2)/1.5)
        self.assertEqual(failover["source"], newer["source"])
        expired, _ = fuse(zone, self.topology, {"p": stale_primary, "x": reading(23.2, now+3700)}, memory,
                          now+3710, self.policy)
        self.assertEqual(expired["quality"], "stale")
        weighted, _ = fuse(self.zone("weighted"), self.topology, {"p": stale_primary, "x": reading(23.2, now+1500)},
                           memory, now+1510, self.policy)
        self.assertEqual(weighted["quality"], "stale")

    def test_failover_deadline_counts_from_the_last_valid_primary_temperature(self):
        memory, zone = {}, self.zone("estimated")
        valid_at = self.learn(zone, memory)-100
        unavailable = {"value": None, "at": valid_at+4400, "source": ["sensor.p", "ha_report"], "quality": "unavailable"}
        late, detail = fuse(zone, self.topology, {"p": unavailable, "x": reading(23.2, valid_at+4450)}, memory,
                            valid_at+4500, self.policy)
        self.assertEqual((late["quality"], detail["failover"]), ("unavailable", False))
        early, detail = fuse(zone, self.topology, {"p": dict(unavailable, at=valid_at+900), "x": reading(23.2, valid_at+1450)},
                             memory, valid_at+1500, self.policy)
        self.assertEqual((early["quality"], detail["failover"]), ("fresh", True))
        self.assertEqual(memory["primary_valid_at"], valid_at)

    def test_offsets_count_each_report_once_and_bound_a_glitch(self):
        memory, zone = {}, self.zone("weighted")
        now = self.learn(zone, memory)
        pairs, offset = memory["x"]["pairs"], memory["x"]["offset"]
        same = {"p": reading(24.0, now-100, "sensor.p"), "x": reading(23.4, now-70)}
        fuse(zone, self.topology, same, memory, now, self.policy)
        self.assertEqual(memory["x"]["pairs"], pairs)
        fuse(zone, self.topology, {"p": reading(24.0, now, "sensor.p"), "x": reading(28.0, now+20)}, memory,
             now+30, self.policy)
        self.assertLessEqual(memory["x"]["offset"]-offset, 0.2+1e-9)

    def test_calibration_offset_shifts_readings_and_resets_learning(self):
        memory, zone = {}, self.zone("weighted")
        self.learn(zone, memory)
        self.topology["sensors"]["x"]["offset_c"] = 0.6
        ref, detail = fuse(zone, self.topology, {"p": reading(24.0, 900, "sensor.p"), "x": reading(23.4, 900)}, memory,
                           910, self.policy)
        self.assertAlmostEqual(ref["value"], 24.0)
        self.assertEqual(detail["sensors"]["x"]["pairs"], 1)


class TopologyFusionTests(unittest.TestCase):
    def definition(self):
        example = json.loads(EXAMPLE.read_bytes())
        example["sensors"]["lounge_auxiliary"]["enabled"] = True
        return example

    def test_estimated_fusion_versions_its_auxiliary_sensors(self):
        example = self.definition()
        primary = configure(example, pilot.ROOMS)["zone_versions"]
        example["zones"]["lounge"]["fusion"] = "estimated"
        estimated = configure(example, pilot.ROOMS)
        self.assertNotEqual(primary["lounge"], estimated["zone_versions"]["lounge"])
        self.assertEqual(primary["sleep"], estimated["zone_versions"]["sleep"])
        example["sensors"]["lounge_auxiliary"]["offset_c"] = -0.4
        self.assertNotEqual(estimated["zone_versions"]["lounge"], configure(example, pilot.ROOMS)["zone_versions"]["lounge"])

    def test_invalid_fusion_or_calibration_is_rejected(self):
        for change in ({"fusion": "kalman"},):
            example = self.definition()
            example["zones"]["lounge"].update(change)
            with self.assertRaises(ValueError):
                configure(example, pilot.ROOMS)
        for offset in (3.5, "0.5", True):
            example = self.definition()
            example["sensors"]["lounge_auxiliary"]["offset_c"] = offset
            with self.assertRaises(ValueError):
                configure(example, pilot.ROOMS)

    def test_primary_zone_identity_is_unchanged_by_fusion_memory(self):
        topology = configure(self.definition(), pilot.ROOMS)
        harness = fixture.DispatchTests()
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        harness.whole_scene(("off", 25, 24.0, 8), ("off", 25, 24.2, 8), ("off", 25, 24.2, 8))
        live = deepcopy(harness.live)
        stamp = "1970-01-01T03:20:01+00:00"
        inputs = {"sensor.example_lounge_auxiliary": {"state": "23.8", "last_reported": stamp}}
        memory = {}
        zones, _ = observations(topology, live, inputs, memory, [], set(), harness.now)
        primary = zones["lounge"]["primary_reading"]
        expected = fingerprint([topology["zone_versions"]["lounge"], primary["source"], zones["lounge"]["target"], False])
        self.assertEqual(memory["trackers"]["lounge"]["identity"], expected)
        self.assertEqual(zones["lounge"]["measured"], primary["value"])
        self.assertEqual(zones["lounge"]["fusion"]["mode"], "primary")


if __name__ == "__main__":
    unittest.main()
