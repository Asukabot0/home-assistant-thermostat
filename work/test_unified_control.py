"""Exercise the configured controller and real journal with sparse HA feedback."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import test_p0_p1_dispatch as fixture
from live_evidence import attach
from service import power_evidence
from steady_learning import record_episode
from topology import configure, extra_inputs
from zone_control import observations, strengthened_last, trend


pilot=fixture.pilot
EXAMPLE=Path(pilot.__file__).with_name("topology.example.json")


class UnifiedCycleTests(unittest.TestCase):
    def setUp(self):
        self.h=fixture.DispatchTests()
        self.addCleanup(self.h.doCleanups)
        self.h.setUp()
        self.topology=configure(json.loads(EXAMPLE.read_bytes()),pilot.ROOMS)
        self.enterContext(patch.object(pilot,"TOPOLOGY",self.topology))
        self.audit=self.enterContext(patch.object(pilot,"audit"))

    def scene(self, bed=("heat",22,23.4,8), lounge=("off",25,24.0,8), study=("off",25,24.0,8)):
        self.h.whole_scene(bed,lounge,study)
        permit=json.loads(self.h.permit.read_bytes())
        for r,s in permit["rooms"].items():
            if s[0]=="off":
                permit["rooms"][r]=["cool",25,"auto"]
        self.h.permit.write_text(json.dumps(permit))
        self.h.park(*(r for r, state in self.h.live.items() if state["signature"][0]=="off"),ago=800)

    def run_cycle(self, seconds=0, bed=None, lounge=None, *, cache=()):
        self.h.now+=seconds
        for room, degrees in (("bedroom",bed),("living",lounge),("study",lounge)):
            if degrees is not None:
                self.h.live[room]["temperature"]=degrees
            self.h.observe(room,new_report=room not in cache)
            with self.h.db:
                self.h.db.execute("""UPDATE decisions SET value=json_set(value,'$.power_evidence.bands.low.seconds',600)
                                 WHERE room=?""",(room,))
        return self.h.snapshot()

    def qualify(self, bed=None, lounge=None):
        self.run_cycle(bed=bed,lounge=lounge)
        self.run_cycle(600,bed=bed,lounge=lounge)
        self.run_cycle(600,bed=bed,lounge=lounge)
        return self.run_cycle(600,bed=bed,lounge=lounge)

    def trace(self):
        return next(x.kwargs["assessment"] for x in reversed(self.audit.call_args_list)
                    if x.args==("cycle",) and "assessment" in x.kwargs)

    def test_one_sided_recovery_keeps_previously_effective_setting_near_target(self):
        self.scene()
        self.run_cycle(bed=23.4)
        self.run_cycle(600,bed=23.4)
        self.run_cycle(600,bed=23.1)
        state=self.run_cycle(600,bed=23.25,cache=("living","study"))
        self.assertEqual(len(self.h.calls),1)
        self.assertEqual(state["commands"][-1]["value"],23)
        self.run_cycle(20,cache=("bedroom","living","study"))
        self.run_cycle(620,cache=("bedroom",))  # Timers alone may not stack adjustments.
        self.assertEqual(len(self.h.calls),1)
        self.run_cycle(600,bed=23.9)
        self.run_cycle(600,bed=23.9)
        self.assertEqual(len(self.h.calls),1)
        self.assertEqual(self.trace()["zones"]["sleep"]["intent"],"maintain")

    def test_confirmed_return_aborts_increase_and_light_to_heavy_keeps_duration(self):
        self.scene(bed=("heat",22,22.4,8))
        self.run_cycle(bed=22.4)
        self.run_cycle(600,bed=22.6)
        self.run_cycle(600,bed=22.9)
        self.run_cycle(600,bed=23.4)
        self.assertEqual(self.h.calls,[])
        outcome=self.trace()["zones"]["sleep"]
        self.assertEqual(outcome["reason"],"returning")
        self.assertGreaterEqual(outcome["duration"],1200)

    def test_dormant_cooling_exits_before_heating(self):
        self.scene(lounge=("cool",25,24,8),study=("off",25,24,8))
        stopped=self.qualify()
        self.assertEqual([(a[1].get("hvac_mode")) for a in self.h.calls],["off"])
        self.assertEqual(stopped["commands"][-1]["room"],"living")
        self.run_cycle(20)
        self.assertEqual(self.h.calls[-1][1].get("temperature"),23)

    def test_paused_unit_keeps_peer_constraint_and_blocks_auxiliary_strengthening(self):
        self.scene(bed=("heat",25,23.4,10),lounge=("heat",22,24.9,8),study=("heat",22,24.9,8))
        self.h.update_state(manual_holds={"living":"fixture"})
        self.qualify()
        trace=self.trace()
        self.assertEqual(trace["zones"]["lounge"]["measured"],24.9)
        self.assertNotIn("study",trace["feasible"].get("sleep",[]))
        self.assertFalse(any(a[1].get("temperature",0)>22 for a in self.h.calls))
        self.assertEqual(trace["zones"]["sleep"]["intent"],"limited")

    def test_auxiliary_strengthening_preserves_home_tracking_boundary(self):
        for mode, bedroom_temperature, living_temperature in (
            ("cool", 22.7, 24.4), ("heat", 23.3, 23.6)
        ):
            with self.subTest(mode=mode):
                self.h.journal.unlink(missing_ok=True)
                self.h.calls.clear()
                self.scene(bed=(mode, 23, bedroom_temperature, 402),
                           lounge=("off", 24, living_temperature, 8),
                           study=("off", 24, living_temperature, 8))
                bedroom = self.h.live["bedroom"]
                signature = bedroom["signature"]
                bedroom["signature"] = (*signature[:4], 23, *signature[5:])
                self.h.update_state(manual_holds={"living": "fixture", "study": "fixture"})

                self.qualify()

                trace = self.trace()
                self.assertTrue(trace["zones"]["lounge"]["qualified"])
                self.assertEqual(trace["zones"]["lounge"]["direction"], mode)
                self.assertNotIn("bedroom", trace["feasible"]["lounge"])
                self.assertEqual(self.h.calls, [])

    def test_auxiliary_start_requires_home_recovery(self):
        self.scene(bed=("off", 25, 24.0, 8), lounge=("off", 24, 24.6, 8), study=("off", 24, 24.6, 8))
        self.h.update_state(manual_holds={"living": "fixture", "study": "fixture"})

        self.qualify()

        trace = self.trace()
        self.assertEqual(trace["zones"]["sleep"]["intent"], "maintain")
        self.assertTrue(trace["zones"]["lounge"]["qualified"])
        self.assertNotIn("bedroom", trace["feasible"]["lounge"])
        self.assertEqual(self.h.calls, [])

    def test_auxiliary_initial_uses_home_target_and_requires_continuing_recovery(self):
        for measured, initial in ((23.4, 24), (22.7, None)):
            with self.subTest(measured=measured):
                self.h.journal.unlink(missing_ok=True)
                self.h.calls.clear()
                self.scene(bed=("off", 25, 23.4, 8), lounge=("off", 24, 24.6, 8),
                           study=("off", 24, 24.6, 8))
                bedroom = self.h.live["bedroom"]
                signature = bedroom["signature"]
                bedroom["signature"] = (*signature[:4], 23, *signature[5:])
                self.h.update_state(manual_holds={"living": "fixture", "study": "fixture"})

                started = self.qualify()
                self.assertEqual([(a["room"], a["value"], a["purpose"]) for a in started["commands"]],
                                 [("bedroom", "cool", "energy_start")])
                self.assertEqual(started["commands"][0]["control"]["zone"], "lounge")

                guarded = self.run_cycle(60, bed=measured)

                self.assertEqual(guarded["commands"][0]["status"], "proxy_echo")
                if initial is None:
                    self.assertEqual(self.trace()["invalidated_initial"], ["bedroom"])
                    self.assertNotIn("bedroom", guarded["transitions"])
                    self.assertEqual(len(self.h.calls), 1)
                else:
                    self.assertEqual((guarded["commands"][-1]["purpose"], guarded["commands"][-1]["value"]),
                                     ("reversal_initial", initial))

    def test_auxiliary_strengthening_waits_for_own_temperature_report(self):
        self.scene(bed=("cool", 24, 24.4, 402), lounge=("off", 24, 24.6, 8), study=("off", 24, 24.6, 8))
        self.h.update_state(manual_holds={"living": "fixture", "study": "fixture"})
        first = self.qualify()
        self.assertEqual(len(self.h.calls), 1)
        self.assertEqual((first["commands"][-1]["purpose"], first["commands"][-1]["control"]["zone"]),
                         ("zone_strengthen", "lounge"))

        self.run_cycle(620, cache=("bedroom",))

        self.assertTrue(self.trace()["zones"]["sleep"]["qualified"])
        self.assertEqual(len(self.h.calls), 1)

        updated = self.run_cycle(60)
        self.assertEqual(len(self.h.calls), 2)
        self.assertEqual((updated["commands"][-1]["purpose"], updated["commands"][-1]["value"]),
                         ("zone_strengthen", 22))

    def test_comfort_overrun_stops_with_fresh_pre_easing_report(self):
        for mode, setting, previous, measured in (("cool", 23, 22, 22.0), ("heat", 24, 25, 25.0)):
            with self.subTest(mode=mode):
                self.h.journal.unlink(missing_ok=True)
                self.h.calls.clear()
                self.scene(bed=(mode, setting, measured, 538))
                bedroom = self.h.live["bedroom"]
                target = 23 if mode == "cool" else 24
                signature = bedroom["signature"]
                bedroom["signature"] = (*signature[:4], target, *signature[5:])
                self.h.recent_adjustment("bedroom", previous, ago=5)
                report_at = bedroom["temperature_report_at"]

                sent = self.run_cycle(cache=("bedroom",))

                self.assertEqual(report_at, self.trace()["zones"]["sleep"]["at"])
                self.assertLess(report_at, sent["commands"][0]["proxy_at"])
                self.assertEqual(self.h.calls, [("services/climate/set_hvac_mode", {
                    "entity_id": pilot.ROOMS["bedroom"]["adapter"], "hvac_mode": "off"
                })])
                self.assertEqual((sent["commands"][-1]["purpose"], sent["commands"][-1]["status"]),
                                 ("energy_off", "accepted"))
                settled = self.run_cycle(20, cache=("bedroom",))
                self.assertEqual(settled["commands"][-1]["status"], "proxy_echo")
                self.assertEqual(len(self.h.calls), 1)

    def test_comfort_overrun_within_minimum_runtime_eases_then_stops_when_dormant(self):
        self.scene(bed=("cool", 24, 23.0, 538))
        bedroom = self.h.live["bedroom"]
        self.h.recent_adjustment("bedroom", 23, ago=5)
        bedroom["adapter"]["last_mode_change"] = self.h.now - 100
        self.h.observe("bedroom", new_report=False)

        eased = self.h.snapshot()

        self.assertEqual(self.h.calls, [("services/climate/set_temperature", {
            "entity_id": pilot.ROOMS["bedroom"]["adapter"], "temperature": 26})])
        self.assertEqual(eased["commands"][-1]["purpose"], "min_run_easing")
        self.assertEqual(self.trace()["rejections"], [])
        echoed = self.run_cycle(20, cache=("bedroom", "living", "study"))
        self.assertEqual(echoed["commands"][-1]["status"], "proxy_echo")
        self.assertEqual(len(self.h.calls), 1)

        bedroom["adapter"]["power"] = 8
        stopped = self.run_cycle(900, bed=23.2)

        self.assertEqual(self.h.calls[-1], ("services/climate/set_hvac_mode", {
            "entity_id": pilot.ROOMS["bedroom"]["adapter"], "hvac_mode": "off"}))
        self.assertEqual(stopped["commands"][-1]["purpose"], "energy_off")

    def test_assistance_without_owner_eases_instead_of_running_on(self):
        self.scene(bed=("cool", 23, 24.0, 402), lounge=("off", 24, 24.9, 8), study=("off", 24, 24.9, 8))
        self.h.update_state(manual_holds={"living": "fixture", "study": "fixture"}, commands=[{
            "room": "bedroom", "kind": "temperature", "value": 23, "before": ["cool", 24, "auto"],
            "sent_at": self.h.now-2500, "proxy_at": self.h.now-2500, "purpose": "zone_strengthen",
            "status": "proxy_echo", "followup": {}, "control": {"zone": "lounge"}}])

        eased = self.run_cycle()

        self.assertIsNone(self.trace()["owner"])
        self.assertEqual((eased["commands"][-1]["purpose"], eased["commands"][-1]["value"]), ("zone_reduce", 24))
        for _ in range(3):
            held = self.run_cycle(600)
        self.assertEqual(len(held["commands"]), 2)
        self.assertEqual(self.h.live["bedroom"]["adapter"]["mode"], "cool")

    def test_assistance_run_without_owner_stops_after_easing(self):
        self.scene(bed=("cool", 25, 24.0, 402), lounge=("off", 24, 24.9, 8), study=("off", 24, 24.9, 8))
        self.h.live["bedroom"]["adapter"]["last_mode_change"] = self.h.now-2500
        started = {"room": "bedroom", "kind": "mode", "value": "cool", "before": ["off", None, "auto"],
                   "sent_at": self.h.now-2500, "proxy_at": self.h.now-2500, "purpose": "energy_start",
                   "status": "proxy_echo", "followup": {}, "control": {"zone": "lounge"}}
        initial = dict(started, kind="temperature", value=25, before=["cool", 24, "auto"], purpose="reversal_initial",
                       sent_at=self.h.now-2480, proxy_at=self.h.now-2480)
        self.h.update_state(manual_holds={"living": "fixture", "study": "fixture"}, commands=[started, initial])

        stopped = self.run_cycle()

        self.assertIsNone(self.trace()["owner"])
        self.assertEqual((stopped["commands"][-1]["purpose"], stopped["commands"][-1]["value"]), ("energy_off", "off"))
        last = next(x.kwargs for x in reversed(self.audit.call_args_list) if x.args == ("cycle",) and "rooms" in x.kwargs)
        self.assertEqual(last["rooms"]["bedroom"]["status"], "exit_after_assistance_ended")

    def test_trend_after_strengthening_is_anchored_on_last_prior_report(self):
        points = [[0, 24.4], [600, 24.4], [1300, 23.9]]
        self.assertIsNone(trend(points, 700, 1310, pilot.POLICY))
        anchored = trend(points, 700, 1310, pilot.POLICY, anchored=True)
        self.assertEqual((anchored["from"], anchored["reports"]), (600, 2))
        self.assertLess(anchored["delta"], -0.4)

    def test_trend_anchor_yields_to_post_command_reports_and_needs_their_span(self):
        reports = [[0, 24.4], [600, 24.6], [900, 24.3], [1200, 23.7]]
        alone = trend(reports, 500, 1210, pilot.POLICY)
        self.assertEqual(trend(reports, 500, 1210, pilot.POLICY, anchored=True), alone)
        self.assertAlmostEqual(alone["delta"], -0.9)
        self.assertIsNone(trend([[0, 24.4], [600, 24.4], [760, 24.3]], 700, 770, pilot.POLICY, anchored=True))

    def test_start_followed_by_its_initial_setting_counts_as_strengthening(self):
        start = {"kind": "mode", "value": "cool", "purpose": "energy_start"}
        initial = {"kind": "temperature", "value": 25, "before": ["cool", 24, "auto"], "purpose": "reversal_initial"}
        reduce = {"kind": "temperature", "value": 25, "before": ["cool", 24, "auto"], "purpose": "zone_reduce"}
        self.assertTrue(strengthened_last([start, initial]))
        self.assertFalse(strengthened_last([start, reduce]))
        self.assertFalse(strengthened_last([initial]))

    def test_comfort_overrun_at_weakest_setting_still_respects_minimum_runtime(self):
        self.scene(bed=("cool", 26, 23.0, 538))
        self.h.recent_adjustment("bedroom", 25, ago=5)
        self.h.live["bedroom"]["adapter"]["last_mode_change"] = self.h.now - 100
        self.h.observe("bedroom", new_report=False)

        self.h.snapshot()

        self.assertEqual(self.h.calls, [])
        self.assertIn([["bedroom", "mode", "off", "energy_off"], "等待本机最短运行时间"], self.trace()["rejections"])

    def test_comfort_overrun_respects_current_observations(self):
        for case in ("stale_temperature", "before_mode_change", "stale_power"):
            with self.subTest(case=case):
                self.h.journal.unlink(missing_ok=True)
                self.h.calls.clear()
                self.scene(bed=("cool", 24, 23.0, 538))
                bedroom = self.h.live["bedroom"]
                self.h.recent_adjustment("bedroom", 23, ago=5)
                if case == "stale_temperature":
                    bedroom["temperature_report_at"] = self.h.now - 1201
                elif case == "before_mode_change":
                    bedroom["adapter"]["last_mode_change"] = self.h.now - 930
                    bedroom["temperature_report_at"] = self.h.now - 950
                self.h.observe("bedroom", new_report=False)
                if case == "stale_power":
                    bedroom["adapter"]["from_power_s"] = 121

                self.h.snapshot()

                self.assertEqual(self.h.calls, [])

    def test_start_inherited_setting_is_safe_and_target_change_keeps_confirmed_mode(self):
        self.scene(bed=("off",22,23.4,8))
        launched=self.qualify()
        self.assertEqual(launched["commands"][-1]["value"],"heat")
        self.h.live["bedroom"]["signature"]=("heat",27,"auto","auto",23,"标准",False)
        state=self.run_cycle(20)
        self.assertEqual(state["commands"][0]["status"],"proxy_echo")
        self.assertEqual(state["commands"][-1]["purpose"],"initial_safety_easing")
        self.assertEqual(state["commands"][-1]["value"],23)
        self.assertEqual(len(self.h.calls),2)
        settled=self.run_cycle(20)
        self.assertEqual(settled["baselines"]["bedroom"][:2],["heat",23])

    def test_http500_and_restarted_prepared_cannot_replay(self):
        self.scene()
        self.run_cycle()
        self.run_cycle(600)
        self.run_cycle(600)
        self.h.now+=600
        self.h.observe("bedroom")
        def fail(*_):
            raise HTTPError("http://fixture",500,"simulated",None,None)
        pilot.cycle(fail)
        state=json.loads(self.h.journal.read_bytes())
        self.assertEqual(state["commands"][-1]["status"],"unconfirmed")
        self.assertIn("bedroom",state["holds"])
        state["commands"][-1]["status"]="prepared"
        self.h.journal.write_text(json.dumps(state))
        self.run_cycle(60)
        self.assertEqual(self.h.calls,[])
        self.assertEqual(json.loads(self.h.journal.read_bytes())["commands"][-1]["status"],"unconfirmed")

    def test_new_zone_and_outdoor_inputs_with_renamed_equipment(self):
        rooms={"a_"+key:value for key,value in pilot.ROOMS.items()}
        example=json.loads(EXAMPLE.read_bytes())
        for info in example["sensors"].values():
            if "from_actuator" in info:
                info["from_actuator"]="a_"+info["from_actuator"]
        example["zones"]={"alpha":example["zones"]["sleep"],"beta":example["zones"]["lounge"],
                          "gamma":{"primary":"office_primary","target":{"celsius":24}}}
        example["connections"]=[{"from":"alpha","to":"beta","coupling":0.35}]
        example["zones"]["beta"]["fusion"]="weighted"
        for item in example["zones"].values():
            if "actuator" in item["target"]:
                item["target"]["actuator"]="a_"+item["target"]["actuator"]
        example["actuators"]={"a_bedroom":{"zone":"alpha","influences":{"alpha":1,"beta":.2},"lock_group":"solo"},
                         "a_living":{"zone":"beta","influences":{"beta":1},"lock_group":"shared"},
                         "a_study":{"zone":"gamma","influences":{"gamma":1,"beta":.25},"lock_group":"shared"}}
        example["sensors"].pop("outdoor_reference")
        for key in ("lounge_auxiliary","office_primary","balcony"):
            example["sensors"][key]["enabled"]=True
        spec=configure(example,rooms)
        self.scene()
        live={"a_"+key:value for key,value in self.h.live.items()}
        stamp="1970-01-01T03:20:01+00:00"
        states={"states/"+s["entity"]:{"state":"24.4" if k=="lounge_auxiliary" else "16" if k=="balcony" else "23.5",
                                        "last_reported":stamp} for k,s in example["sensors"].items() if "entity" in s}
        inputs=extra_inputs(states.__getitem__,spec)
        memory={}
        zones,outdoors=observations(spec,live,inputs,memory,[],set(),self.h.now)
        self.assertEqual(set(zones),{"alpha","beta","gamma"})
        self.assertGreater(zones["beta"]["measured"],24)
        self.assertEqual(zones["gamma"]["measured"],23.5)
        self.assertEqual(outdoors["balcony"]["value"],16)
        cfg=deepcopy(example)
        cfg["observation_version"]="config-switch"
        observations(configure(cfg,rooms),live,inputs,memory,[],set(),self.h.now+50)
        self.assertEqual(memory["trackers"]["beta"]["changed_at"],self.h.now)
        cfg["sensors"]["lounge_auxiliary"]["weight"]=0.6
        observations(configure(cfg,rooms),live,inputs,memory,[],set(),self.h.now+70)
        self.assertEqual(memory["trackers"]["beta"]["changed_at"],self.h.now+70)
        self.assertEqual(memory["trackers"]["alpha"]["changed_at"],self.h.now)

    def test_opposed_real_cycle_yields_then_observes_and_serves_each_zone(self):
        self.scene(bed=("heat",22,23.5,300),lounge=("cool",25,24.5,300),study=("off",25,24.5,8))
        reduced=self.qualify()
        self.assertEqual([(a["room"],a["value"],a["purpose"]) for a in reduced["commands"]],[("bedroom",21,"handoff_reduce")])
        self.run_cycle(20,cache=("bedroom","living","study"))
        self.assertEqual(self.trace()["handoff"]["phase"],"verify_reduction")
        self.assertEqual(len(self.h.calls),1)
        stopping=self.run_cycle(600)
        self.assertEqual(stopping["commands"][-1]["purpose"],"handoff_off")
        self.assertEqual(stopping["commands"][-1]["value"],"off")
        # Simulate an echo with continued 300 W. It gives no authority to
        # apply a stronger cooling instruction, even after the time guard.
        self.run_cycle(20,cache=("bedroom","living","study"))
        proving=self.run_cycle(600)
        self.assertEqual(len(self.h.calls),2)
        self.assertEqual(proving["commands"][-1]["status"],"proxy_echo")
        self.assertEqual(self.trace()["handoff"]["phase"],"verify_exit")
        self.h.live["bedroom"]["adapter"]["power"]=8
        self.h.now+=600
        self.run_cycle()
        for _ in range(12):
            for room,row in self.h.live.items():
                row["adapter"]["power"]=8 if row["adapter"]["mode"]=="off" else 300
            end=self.run_cycle(600)
        self.assertTrue(any(a["room"]=="living" and a["purpose"]=="zone_strengthen" and a["status"]=="proxy_echo" for a in end["commands"]))
        self.assertTrue(any(a["room"]=="living" and a["purpose"]=="handoff_off" and a["status"]=="proxy_echo" for a in end["commands"]))
        self.assertTrue(any(a["room"]=="bedroom" and a["value"]=="heat" and a["status"]=="proxy_echo" for a in end["commands"]))
        self.assertEqual(end["holds"],{})
        self.assertFalse(any(a["room"]=="bedroom" and a["value"]=="cool" for a in end["commands"]))

    def test_conflict_explains_comfort_limit_with_two_unmet_demands(self):
        self.scene(bed=("heat",22,23.1,300),lounge=("cool",25,24.9,300))
        self.qualify()
        self.assertEqual(self.h.calls,[])
        self.assertEqual(self.trace()["unmet"],{"lounge":"no_comfort_safe_assignment","sleep":"no_comfort_safe_assignment"})
        last=next(x.kwargs for x in reversed(self.audit.call_args_list) if x.args==("cycle",) and "rooms" in x.kwargs)
        self.assertEqual(last["rooms"]["bedroom"]["status"],"no_comfort_safe_assignment")

    def test_heat_and_cool_recovery_survive_single_boundary_report_and_restart(self):
        for mode, outside, boundary, relapse, settled, setting in (
            ("heat",23.4,23.7,23.6,23.9,22),("cool",24.6,24.3,24.4,24.1,25)
        ):
            with self.subTest(mode=mode):
                self.h.journal.unlink(missing_ok=True)
                self.h.calls.clear()
                self.scene(bed=(mode,setting,outside,8))
                state=self.qualify()
                self.assertEqual(state["commands"][-1]["purpose"],"zone_strengthen")
                self.run_cycle(620,bed=boundary)
                before=self.trace()["zones"]["sleep"]
                self.assertTrue(before["recovery_active"])
                self.assertFalse(before["qualified"])
                self.assertEqual(before["settling_reports"],1)
                persisted=json.loads(self.h.journal.read_bytes())["closed_loop"]["trackers"]["sleep"]["recovery"]
                self.assertEqual(before["evidence_from"],persisted["since"])
                self.run_cycle(610,bed=boundary,cache=("bedroom",))
                self.assertEqual(self.trace()["zones"]["sleep"]["settling_reports"],1)
                self.assertEqual(len(self.h.calls),1)
                self.run_cycle(60,bed=relapse)
                self.assertEqual(len(self.h.calls),2)
                self.assertEqual(self.trace()["zones"]["sleep"]["evidence_from"],persisted["since"])
                self.run_cycle(600,bed=settled)
                self.assertTrue(self.trace()["zones"]["sleep"]["recovery_active"])
                self.run_cycle(600,bed=settled)
                self.assertFalse(self.trace()["zones"]["sleep"]["recovery_active"])
                self.assertEqual(self.trace()["zones"]["sleep"]["intent"],"maintain")

    def test_stale_stream_and_new_target_do_not_reuse_old_recovery(self):
        self.scene()
        self.qualify()
        self.run_cycle(1210,cache=("bedroom",))
        self.assertFalse(self.trace()["zones"]["sleep"]["qualified"])
        self.run_cycle(20)
        old=self.h.live["bedroom"]["signature"]
        self.h.live["bedroom"]["signature"]=(*old[:4],25,*old[5:])
        self.run_cycle(600)
        self.assertFalse(self.trace()["zones"]["sleep"]["recovery_active"])
        self.assertEqual(len(self.h.calls),1)

    def test_unused_disabled_sensor_preserves_journal_and_observation_dependencies(self):
        self.scene()
        self.qualify()
        before=self.run_cycle(20,cache=("bedroom","living","study"))
        obs=before["closed_loop"]["trackers"]
        definition=json.loads(EXAMPLE.read_bytes())
        definition["sensors"]["unused"]={"enabled":False,"role":"auxiliary","entity":"sensor.does_not_exist"}
        after_topo=configure(definition,pilot.ROOMS)
        self.assertEqual(self.topology["action_versions"],after_topo["action_versions"])
        old_sources={key:v["identity"] for key,v in obs.items()}
        memory={"setpoint":24,"target":24,"source_version":old_sources["sleep"],
                "learning_version":self.topology["learning_versions"]["bedroom"]}
        before["closed_loop"]["maintenance"]={"bedroom":{"heat":memory}}
        self.h.journal.write_text(json.dumps(before))
        with patch.object(pilot,"TOPOLOGY",after_topo):
            after=self.run_cycle(20,cache=("bedroom","living","study"))
        self.assertEqual(after["closed_loop"]["trackers"],obs)
        self.assertEqual(after["closed_loop"]["maintenance"]["bedroom"]["heat"],memory)
        self.assertEqual(after["baselines"],before["baselines"])
        self.assertEqual(len(after["commands"]),len(before["commands"]))

    def test_real_sqlite_row_learning_episode(self):
        db=self.h.db
        end=14000
        db.execute("CREATE TABLE observations (tick,adapter_mode,reported,measured,desired,temp_report)")
        db.executemany("INSERT INTO observations VALUES (?,'heat',25,24,24,?)",[(t,t) for t in range(end-1200,end+1,60)])
        db.executemany("INSERT INTO decisions VALUES ('bedroom',?,'x',0,'{\"learning_session\":1}')",
                       [(t,) for t in range(end-1200,end,60)])
        history=db.execute("SELECT * FROM observations ORDER BY tick").fetchall()
        record_episode(db,"bedroom",end,history,{"commands":[]},1,1200)
        self.assertEqual(db.execute("SELECT count(*) FROM stable_episode_quality").fetchone()[0],1)


class LivePowerTests(unittest.TestCase):
    def test_live_pair_bridges_lag_at_same_bands_and_real_gap_still_resets(self):
        context=["sensor.test","1970-01-01T00:00:00+00:00"]
        bands={"low":8,"above_low":8,"high":190,"learning":35}
        reports=[(t,1) for t in range(600,841,60)]
        old=power_evidence(600,[(600,1)],context,bands,None)
        lag=power_evidence(840,reports,context,bands,old)
        shadow={"power_evidence":lag,"ledger_reports":reports,"edges":[190],"reversal_low_cutoff":8,"low_power_cutoff":35}
        active={"power_context":context,"adapter":{"power_report_at":900,"power":2}}
        memory={}
        for _ in range(3):
            merged=attach(910,active,shadow,memory)
            self.assertEqual(merged["power_evidence"]["bands"]["low"]["seconds"],300)
        active["adapter"].update(power_report_at=1080,power=2)
        self.assertEqual(attach(1090,active,shadow,memory)["power_evidence"]["bands"]["low"]["seconds"],0)
        active["power_context"]=["sensor.test","1970-01-01T00:18:01+00:00"]
        active["adapter"]["power_report_at"]=1140
        self.assertEqual(attach(1150,active,shadow,memory)["power_evidence"]["bands"]["low"]["seconds"],0)


if __name__=="__main__":
    unittest.main()
