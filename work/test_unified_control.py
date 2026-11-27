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
from zone_control import observations


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
        self.assertEqual(memory["trackers"]["beta"]["changed_at"],self.h.now+50)

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
