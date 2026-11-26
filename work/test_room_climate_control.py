from dataclasses import replace
from datetime import datetime,timedelta,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"home-assistant/config/custom_components/room_climate_control"))
from policy import *
from telemetry import Telemetry

class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,11,24,0,0,tzinfo=timezone.utc)
        self.data=Evidence(Reading(24.6,self.now),Reading(390,self.now),"cool",24,"cool",
                           self.now-timedelta(hours=2),"cool",self.now-timedelta(hours=2),
                           self.now-timedelta(minutes=3),True,independent_setpoint=24)
        self.limits=Limits(authorized=True,absolute_mode_verified=True)
        self.request=Request("cool",24)
        self.sent=Pending(Command("setpoint",24),self.now-timedelta(minutes=22))

    def result(self, data=None, request=None, history=None, limits=None, offset=0):
        return decide(self.now,request or self.request,data or self.data,limits or self.limits,
                      history or History(),21,offset)

    def test_observation_and_stale_never_dispatch(self):
        self.assertIsNone(self.result(limits=Limits()).command)
        room=Reading(28,self.now-timedelta(hours=1))
        self.assertEqual(self.result(data=replace(self.data,room=room)).status,"stale_feedback")

    def test_one_step_and_adjustable_cooling_offset(self):
        self.assertEqual(self.result().command,Command("setpoint",23))
        self.assertEqual(self.result(offset=3).status,"deduplicated")

    def test_proxy_high_load_is_not_verified(self):
        d=replace(self.data,independent_mode=None)
        self.assertEqual(activity(self.now,d,self.limits),("load_detected","inferred"))
        self.assertIsNone(self.result(d,Request("off",24)).command)

    def test_unchanged_proxy_mode_does_not_erase_recent_power_evidence(self):
        old=self.now-timedelta(minutes=40)
        self.assertTrue(has_transport_evidence(self.now,old,self.data.power,self.limits))
        self.assertFalse(has_transport_evidence(self.now,old,Reading(99,old),self.limits))
        self.assertFalse(has_transport_evidence(self.now,None,self.data.power,self.limits))

    def test_confirmed_runtime_requires_independent_setpoint_for_setpoint_adjustment(self):
        no_setpoint=replace(self.data,independent_setpoint=None)
        self.assertEqual(self.result(no_setpoint).status,"missing_setpoint")
        disagreement=replace(self.data,adapter_setpoint=27,independent_setpoint=25)
        self.assertEqual(self.result(disagreement).command,Command("setpoint",24))

    def test_validated_ir_still_requires_recent_proxy_mode_for_setpoint_adjustment(self):
        limits=replace(self.limits,ir_setpoint_validated=True)
        uncertain=replace(self.data,independent_mode=None,independent_setpoint=None)
        self.assertEqual(self.result(uncertain,limits=limits).status,"missing_setpoint")
        result=self.result(replace(uncertain,proxy_report_fresh=True),limits=limits)
        self.assertEqual(result.command,Command("setpoint",23))
        self.assertEqual(result.confidence,"inferred")

    def test_quiet_rate_limit_and_pending_confirmation(self):
        self.assertEqual(self.result(request=replace(self.request,quiet=True),history=History(last_command=self.sent)).status,"rate_limit")
        self.assertEqual(self.result(history=History(pending=self.sent)).status,"confirmation_timeout")
        self.assertEqual(self.result(history=History(manual_latched=True)).status,"manual_hold")

    def test_reverse_uses_absolute_off_and_guards(self):
        request=Request("heat",24)
        self.assertEqual(self.result(request=request).command,Command("mode","off"))
        d=replace(self.data,phase_since=self.now-timedelta(minutes=14))
        self.assertEqual(self.result(data=d,request=request).status,"minimum_run")
        off=replace(d,independent_mode="off",last_active_start=self.now-timedelta(minutes=59))
        self.assertEqual(self.result(data=off,request=request).status,"reverse_lock")
        self.assertEqual(self.result(data=replace(off,phase_since=self.now),request=request).status,"minimum_off")

    def test_auto_demand_and_shared_conflict(self):
        self.assertEqual(demand(Request("auto",24),24.51,"heat"),"cool")
        self.assertEqual(demand(Request("auto",24,True),24.51,"heat"),"heat")
        heating=replace(self.data,independent_mode="heat",adapter_mode="heat")
        self.assertIsNone(choose_group([("living",Request("cool"),self.data,self.result()),
                                        ("study",Request("heat"),heating,self.result())]))

    def test_group_allows_safe_off_step_for_verified_reversal(self):
        request=Request("heat",24)
        decision=self.result(request=request)
        self.assertEqual(decision.command,Command("mode","off"))
        self.assertEqual(choose_group([("bedroom",request,self.data,decision)]),"bedroom")

    def test_group_blocks_start_when_neighbor_ir_load_may_be_opposite(self):
        off=replace(self.data,independent_mode="off",adapter_mode="off",
                    last_active_mode="heat")
        request=Request("heat",24)
        start=self.result(data=off,request=request)
        other=replace(self.data,independent_mode=None,adapter_mode="cool")
        observing=self.result(data=other,limits=Limits())
        self.assertEqual(start.command,Command("mode","heat"))
        self.assertEqual(observing.confidence,"inferred")
        self.assertIsNone(choose_group([("living",request,off,start),
                                        ("study",Request(),other,observing)]))

    def test_off_after_boot_requires_reverse_history_or_full_wait(self):
        off=replace(self.data,independent_mode="off",last_active_mode=None,last_active_start=None,
                    history_ready_since=self.now-timedelta(minutes=6))
        self.assertEqual(self.result(data=off).status,"unknown_reverse_history")
        self.assertEqual(self.result(data=replace(off,
            history_ready_since=self.now-timedelta(minutes=61))).command,Command("mode","cool"))

    def test_command_journal_recovers_latest_uncertain_send(self):
        with TemporaryDirectory() as tmp:
            t=Telemetry(Path(tmp)/"journal.sqlite3")
            t.initialize()
            ts=int(self.now.timestamp())
            t.command("room",ts,"setpoint",24,"prepared")
            t.command("room",ts+1,"setpoint",23,"canceled")
            self.assertEqual(t.last_command("room"),
                {"sent":ts,"kind":"setpoint","value":"24","status":"prepared"})
            t.command("room",ts,"setpoint",24,"verified",ts+2)
            self.assertEqual(t.last_command("room")["status"],"verified")

    def test_quantization(self):
        self.assertAlmostEqual(step_toward(16,18,Limits(step=0.3)),16.9)
        self.assertEqual(step_toward(16,26,Limits(step=2)),16)

    def test_daily_metrics_and_learning_quality_gate(self):
        with TemporaryDirectory() as tmp:
            t=Telemetry(Path(tmp)/"sample.sqlite3")
            t.initialize()
            time=int(self.now.timestamp())
            t.record([("room",time,"2026-11-23",24,55,24,23,"cool","load_detected","inferred",200,None,time,"observing",None)])
            t.maintenance(self.now,{"room":{"cool":2,"heat":1}})
            self.assertEqual(t.current_summary("room")["valid_samples"],1)
            self.assertEqual(t.current_models("room",{"cool":2,"heat":1}),{
                "cool":{"offset":2,"generation":None},"heat":{"offset":1,"generation":None}})
            self.assertIsNone(t.quality_model([],[],"cool"))

    def test_daily_metrics_separate_ambiguous_commands_and_reset_gaps(self):
        def observation(tick, measured, desired, quality, mode):
            return {"tick":tick,"measured":measured,"desired":desired,"quality":quality,
                    "observed":mode,"confidence":"verified","watts":None}

        metrics=Telemetry.evaluate([
            observation(0,26,24,"observing","off"),
            observation(60,25,24,"stale_feedback","off"),
            observation(240,24,24,"observing","cool"),
            observation(300,24,23,"observing","cool"),
            observation(360,24,24,"observing","cool"),
        ], [{"status":status} for status in ("prepared","canceled","sent","verified","transport_failed")])
        self.assertEqual(metrics["command_count"],2)
        self.assertEqual(metrics["ambiguous_dispatches"],2)
        self.assertEqual(metrics["possible_beeps"],4)
        self.assertEqual(metrics["verified_command_rate"],0.25)
        self.assertEqual(metrics["verified_starts"],0)
        self.assertIsNone(metrics["median_recovery_minutes"])
        self.assertIsNone(metrics["estimated_kwh"])

    def test_weekly_learning_is_bounded_and_can_rollback(self):
        with TemporaryDirectory() as tmp:
            t=Telemetry(Path(tmp)/"model.sqlite3")
            t.initialize()
            monday=datetime(2026,11,23,4,tzinfo=timezone.utc)
            configs={"room":{"heat":0.0,"cool":0.0}}
            def samples(first, measured):
                out=[]
                for day in range(1,4):
                    origin=first-timedelta(days=day)
                    for offset in range(10):
                        tick=int((origin+timedelta(minutes=3*offset)).timestamp())
                        out.append(("room",tick,origin.date().isoformat(),measured,55,24,26,
                                    "heat","heat","verified",440,11,tick,"rate_limit",26))
                t.record(out)
            samples(monday,23.9)
            t.maintenance(monday,configs)
            self.assertEqual(t.current_models("room",configs["room"])["heat"],{"offset":0.5,"generation":1})
            samples(monday+timedelta(days=7),25.2)
            t.maintenance(monday+timedelta(days=7),configs)
            self.assertEqual(t.current_models("room",configs["room"])["heat"],{"offset":0.0,"generation":2})

    def test_manual_mode_offset_versions_only_changed_values(self):
        with TemporaryDirectory() as tmp:
            t=Telemetry(Path(tmp)/"model.sqlite3")
            t.initialize()
            defaults={"cool":2.0,"heat":1.0}
            t.save_manual_offsets("room",{"cool":1.25},t.current_models("room",defaults))
            t.save_manual_offsets("room",{"cool":1.25,"heat":1.0},t.current_models("room",defaults))
            self.assertEqual(t.current_models("room",defaults),{
                "cool":{"offset":1.25,"generation":1},"heat":{"offset":1.0,"generation":None}})

if __name__=="__main__": unittest.main(verbosity=2)
