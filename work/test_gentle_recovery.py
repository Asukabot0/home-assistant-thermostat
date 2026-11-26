"""Exercise modest comfort recovery through the real dispatch cycle and journal."""
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import test_p0_p1_dispatch as base


class GentleRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.h = base.DispatchTests()
        self.addCleanup(self.h.doCleanups)
        self.h.setUp()
        self.audit = self.enterContext(patch.object(base.pilot, "audit"))

    def scene(self, bedroom=23.3, living=24, *, active=None, study=False):
        h = self.h
        h.journal.unlink(missing_ok=True)
        h.now = base.NOW
        h.calls.clear()
        h.whole_scene(("heat", 22, bedroom, 8), ("cool", 26, living, 8),
                      ("cool", 25, living, 8) if study else None)
        h.park(*(r for r in h.live if r != active), ago=720)
        h.observe("bedroom", cadence=600)
        h.observe("living", cadence=600)
        self.audit.reset_mock()
        return h

    def assessment(self):
        return next((event.kwargs["assessment"] for event in reversed(self.audit.call_args_list)
                     if event.args == ("cycle",) and "assessment" in event.kwargs), {})

    def set_room(self, room, *, degrees=None, target=None, mode=None, setting=None, minutes=0, **history):
        h = self.h
        h.now += minutes*60
        v = h.live[room]
        if degrees is not None:
            v["temperature"] = degrees
        if target is not None:
            v["signature"] = (*v["signature"][:4], target, *v["signature"][5:])
        if mode is not None:
            v["signature"] = (mode, setting, *v["signature"][2:])
            v["adapter"].update(mode=mode, setpoint=setting)
        for room_name in ("bedroom", "living", "study"):
            h.observe(room_name, **(history if room_name==room else {}))
        return v

    def test_each_independent_zone_can_start_one_gentle_unit_in_comfort_band(self):
        for room, mode, bedroom, living in (("bedroom","heat",23.3,24), ("living","cool",24,24.6)):
            with self.subTest(room=room):
                h = self.scene(bedroom,living,study=True)
                state = h.snapshot()
                self.assertEqual(self.assessment()["normal_need"], "hold")
                self.assertEqual(self.assessment()["gentle"]["state"], "selected")
                self.assertEqual(len(h.calls), 1)
                action = state["commands"][-1]
                self.assertEqual((action["room"],action["value"],action["purpose"]), (room,mode,"energy_start"))
                self.assertEqual(action["gentle"]["zone"], room)

    def test_short_cache_stale_and_recovering_inputs_do_not_trigger_gentle(self):
        for reason, options in (("waiting_dwell",{"minutes":19}),
                                ("waiting_independent_reports",{"minutes":20}),
                                ("stale_temperature",{}),
                                ("returning_to_ideal",{"change":.2})):
            with self.subTest(reason=reason):
                h = self.scene(23.4,24)
                h.observe("bedroom", **options)
                if reason=="waiting_independent_reports":
                    with h.db:
                        h.db.execute("""UPDATE decisions SET value = json_set(value, '$.temperature_report_at',
                                     CASE WHEN tick < ? THEN ? ELSE ? END) WHERE room = 'bedroom'""",
                                     (base.TICK-600,base.TICK-1260,base.TICK-600))
                    h.live["bedroom"].update(temperature_report_at=base.TICK-600,temperature_age=610)
                if reason=="stale_temperature":
                    h.live["bedroom"]["temperature_age"] = 1000
                if reason=="returning_to_ideal":
                    # All reports were within the gentle band. The last two
                    # independently report sufficient movement toward it.
                    def history(tick):
                        return 23.40 if tick<base.TICK-300 else 23.49
                    h.live["bedroom"]["temperature"] = 23.49
                    h.observe("bedroom", history={t:history(t) for t in range(base.TICK-1920,base.TICK+1,60)})
                h.snapshot()
                self.assertEqual(h.calls,[])
                self.assertEqual(self.assessment()["gentle"]["zones"]["bedroom"]["reason"], reason)

    def test_low_power_same_mode_uses_one_step_and_waits_for_new_feedback(self):
        h = self.scene(active="bedroom")
        # Earlier confirmed action makes a coherent post-action trend unknown
        # while the independent 20-minute temperature evidence remains valid.
        h.recent_adjustment("bedroom",21,ago=605)
        h.observe("bedroom", cadence=600)
        action = h.snapshot()["commands"][-1]
        self.assertEqual((action["purpose"],action["value"]), ("gentle_temperature",23))
        self.assertIsNone(base.pilot.publish.call_args.args[-1]["bedroom"]["prediction_10m_c"])
        self.set_room("bedroom",minutes=3,new_report=False)
        h.snapshot()
        self.assertEqual(len(h.calls),1)
        self.set_room("bedroom",minutes=8,new_report=False)
        h.snapshot()
        self.assertEqual(len(h.calls),1)
        self.set_room("bedroom",minutes=1)
        action = h.snapshot()["commands"][-1]
        self.assertEqual((action["purpose"],action["value"]), ("gentle_temperature",24))

    def test_initial_uses_conservative_setting_then_exits_without_stopping(self):
        h = self.scene()
        state = h.snapshot()
        started = state["commands"][-1]
        self.set_room("bedroom",mode="heat",setting=20,minutes=1)
        state = h.snapshot()
        action = state["commands"][-1]
        self.assertEqual(self.assessment()["normal_need"], "hold")
        self.assertEqual((action["purpose"], action["value"]), ("reversal_initial",22))
        self.assertEqual(action["transaction"], started["transaction"])
        self.assertIn("gentle", action)
        self.set_room("bedroom",degrees=23.6,minutes=2)
        self.assertIn("gentle", h.snapshot())
        self.assertEqual(len(h.calls),2)
        self.set_room("bedroom",degrees=23.7,minutes=2)
        state = h.snapshot()
        self.assertNotIn("gentle", state)
        self.assertEqual(self.assessment()["gentle"]["exit"]["reason"], "entered_exit_band")
        self.assertEqual(state["baselines"]["bedroom"][0], "heat")
        self.assertEqual(len(h.calls),2)

    def test_opposed_gentle_demands_keep_one_owner_and_never_add_opposing_mode(self):
        h = self.scene(23.3,24.6,study=True)
        first = h.snapshot()["commands"][-1]
        self.assertEqual((first["room"],first["value"]), ("bedroom","heat"))
        self.set_room("bedroom",mode="heat",setting=23,minutes=1)
        state = h.snapshot()
        self.assertEqual(state["gentle"]["zone"],"bedroom")
        self.assertEqual(len(h.calls),1)
        self.set_room("bedroom",degrees=23.8,minutes=3)
        self.assertNotIn("gentle",h.snapshot())
        h.snapshot()
        self.assertEqual(self.assessment()["gentle"]["zones"]["living"]["reason"], "direction_hold")
        self.assertEqual(len(h.calls),1)

    def test_peer_forecast_blocks_strengthening_at_comfort_edge(self):
        h = self.scene(23.3,24.6)
        h.observe("living", change=.4)
        h.snapshot()
        self.assertEqual(h.calls,[])
        self.assertEqual(self.assessment()["gentle"]["zones"]["bedroom"]["reason"], "peer_comfort_risk")

    def test_normal_demand_preempts_opposing_gentle_intent(self):
        h = self.scene()
        h.snapshot()
        self.set_room("bedroom",mode="heat",setting=23,minutes=1)
        h.snapshot()
        self.set_room("living",degrees=25.0,minutes=1)
        state = h.snapshot()
        self.assertNotIn("gentle",state)
        self.assertEqual(self.assessment()["gentle"]["exit"]["reason"],"normal_demand_priority")
        self.assertFalse(any(args.get("hvac_mode")=="cool" for _,args in h.calls[1:]))

    def test_target_change_or_pause_discards_only_unsent_initial(self):
        for change in ("target", "manual", "permit"):
            with self.subTest(change=change):
                h = self.scene()
                h.snapshot()
                self.set_room("bedroom",mode="heat",setting=20,minutes=1,
                              target=23.5 if change=="target" else None)
                if change=="manual":
                    h.update_state(manual_holds={"bedroom":"paused"})
                elif change=="permit":
                    permit=json.loads(h.permit.read_text())
                    permit["enabled"]=False
                    h.permit.write_text(json.dumps(permit))
                state=h.snapshot()
                self.assertNotIn("gentle",state)
                self.assertEqual(len(h.calls),1)
                if change=="permit":
                    self.assertEqual(state["commands"][-1]["status"],"accepted")
                    permit["enabled"]=True
                    h.permit.write_text(json.dumps(permit))
                    state=h.snapshot()
                self.assertEqual(state["commands"][-1]["status"],"proxy_echo")
                self.assertEqual(state["baselines"]["bedroom"],["heat",20,"auto"])
                self.assertEqual(len(h.calls),1)

    def test_guarded_peer_can_cause_weakening_instead_of_more_gentle_output(self):
        h = self.scene(active="bedroom")
        first = h.snapshot()["commands"][-1]
        self.assertEqual((first["purpose"], first["value"]), ("gentle_temperature", 23))
        self.set_room("living", degrees=24.6, minutes=12, change=.45)
        h.live["bedroom"]["adapter"]["power"] = 150
        h.observe("bedroom")
        state = h.snapshot()
        self.assertEqual(self.assessment()["gentle"]["reason"], "peer_comfort_risk")
        self.assertEqual((state["commands"][-1]["purpose"],state["commands"][-1]["value"]),
                         ("coordinated_easing",22))

    def test_target_change_during_accepted_revokes_plan_but_keeps_in_flight_record(self):
        h = self.scene()
        h.auto_echo = False
        h.live["bedroom"]["adapter"]["updated_at"] = h.now-60
        first = h.snapshot()["commands"][-1]
        self.assertEqual(first["status"], "accepted")
        self.set_room("bedroom", target=23.5, minutes=1)
        state = h.snapshot()
        self.assertNotIn("gentle",state)
        self.assertEqual(state["commands"][-1],first)
        self.assertEqual(len(h.calls),1)
        self.assertEqual(self.assessment()["gentle_exit"]["reason"],"target_or_source_changed")

    def test_optional_http_500_and_prepared_or_accepted_after_restart_do_not_resend(self):
        for result in ("http500","prepared","accepted"):
            with self.subTest(result=result):
                h=self.scene()
                h.auto_echo=False
                h.live["bedroom"]["adapter"]["updated_at"]=h.now-60
                if result=="http500":
                    def fail(path, fields):
                        raise HTTPError("http://fixture",500,"fixture",{},None)
                    with patch.object(h,"request",side_effect=fail) as call:
                        state=h.snapshot()
                        self.assertEqual(call.call_count,1)
                    self.assertEqual(state["commands"][-1]["status"],"unconfirmed")
                    self.assertIn("bedroom",h.snapshot()["holds"])
                else:
                    state=h.snapshot()
                    self.assertEqual(state["commands"][-1]["status"],"accepted")
                    if result=="prepared":
                        state["commands"][-1]["status"]="prepared"
                        h.update_state(commands=state["commands"])
                    h.now+=60
                    state=h.snapshot()
                    self.assertEqual(len(h.calls),1)
                    self.assertEqual(state["commands"][-1]["status"],"accepted" if result=="accepted" else "unconfirmed")
                    self.assertFalse(state["holds"] if result=="accepted" else not state["holds"])


if __name__ == "__main__":
    unittest.main()
