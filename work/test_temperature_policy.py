"""Exercise proportional setpoints, overshoot, and asynchronous feedback."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "home-assistant/shadow"))
from model import Activity, advise
from pilot_gate import eligible, matching_feedback, refresh_temperature
from service import power_span


TICK = 12_000
TARGET = 24
HEAT = (20, 25)
COOL = (22, 26)
ACTIVE = Activity("高负荷", "推测·中", "fixture")


def history(measured, setpoint, mode="heat", minutes=22):
    return [{"tick": TICK - minute * 60, "temp_report": TICK - minute * 60,
             "measured": measured, "reported": setpoint, "adapter_mode": mode,
             "desired": TARGET, "observed_at": TICK - minute * 60 + 33}
            for minute in range(minutes, -1, -1)]


def decide(points, mode="heat", *, sample=None, **overrides):
    return advise(TICK, points[-1] if sample is None else sample, {"target": TARGET, "quiet": False},
                  points, ACTIVE, None, {}, fixed_mode=mode, limits=HEAT if mode == "heat" else COOL,
                  **overrides)


class TemperaturePolicyTests(unittest.TestCase):
    def test_above_soft_limit_recovers_in_requested_direction(self):
        result = decide(history(24.6, 27), high_power=True, high_power_seconds=180)
        self.assertTrue(result["hypothetical_send"])
        self.assertEqual(result["setpoint"], 24)

    def test_above_soft_limit_cannot_reverse_heating_increase(self):
        result = decide(history(23.0, 27))
        self.assertFalse(result["hypothetical_send"])
        self.assertIsNone(result["setpoint"])
        self.assertEqual(result["reason"], "设温已在自动限制边界")

    def test_large_small_and_half_degree_deviations(self):
        for measured, start, target in ((25.8, 25, 23), (24.8, 24, 23),
                                        (25.5, 25, 23), (26.5, 25, 22)):
            with self.subTest(room=measured):
                result = decide(history(measured, start))
                self.assertTrue(result["hypothetical_send"])
                self.assertEqual(result["setpoint"], target)

    def test_three_minutes_at_dynamic_high_power_preempt_the_dwell(self):
        high_th = 192
        power = [(TICK - 180, 210), (TICK - 80, 230), (TICK - 25, 205)]
        duration = power_span(TICK, power, lambda watts: watts > high_th)
        for room, mode, setting, expected in ((24.5, "heat", 24, 23), (23.5, "cool", 24, 25)):
            with self.subTest(mode=mode):
                points = history(room, setting, mode, minutes=12)
                self.assertFalse(decide(points, mode, high_power=True, high_power_seconds=179)["hypothetical_send"])
                result = decide(points, mode, high_power=True, high_power_seconds=duration)
                self.assertTrue(result["hypothetical_send"])
                self.assertTrue(result["direct_high_overshoot"])
                self.assertEqual(result["setpoint"], expected)
                self.assertFalse(decide(points, mode, high_power=True, high_power_seconds=duration,
                                        last_beep_at=TICK-60)["hypothetical_send"])
                self.assertFalse(decide(points, mode, high_power=True, high_power_seconds=duration,
                                        high_power_hold=240)["hypothetical_send"])
        interrupted = power[:2] + [(TICK - 60, high_th), power[-1]]
        self.assertEqual(power_span(TICK, interrupted, lambda watts: watts > high_th), 25)

    def test_high_power_does_not_preempt_when_temperature_is_recovering(self):
        points = history(24.5, 24, minutes=12)
        for point in points:
            point["measured"] += (TICK - point["tick"]) / 1800
        self.assertFalse(decide(points, high_power=True, high_power_seconds=180)["hypothetical_send"])

    def test_cross_minute_feedback_recalculates_and_keeps_safety_checks(self):
        points = history(24.8, 25)
        old = points[-1]
        live = {"signature": ("heat", 25, "auto"), "temperature": "25.8", "temperature_age": 10,
                "temperature_report_at": TICK + 105,
                "adapter": {"mode": "heat", "setpoint": 25, "power": 210, "from_power_s": 20}}
        now = TICK + 115
        self.assertIsNotNone(matching_feedback(now, live, old, TARGET))
        new = refresh_temperature(old, live, now)
        self.assertEqual(new["measured"], 25.8)
        result = decide(points, sample=new, decision_at=now)
        self.assertEqual(result["setpoint"], 23)
        shadow = {"sample": new, "witness": "matched", "followup_complete": True,
                  "decision": result, "edges": [192]}
        self.assertEqual(eligible(now, live["signature"], live, shadow, setpoints=(23,))[1], 23)
        self.assertIs(refresh_temperature(old, live, TICK + 154), old)
        live["adapter"]["setpoint"] = 24
        self.assertIsNotNone(matching_feedback(now, live, new, TARGET))


if __name__ == "__main__":
    unittest.main()
