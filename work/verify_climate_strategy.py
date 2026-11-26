"""Verify threshold-independent policy decisions before live command checks."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "home-assistant/shadow"))
from model import Activity, advise, automatic_setpoint, predict


TICK = 12000
PROFILE = {"target": 24, "quiet": False, "offsets": {"heat": 0, "cool": 0}}
ACTIVE = Activity("高功率运行", "推测·中", "fixture")
LOW = Activity("低功率变频迹象", "推测·低", "fixture")
HEAT_LIMITS = (20, 25)
COOL_LIMITS = (22, 26)


def history(mode, setting, reading, *, minutes=40, outdoors=15.8):
    return [{"tick": TICK-age*60, "temp_report": TICK-age*60,
             "measured": reading(age), "adapter_mode": mode, "reported": setting,
             "desired": 24, "outdoor": outdoors} for age in range(minutes,-1,-1)]


jitter = history("cool", 25, lambda age: 23.1 + 0.1*(age%2))
assert predict(TICK, jitter) == 0

warming = history("heat", 24, lambda age: 24.2 - 0.7*age/15)
assert predict(TICK, warming) == 0.7
ordinary = advise(TICK, warming[-1], PROFILE, warming, ACTIVE, None, {}, fixed_mode="heat")
assert not ordinary["hypothetical_send"]
early = advise(TICK, warming[-1], PROFILE, warming, ACTIVE, None, {}, fixed_mode="heat",
               limits=HEAT_LIMITS, high_power=True)
assert early["early_overshoot"] and early["hypothetical_send"] and early["setpoint"] == 23
ramp = advise(TICK, warming[-1], PROFILE, warming, LOW, None, {}, fixed_mode="heat",
              limits=HEAT_LIMITS, mode_changed_at=TICK-1800)
assert ramp["hypothetical_send"] and ramp["early_overshoot"] and ramp["setpoint"] == 23
waiting = advise(TICK, warming[-1], PROFILE, warming, ACTIVE, None, {}, fixed_mode="heat",
                 limits=HEAT_LIMITS, high_power=True, last_beep_at=TICK-60)
assert waiting["early_overshoot"] and not waiting["hypothetical_send"]
quiet = advise(TICK, warming[-1], dict(PROFILE, quiet=True), warming, ACTIVE, None, {},
               fixed_mode="heat", limits=HEAT_LIMITS, high_power=True)
assert not quiet["hypothetical_send"]

cooling = history("cool", 25, lambda age: 23.8 + 0.7*age/15)
early_cool = advise(TICK, cooling[-1], PROFILE, cooling, ACTIVE, None, {}, fixed_mode="cool",
                    limits=COOL_LIMITS, high_power=True)
assert early_cool["early_overshoot"] and early_cool["hypothetical_send"] and early_cool["setpoint"] == 26

cooling = history("cool", 25, lambda _: 23.0)
weakening = advise(TICK, cooling[-1], PROFILE, cooling, LOW, None, {}, fixed_mode="cool",
                   limits=COOL_LIMITS, easing=True)
assert weakening["hypothetical_send"] and weakening["setpoint"] == 26
wrong_direction = history("cool", 25, lambda _: 26.0)
wrong = advise(TICK, wrong_direction[-1], PROFILE, wrong_direction, LOW, None, {}, fixed_mode="cool",
               limits=COOL_LIMITS, easing=True)
assert not wrong["hypothetical_send"]
for row in cooling[-10:]:
    row["reported"] = 26
reversal = advise(TICK, cooling[-1], PROFILE, cooling, LOW, None, {})
assert reversal["reversal_demand"] == "heat" and reversal["reversal_qualified"]
assert not reversal["hypothetical_send"] and reversal["mode_ready_at"] > TICK

assert automatic_setpoint(22, -1, COOL_LIMITS) is None
assert automatic_setpoint(26, 1, COOL_LIMITS) is None
assert automatic_setpoint(20, -1, HEAT_LIMITS) is None
assert automatic_setpoint(25, 1, HEAT_LIMITS) is None
assert automatic_setpoint(27, -1, HEAT_LIMITS) == 24
print("Verified quantization filtering, high-power and ramp overshoot, easing, and automatic limits.")
