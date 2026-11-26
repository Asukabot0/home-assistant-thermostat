"""Temperature and allocation rules for the connected rooms."""
import json
from pathlib import Path
from statistics import median

from model import automatic_setpoint, numeric, setpoint_bounds


POLICY = json.loads(Path(__file__).with_name("house_policy.json").read_text())
ACTIVE = {"heat", "cool"}
INDEPENDENT = ("bedroom", "living")


def recent_trend(history, sample, now, since, policy=POLICY):
    """Estimate change using independent reports after the latest control context."""
    identity = tuple(sample.get(key) for key in ("adapter_mode", "reported", "desired", "temperature_source"))
    points = {}
    for row in reversed([*history, sample]):
        if row["tick"] < now - policy["trend_window_seconds"]:
            break
        if tuple(row.get(key) for key in ("adapter_mode", "reported", "desired", "temperature_source")) != identity:
            break
        stamp, value = row["temp_report"], row["measured"]
        if (value is not None and stamp is not None
                and max(since, now-policy["trend_window_seconds"]) < stamp <= now):
            points.setdefault(stamp, value)
    points = sorted(points.items())
    if (len(points) < 3 or points[-1][0] - points[0][0] < policy["minimum_trend_span_seconds"]
            or now - points[-1][0] > policy["trend_report_max_age_seconds"]):
        return None
    rates = [(v2-v1)/(t2-t1) for i, (t1, v1) in enumerate(points) for t2, v2 in points[i+1:]
             if t2-t1 >= 120]
    return median(rates) * policy["forecast_seconds"] if rates else None


def outside_for(history, sample, direction, now, policy=POLICY):
    """Accumulate the excursion across setpoint/mode changes, within one target and sensor."""
    if sample["measured"] is None or direction * (sample["measured"]-sample["desired"]) <= policy["comfort_band"]:
        return 0
    since = sample["tick"]
    for row in reversed(history[:-1]):
        if (since-row["tick"] != 60 or row["desired"] != sample["desired"]
                or row.get("temperature_source") != sample.get("temperature_source")
                or row["measured"] is None or row["temp_report"] is None
                or not -59 <= row["tick"]-row["temp_report"] <= 1200
                or direction * (row["measured"]-sample["desired"]) <= policy["comfort_band"]):
            break
        since = row["tick"]
    return now - since


def thermostat_view(history, sample, now, since, policy=POLICY):
    return {
        "trend": recent_trend(history, sample, now, since, policy),
        "cold_seconds": outside_for(history, sample, -1, now, policy),
        "warm_seconds": outside_for(history, sample, 1, now, policy),
    }


def recovering(delta, predicted, policy=POLICY):
    if predicted is None or not delta:
        return False
    distance = max(0.0, abs(delta) - policy["comfort_band"])
    required = max(policy["minimum_progress_c"], distance * policy["forecast_seconds"] / policy["recovery_horizon_seconds"])
    return predicted * (-1 if delta > 0 else 1) >= required


def temperature_change(mode, current, measured, target, view, limits, *, has_output, weakening=False,
                       policy=POLICY):
    bounds = setpoint_bounds(limits)
    if bounds is None or mode not in ACTIVE:
        return None, "等待有效的模式和软限"
    delta = measured-target
    thermal_direction = 1 if mode == "heat" else -1
    deviation = thermal_direction * delta
    predicted = view["trend"]
    early = (has_output and predicted is not None and abs(deviation) <= policy["comfort_band"]
             and predicted * thermal_direction > 0
             and deviation + predicted * thermal_direction > policy["ideal_band"])
    if weakening:
        step = -thermal_direction * policy["ordinary_step"]
        reason = "逐步减弱无需求方向或冗余输出"
    elif early:
        step = -thermal_direction * policy["ordinary_step"]
        reason = "预计越过理想带，提前减载"
    elif abs(delta) <= policy["comfort_band"]:
        return None, "在容忍带内，维持"
    elif recovering(delta, predicted, policy):
        return None, "室温正在有效回归"
    else:
        dwell = view["warm_seconds" if delta > 0 else "cold_seconds"]
        if dwell < policy["temperature_dwell_seconds"]:
            return None, "等待持续温差"
        magnitude = policy["ordinary_step"]
        if abs(delta) >= policy["large_deviation"] and predicted is not None and predicted * delta >= 0:
            magnitude = policy["large_step"]
        step = -magnitude if delta > 0 else magnitude
        reason = "根据持续温差调温"
    if current > bounds[1] and step < 0:
        return bounds[1], "归位模式软限"
    if current < bounds[0] and step > 0:
        return bounds[0], "归位模式软限"
    proposed = automatic_setpoint(current, step, limits)
    return proposed, reason if proposed is not None else "达到模式软限，等待全屋分配"


def whole_house_need(zones, policy=POLICY):
    deltas = [z["measured"]-z["target"] for z in zones.values()]
    cold = any(delta < -policy["comfort_band"] for delta in deltas)
    warm = any(delta > policy["comfort_band"] for delta in deltas)
    if cold and warm:
        return "conflict"
    return "heat" if cold else "cool" if warm else "hold"
