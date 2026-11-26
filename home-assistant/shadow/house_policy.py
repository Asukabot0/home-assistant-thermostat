"""Temperature and allocation rules for the connected rooms."""
import json
from pathlib import Path
from statistics import median

from model import automatic_setpoint, numeric, setpoint_bounds


POLICY = json.loads(Path(__file__).with_name("house_policy.json").read_text())
ACTIVE = {"heat", "cool"}
INDEPENDENT = ("bedroom", "living")


def independent_points(history, sample, now, since, window, *, same_operation):
    """Estimate change using independent reports after the latest control context."""
    keys = ("desired", "temperature_source")
    if same_operation:
        keys += ("adapter_mode", "reported")
    identity = tuple(sample.get(key) for key in keys)
    points = {}
    for row in reversed([*history, sample]):
        if row["tick"] < now - window or row["measured"] is None or row["temp_report"] is None:
            break
        if tuple(row.get(key) for key in keys) != identity:
            break
        stamp, value = row["temp_report"], row["measured"]
        if (value is not None and stamp is not None
                and max(since, now-window) < stamp <= now):
            points.setdefault(stamp, value)
    return sorted(points.items())


def recent_trend(history, sample, now, since, policy=POLICY):
    points = independent_points(history, sample, now, since, policy["trend_window_seconds"], same_operation=True)
    dense = (len(points) >= 3 and points[-1][0] - points[0][0] >= policy["minimum_trend_span_seconds"]
             and now - points[-1][0] <= policy["trend_report_max_age_seconds"])
    if not dense:
        points = independent_points(history, sample, now, since, policy["sparse_trend_window_seconds"], same_operation=True)
        if len(points) < 2 or now - points[-1][0] > policy["sparse_report_age_seconds"]:
            return None
        span = points[-1][0] - points[0][0]
        rapid = (span >= policy["rapid_report_span_seconds"]
                 and abs(points[-1][1]-points[0][1]) >= policy["rapid_report_change_c"]
                 and now-points[-1][0] <= policy["trend_report_max_age_seconds"])
        if span < policy["sparse_report_span_seconds"] and not rapid:
            return None
    rates = [(v2-v1)/(t2-t1) for i, (t1, v1) in enumerate(points) for t2, v2 in points[i+1:]
             if t2-t1 >= 120]
    return median(rates) * policy["forecast_seconds"] if rates else None


def outside_for(history, sample, direction, now, policy=POLICY):
    """Accumulate the excursion across setpoint/mode changes, within one target and sensor."""
    if sample["measured"] is None or direction * (sample["measured"]-sample["desired"]) <= policy["comfort_band"]:
        return 0, now
    since = sample["tick"]
    for row in reversed(history[:-1]):
        if (since-row["tick"] != 60 or row["desired"] != sample["desired"]
                or row.get("temperature_source") != sample.get("temperature_source")
                or row["measured"] is None or row["temp_report"] is None
                or not -59 <= row["tick"]-row["temp_report"] <= 1200
                or direction * (row["measured"]-sample["desired"]) <= policy["comfort_band"]):
            break
        since = row["tick"]
    return now - since, since


def thermostat_view(history, sample, now, since, policy=POLICY):
    view = {"trend": recent_trend(history, sample, now, since, policy), "report_at": sample["temp_report"]}
    for label, direction in (("cold", -1), ("warm", 1)):
        duration, start = outside_for(history, sample, direction, now, policy)
        # A minute row can lag its temperature report by up to one minute.
        points = independent_points(history, sample, now, start-60, 3900, same_operation=False)
        outside = [at for at, temp in points if direction*(temp-sample["desired"]) > policy["comfort_band"]]
        view[f"{label}_seconds"] = duration
        view[f"{label}_confirmed"] = len(outside) >= 2 and outside[-1]-outside[0] >= policy["sparse_report_span_seconds"]
    return view


def recovering(delta, predicted, policy=POLICY):
    if predicted is None or not delta:
        return False
    distance = max(0.0, abs(delta) - policy["comfort_band"])
    required = max(policy["minimum_progress_c"], distance * policy["forecast_seconds"] / policy["recovery_horizon_seconds"])
    return predicted * (-1 if delta > 0 else 1) >= required


def temperature_change(mode, current, measured, target, view, limits, *, has_output, weakening=False, direct_high=False,
                       policy=POLICY):
    bounds = setpoint_bounds(limits)
    if bounds is None or mode not in ACTIVE:
        return None, "等待有效的模式和软限"
    delta = measured-target
    thermal_direction = 1 if mode == "heat" else -1
    deviation = thermal_direction * delta
    predicted = view["trend"]
    early = (has_output and predicted is not None and abs(deviation) <= policy["comfort_band"]
             and predicted * thermal_direction >= policy["minimum_progress_c"]
             and deviation + predicted * thermal_direction > policy["ideal_band"])
    if direct_high:
        step = -thermal_direction * policy["ordinary_step"]
        reason = "持续高功率超过理想线，直接减载"
    elif early:
        step = -thermal_direction * policy["ordinary_step"]
        reason = "预计越过理想带，提前减载"
    elif weakening:
        step = -thermal_direction * policy["ordinary_step"]
        reason = "逐步减弱无需求方向或冗余输出"
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
