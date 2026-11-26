"""Room-specific power regimes and advisory thermostat decisions."""
from dataclasses import dataclass
from math import expm1, isfinite, log1p, sqrt
from statistics import median

from cadence import BEEP_INTERVAL


RAPID_REPORT_MAX_AGE = 5 * 60
OUTDOOR_ASSIST_DELTA = 6.0
REVERSAL_MARGIN = 0.4
OUTDOOR_REVERSAL_MARGIN = 0.1
OVERSHOOT_HIGH_POWER_SECONDS = 180


def numeric(value):
    try:
        value = float(value)
        return value if isfinite(value) else None
    except (TypeError, ValueError):
        return None


def clip(value, minimum, maximum):
    return min(maximum, max(minimum, value))


def temperature_step(delta):
    return max(1, int(abs(delta) + 0.5))


def setpoint_bounds(limits=None):
    lower, upper = limits or (16, 30)
    lower, upper = max(16, lower), min(30, upper)
    return (lower, upper) if lower <= upper else None


def increases_output(mode, current, proposed):
    return (mode == "heat" and proposed > current) or (mode == "cool" and proposed < current)


def automatic_setpoint(prior, step, limits=None):
    bounds = setpoint_bounds(limits)
    if bounds is None or not 16 <= prior <= 30:
        return None
    lower, upper = bounds
    if step < 0 and prior > lower:
        return max(lower, min(prior, upper) + step)
    if step > 0 and prior < upper:
        return min(upper, max(prior, lower) + step)
    return None


def buckets(points, seconds=30):
    groups = {}
    for stamp, value in points:
        if value is not None and value >= 0:
            groups.setdefault(int(stamp) // seconds, []).append(value)
    return [(i * seconds, median(values)) for i, values in sorted(groups.items())]


def calibrate(points):
    """Find supported bands on log power, balancing time buckets instead of event bursts."""
    series = buckets(points)
    if len(series) < 18 or series[-1][0] - series[0][0] < 1800:
        return None
    power = sorted(log1p(w) for _, w in series)
    center = median(power)
    one_loss = sum((x - center) ** 2 for x in power)
    previous_loss = one_loss
    best = None
    for count in (2, 3):
        centers = [power[int((len(power) - 1) * (i + 0.5) / count)] for i in range(count)]
        for _ in range(24):
            groups = [[] for _ in centers]
            for x in power:
                groups[min(range(count), key=lambda i: abs(centers[i] - x))].append(x)
            updated = [median(group) if group else c for c, group in zip(centers, groups)]
            if updated == centers:
                break
            centers = updated
        losses = [sum((x - c) ** 2 for x in group) for c, group in zip(centers, groups)]
        loss = sum(losses)
        if min(map(len, groups)) < max(4, int(len(power) * 0.07)) or not previous_loss:
            continue
        deviations = [sqrt(loss / len(g)) for loss, g in zip(losses, groups)]
        separated = all(b - a > max(log1p(0.35), da + db)
                        for a, b, da, db in zip(centers, centers[1:], deviations, deviations[1:]))
        if not separated or loss > previous_loss * (0.65 if count == 2 else 0.70):
            continue
        previous_loss = loss
        best = {"centers_w": [round(expm1(c), 2) for c in centers],
                "edges_w": [round(expm1((a + b) / 2), 2) for a, b in zip(centers, centers[1:])],
                "counts": [len(g) for g in groups], "samples": len(power),
                "span_minutes": (series[-1][0] - series[0][0]) // 60,
                "fit": round(1 - loss / one_loss, 3)}
    return best


def band(watts, model):
    return sum(watts > edge for edge in model["edges_w"])


@dataclass(frozen=True)
class Activity:
    state: str
    confidence: str
    reason: str


def identify(tick, sample, recent, model, history):
    unknown = lambda note: Activity("未知", "未知", note)
    if not sample or sample["watts"] is None or not recent or recent[-1][1] != sample["watts"]:
        return unknown("缺少与当前采样相符的功率上报")
    if tick - recent[-1][0] > 600:
        return unknown("最近一次功率上报超过10分钟")
    if not model:
        return unknown("功率历史尚未形成稳定分群")
    power = sample["watts"]
    now_band = band(power, model)
    high = len(model["edges_w"])
    mode = sample["adapter_mode"]
    if mode in {"off", "fan_only"}:
        if now_band != 0:
            return unknown("伴侣模式与功率分群不一致")
        state = "待机迹象" if mode == "off" else "送风迹象"
        return Activity(state, "推测·低", "伴侣模式与该机低功率组一致")
    if mode not in {"heat", "cool"}:
        return unknown("伴侣当前模式不明确")

    if now_band == high:
        recent_minutes = [r for r in history if 0 <= tick - r["tick"] <= 720]
        if (mode == "heat" and sample["outdoor"] is not None and sample["outdoor"] <= 8
            and len(recent_minutes) >= 6 and all(r["adapter_mode"] == "heat" and r["watts"] is not None for r in recent_minutes)
            and all(b["tick"]-a["tick"] == 60 for a,b in zip(recent_minutes,recent_minutes[1:]))):
            bands = [band(r["watts"], model) for r in recent_minutes]
            last_high = next((i for i in range(len(bands)-2, -1, -1) if bands[i] == high), None)
            if last_high is not None and 2 <= len(bands)-last_high-2 <= 8 and all(b == 0 for b in bands[last_high+1:-1]):
                return Activity("疑似除霜后恢复", "推测·低", "低外温、伴侣持续报制热，功率呈短时高低高")
        trail = [x for x in recent if x[1] is not None and -59 <= tick - x[0] <= 180]
        if len(trail) >= 3 and band(trail[-1][1], model) == high:
            last_lower = next((i for i in range(len(trail)-2, -1, -1) if band(trail[i][1], model) == 0), None)
            if (last_lower is not None and last_lower >= 1 and band(trail[last_lower-1][1], model) == 0
                and trail[-1][0] - trail[last_lower][0] <= 90):
                return Activity("压缩机启动迹象", "推测·低", "连续低负荷后短时间跳入该机高功率组")
        mode_text = "制热" if mode == "heat" else "制冷"
        return Activity(f"高功率运行（伴侣报{mode_text}）", "推测·中", "实时功率位于该机独立校准的高组")
    if high == 2 and now_band == 1:
        return Activity("低功率变频迹象", "推测·低", "位于该机低组与高组之间，运行部件未独立确认")
    return unknown("低功率组可能含送风、待机或低负荷，无法区分")


def unique_temperature(history):
    return sorted({r["temp_report"]: r["measured"] for r in history
                   if r["temp_report"] is not None and r["measured"] is not None}.items())


def predict(tick, history):
    points = unique_temperature(r for r in history if 0 <= tick - r["tick"] <= 2700)
    if len(points) < 3 or points[-1][0] - points[0][0] < 600 or tick - points[-1][0] > 1200:
        return None
    edge_count = min(3, len(points)//2)
    initial = median(value for _, value in points[:edge_count])
    final = median(value for _, value in points[-edge_count:])
    if abs(final-initial) < 0.2 - 1e-6:
        return 0.0
    rates = [(v2-v1)*900/(t2-t1) for i,(t1,v1) in enumerate(points) for t2,v2 in points[i+1:] if t2-t1 >= 120]
    if not rates:
        return None
    return round(clip(median(rates), -2, 2), 2)


def rapid_disturbance(tick, window, direction, target, last_beep_at):
    points = [(stamp, value) for stamp, value in unique_temperature(window) if stamp > last_beep_at]
    if len(points) < 2:
        return False
    stamp, value = points[-1]
    return (-59 <= tick-stamp <= RAPID_REPORT_MAX_AGE and direction * (value-target) >= 1.5
            and direction * (value-points[-2][1]) >= 0
            and any(120 <= stamp-before <= 900 and direction * (value-previous) >= 1.0
                    for before, previous in points[:-1]))


def outdoor_support(outdoor, target):
    if outdoor is None:
        return None
    if outdoor <= target - OUTDOOR_ASSIST_DELTA:
        return "heat"
    if outdoor >= target + OUTDOOR_ASSIST_DELTA:
        return "cool"
    return None


def mode_run(tick, history, mode):
    """Return the start of contiguous observed mode and any observed preceding mode."""
    since = tick
    for row in reversed(history[:-1]):
        if since-row["tick"] != 60 or tick-row["tick"] > 3600:
            break
        if row["adapter_mode"] != mode:
            return since, row["adapter_mode"]
        since = row["tick"]
    return since, None


def advise(tick, sample, profile, history, activity, last, estimates, *, fixed_mode=None, last_beep_at=None,
           reversal_from_low_power=False, limits=None, high_power=False, easing=False, ignore_beep=False,
           mode_changed_at=None, decision_at=None, high_power_seconds=0,
           high_power_hold=OVERSHOOT_HIGH_POWER_SECONDS):
    easing = easing and fixed_mode in {"cool", "heat"}
    value = sample["measured"] if sample else None
    target = profile["target"]
    delta = value - target if value is not None else None
    comfort = 0.95 if profile["quiet"] else 0.7
    mode = fixed_mode or ("cool" if delta is not None and delta > comfort else
            "heat" if delta is not None and delta < -comfort else
            sample["adapter_mode"] if sample and sample["adapter_mode"] in {"heat", "cool"} else "hold")
    if isinstance(limits, dict):
        limits = limits.get(mode)
    result = {"mode": mode, "setpoint": None, "send_after": None, "hypothetical_send": False,
              "command_kind": None, "rapid_override": False, "weather_bias": None,
              "weather_assisted_reversal": False, "reversal_half_band": None,
              "reversal_demand": None, "reversal_qualified": False, "mode_ready_at": None,
              "early_overshoot": False, "early_ramp_at": None, "direct_high_overshoot": False,
              "prediction_15m_delta": predict(tick, history), "target": target,
              "measured": value, "reason": "", "confidence": activity.confidence}

    def hold(reason):
        result["reason"] = reason
        return result

    if (not sample or value is None or sample["temp_report"] is None
        or not -59 <= (tick if decision_at is None else decision_at) - sample["temp_report"] <= 1200):
        return hold("室温缺失或已过期")
    outdoor = numeric(sample["outdoor"]) if fixed_mode is None else None
    result["weather_bias"] = outdoor_support(outdoor, target)
    prior_mode = sample["adapter_mode"]
    if fixed_mode and prior_mode != fixed_mode:
        return hold("伴侣已离开试点模式，等待手动核对")
    predicted = result["prediction_15m_delta"]
    weakening_step = 1 if prior_mode == "cool" else -1
    direction_of_overshoot = -weakening_step
    direct_high = (fixed_mode in {"cool", "heat"} and not profile["quiet"] and high_power
                   and high_power_seconds >= high_power_hold and direction_of_overshoot * delta >= 0.5
                   and (predicted is None or direction_of_overshoot * predicted > -0.2))
    early = False
    if fixed_mode in {"cool", "heat"} and predicted is not None and not easing and not profile["quiet"]:
        observed_since, previous_mode = mode_run(tick, history, fixed_mode)
        changed_at = mode_changed_at
        if changed_at is None and previous_mode in {"cool", "heat"}:
            changed_at = observed_since
        ramp = changed_at is not None and 0 <= tick-changed_at <= 3600
        # Heating overshoots on a rising trend and cooling on a falling trend.
        early = ((ramp or high_power) and direction_of_overshoot*delta >= 0
                 and direction_of_overshoot*predicted > 0
                 and direction_of_overshoot*(delta+predicted) > comfort + 1e-6)
        if early and ramp:
            result["early_ramp_at"] = changed_at
    early = early or direct_high
    if mode == "hold" or (abs(delta) <= comfort and not early):
        return hold("在舒适区内，维持现状")
    switching = prior_mode in {"heat", "cool"} and prior_mode != mode
    threshold = comfort
    if switching:
        weather_assisted = not profile["quiet"] and mode == result["weather_bias"]
        margin = OUTDOOR_REVERSAL_MARGIN if weather_assisted else REVERSAL_MARGIN
        threshold = round(comfort + margin, 2)
        result.update(weather_assisted_reversal=weather_assisted, reversal_half_band=threshold)
        if not profile["quiet"] and abs(delta) > threshold + 1e-6:
            result["reversal_demand"] = mode
        observed_since, _ = mode_run(tick, history, prior_mode)
        result["mode_ready_at"] = max(observed_since + 3600, last["tick"] + 3600 if last else tick)
    prior = sample["reported"]
    if fixed_mode or mode == prior_mode:
        if prior is not None and 16 <= prior <= 30:
            direction = weakening_step if early else -1 if delta > 0 else 1
            if easing and direction != weakening_step:
                return hold("等待换向期间只允许减弱输出")
            result["setpoint"] = automatic_setpoint(prior, direction * temperature_step(delta), limits)
    else:
        offset = profile["offsets"][mode]
        estimate = estimates.get(mode) if switching else None
        if estimate:
            offset = estimate["offset_c"]
        # Bounded feed-forward. It is a heuristic, not a learned causal effect.
        load = clip((target - outdoor) / 18, -1, 1) if outdoor is not None else 0
        trim = clip((target - value) * 0.5, -1.5, 1.5)
        result["setpoint"] = clip(round(target + offset + load + trim), 16, 30)
    if profile["quiet"]:
        return hold("睡眠静音预设已开启，保留建议并暂缓发送")
    duration = 1800 if switching else 1200
    direction = (1 if delta > 0 else -1) if fixed_mode else (1 if mode == "cool" else -1)
    window = [r for r in history if tick-duration <= r["tick"] <= tick]
    # The thermal dwell survives a tracked, deliberate weakening command.
    # Same-mode requests still require their own continuous stable setpoint.
    sustained = (len(window) >= duration//60+1
                 and all(b["tick"]-a["tick"] == 60 for a,b in zip(window,window[1:]))
                 and all(r["measured"] is not None and r["temp_report"] is not None
                         and -59 <= r["tick"]-r["temp_report"] <= 1200
                         and r["adapter_mode"] == prior_mode and (switching or r["reported"] == prior)
                         and abs(r["desired"] - target) <= 0.01
                         and direction * (r["measured"]-target) > threshold + 1e-6 for r in window))
    returning = predicted is not None and abs(value + predicted - target) <= comfort
    if switching:
        result["reversal_qualified"] = sustained and not returning and result["reversal_demand"] is not None
    increase = (not switching and prior is not None and result["setpoint"] is not None
                and increases_output(prior_mode, prior, result["setpoint"]))
    if activity.confidence != "推测·中" and not ((switching and reversal_from_low_power) or early or easing or increase):
        return hold("空调活动证据不足，暂缓建议发送")
    if prior_mode not in {"heat", "cool"} or prior is None:
        return hold("缺少可信的冷热模式或伴侣设温，暂缓建议发送")
    if result["setpoint"] is None or (not switching and result["setpoint"] == prior):
        return hold("设温已在自动限制边界" if limits else "建议设温与伴侣设温一致")
    if not sustained and not early:
        return hold(f"冷热切换需连续30分钟偏离目标超过{threshold:g}℃并保持当前模式" if switching else
                    "等待连续20分钟越界及稳定的伴侣设温")
    if returning and not direct_high:
        return hold("近15分钟趋势预计自行回到舒适区")
    kind = "mode" if switching else "temperature"
    if last and last["mode"] == mode and last.get("command_kind", "temperature") == kind:
        same_proposal = (last.get("reported_mode") == prior_mode if switching else
                         last["setpoint"] == result["setpoint"] and last["from_setpoint"] == prior)
        changed_since = any(r["tick"] > last["tick"] and r["adapter_mode"] in {"off", "heat", "cool"}
                            and (r["adapter_mode"] != prior_mode or (not switching and r["reported"] not in {None, prior}))
                            for r in history)
        if same_proposal and not changed_since:
            return hold("相同的影子指令已计数，等伴侣状态或建议变化")
    if last:
        last_beep_at = max(last["tick"], last_beep_at if last_beep_at is not None else last["tick"])
    beep_ready = last_beep_at + BEEP_INTERVAL if not ignore_beep and last_beep_at is not None else tick
    override = (not switching and not early and beep_ready > tick
                and rapid_disturbance(tick, window, direction, target, last_beep_at))
    earliest = tick if override else max(tick, beep_ready)
    if switching:
        earliest = max(earliest, result["mode_ready_at"])
    result.update(command_kind=kind, send_after=earliest, rapid_override=override, early_overshoot=early,
                  direct_high_overshoot=direct_high)
    if earliest > tick:
        return hold("等待60分钟冷热切换间隔" if switching and result["mode_ready_at"] > tick else
                    "等待从上次有记录的蜂鸣起满30分钟")
    result["hypothetical_send"] = True
    if direct_high:
        return hold("功率持续在高位且越过理想温度线，提前减弱输出")
    if early:
        return hold("依据室温趋势提前减弱输出，防止超调")
    if override:
        return hold("室温快速远离目标，提前建议同模式调温")
    if result["weather_assisted_reversal"]:
        return hold("室外气温支持当前室内温差方向，模拟一条模式指令")
    return hold("持续越过冷热切换滞回，模拟一条模式指令" if switching else "持续越界后建议调温，仅计入影子指令")
