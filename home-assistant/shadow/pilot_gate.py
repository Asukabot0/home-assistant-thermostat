"""Shared checks for a witnessed, single-step heating pilot."""
from contextlib import closing
from datetime import datetime
import json
from time import time

from cadence import latest_beeps
from house_policy import thermostat_view
from model import Activity, OVERSHOOT_HIGH_POWER_SECONDS, RAPID_REPORT_MAX_AGE, advise, increases_output, numeric
from service import CONFIG, DATA, POWER_REPORT_GAP_SECONDS, initial_setpoint, open_readonly, power_context


ROOMS = json.loads(CONFIG.read_text(encoding="utf-8"))
TEMPERATURE = {room: config["temperature"] for room, config in ROOMS.items()}
WITNESSES = {room: config.get("witness_trial_id") for room, config in ROOMS.items()}
FEEDBACK_WINDOW_SECONDS = 120


def report_age(state, observer_report, now):
    # The observer obtains the latest same-value report within HA. The REST
    # representation may still carry the earlier time for an unchanged state.
    reports = [state["last_reported"]]
    if observer_report:
        reports.append(observer_report)
    return now - max(datetime.fromisoformat(value).timestamp() for value in reports)


def refresh_temperature(sample, live, now):
    if live is None or sample["measured"] is None:
        return sample
    measured = numeric(live["temperature"])
    reported = live.get("temperature_report_at")
    observed = sample.get("observed_at")
    if (measured is not None and measured != sample["measured"] and observed is not None
            and reported is not None and observed <= reported <= now <= observed + FEEDBACK_WINDOW_SECONDS
            and 0 <= live["temperature_age"] <= 1200):
        return dict(sample, measured=measured, temp_report=reported)
    return sample


def current_power_evidence(now, live, shadow):
    evidence = shadow.get("power_evidence")
    reported = evidence.get("last_report") if evidence else None
    return (reported is not None and evidence.get("context") is not None
            and evidence["context"] == live.get("power_context")
            and 0 <= now - reported <= POWER_REPORT_GAP_SECONDS
            and 0 <= live["adapter"]["from_power_s"] <= POWER_REPORT_GAP_SECONDS)


def credible_output(now, live, shadow):
    watts = numeric(live["adapter"]["power"])
    cutoff = shadow.get("reversal_low_cutoff")
    edges = shadow.get("edges", [])
    if cutoff is None and edges:
        cutoff = max(edges)
    return watts is not None and cutoff is not None and watts > cutoff and current_power_evidence(now, live, shadow)


def room_advice(room, fixed_mode="heat", *, offsets=None, mode_changed_at=None, ignore_reversal_beep=False,
                feedback=None, house_since=None):
    with closing(open_readonly(DATA / "shadow.sqlite3")) as db:
        trial = db.execute("SELECT value FROM trials WHERE id=? AND room=?", (WITNESSES[room], room)).fetchone()
        entries = db.execute("SELECT tick,value FROM decisions WHERE room=? ORDER BY tick DESC LIMIT 66", (room,)).fetchall()
        now = time()
        beep = latest_beeps(db, now).get(room) if house_since is None else None
        reversal_beep = (latest_beeps(db, now, for_reversal=True).get(room)
                         if offsets is not None and house_since is None else None)
        limits = ROOMS[room]["soft_limits"]
        initial = {}
        if entries and offsets is not None:
            target = json.loads(entries[0]["value"])["target"]
            initial = {mode: initial_setpoint(db, room, mode, now, target, limits[mode]) for mode in limits}
    if not trial or not entries:
        raise ValueError("Missing witnessed trial or shadow samples")
    witness = json.loads(trial[0])
    history = []
    for row in reversed(entries):
        value = json.loads(row["value"])
        history.append({"tick": row["tick"], "adapter_mode": value["reported_mode"], "reported": value["from_setpoint"],
                        "measured": value["measured"] if value["source_fresh"] else None,
                        "temp_report": value["temperature_report_at"], "desired": value["target"],
                        "temperature_source": value.get("temperature_source"),
                        "outdoor": value.get("outdoor"), "observed_at": value.get("observed_at"),
                        "power_state_updated_at": value.get("power_state_updated_at")})
    latest = refresh_temperature(history[-1], feedback, now)
    decision_at = now if latest is not history[-1] else None
    activity = Activity(value["activity"], value["confidence"], value["power_reason"])
    profile = {"target": value["target"], "quiet": value["quiet"]}
    edges = (value.get("model") or {}).get("edges_w", [])
    power_at, power = value.get("power_report_at"), value.get("power_report_w")
    high = bool(edges and power is not None and power_at is not None and -59 <= latest["tick"]-power_at <= 600
                and power > max(edges))
    high_seconds = value.get("above_high_seconds", 0)
    if feedback is not None:
        live_power = numeric(feedback["adapter"]["power"])
        if not edges or live_power is None or live_power <= max(edges) or not current_power_evidence(now, feedback, value):
            high_seconds = 0
    normal_options = {"fixed_mode": fixed_mode, "last_beep_at": beep.at if beep is not None else None,
                      "limits": limits.get(fixed_mode), "high_power": high, "mode_changed_at": mode_changed_at,
                      "decision_at": decision_at, "high_power_seconds": high_seconds,
                      "high_power_hold": ROOMS[room].get("overshoot_high_power_seconds", OVERSHOOT_HIGH_POWER_SECONDS)}
    advice = advise(latest["tick"], latest, profile, history, activity, None, {}, **normal_options)
    low_seconds = value.get("low_power_seconds", 0)
    reversal = easing = None
    if offsets is not None and house_since is None:
        profile["offsets"] = offsets
        reversal = advise(latest["tick"], latest, profile, history, activity, None, {},
                          last_beep_at=reversal_beep.at if reversal_beep is not None else None,
                          reversal_from_low_power=low_seconds >= 300, ignore_beep=ignore_reversal_beep,
                          decision_at=decision_at, limits=limits)
        if fixed_mode in limits:
            easing = advise(latest["tick"], latest, profile, history, activity, None, {}, **normal_options, easing=True)
    view = thermostat_view(history, latest, now, house_since) if house_since is not None and latest["measured"] is not None else None
    return {"witness": witness["physical"], "last_recorded_beep": beep._asdict() if beep is not None else None,
            "followup_complete": witness["completed_followup"], "sample": latest, "decision": advice,
            "house_view": view,
            "reversal": reversal, "easing": easing, "initial_setpoints": initial,
            "reversal_beep_at": reversal_beep.at if reversal_beep is not None else None,
            "low_power_cutoff": value.get("low_power_cutoff"), "low_power_seconds": low_seconds,
            "reversal_low_cutoff": value.get("reversal_low_cutoff"),
            "power_evidence": value.get("power_evidence"),
            "above_low_seconds": value.get("above_low_seconds", 0), "edges": edges}


def room_feedback(request, room):
    config = ROOMS[room]
    adapter = request("states/" + config["adapter"])
    power = request("states/" + config["power"])
    sensor = request("states/" + TEMPERATURE[room])
    thermostat = request("states/climate." + room + "_room_thermostat")
    now = time()
    attrs = thermostat["attributes"]
    mode = adapter["state"]
    setpoint = adapter["attributes"].get("temperature")
    signature = (mode, setpoint, adapter["attributes"].get("fan_mode"), thermostat["state"],
                 attrs.get("temperature"), attrs.get("preset_mode"), attrs.get("automatic_control_enabled"))
    context = power_context(attrs, config["power"])
    power_age = report_age(power, attrs.get("power_reported_at") if context else None, now)
    temperature_age = report_age(sensor, attrs.get("temperature_reported_at"), now)
    return {"signature": signature, "temperature": sensor["state"], "temperature_age": temperature_age,
            "temperature_report_at": now - temperature_age, "power_context": context,
            "temperature_source": [TEMPERATURE[room], context[1]] if context else None,
            "native_authorized": attrs.get("configured_for_actuation"),
            "offsets": {m: attrs.get(m + "ing_offset") for m in ("heat", "cool")},
            "adapter": {"mode": mode, "setpoint": setpoint, "power": power["state"], "from_power_s": power_age,
                        "power_state_updated_at": datetime.fromisoformat(power["last_updated"]).timestamp(),
                        "modes": adapter["attributes"].get("hvac_modes", []),
                        "updated_at": datetime.fromisoformat(adapter["last_updated"]).timestamp(),
                        "last_mode_change": datetime.fromisoformat(adapter["last_changed"]).timestamp()}}


def matching_feedback(now, live, sampled, target, *, reversal=False):
    adapter = live["adapter"]
    if not (0 <= now-sampled["tick"] <= 180 and 0 <= live["temperature_age"] <= 1200):
        return "温度或采样过期，暂停发令"
    try:
        power = float(adapter["power"])
        measured = float(live["temperature"])
    except (TypeError, ValueError):
        return "温度或功率不可用，暂停发令"
    if not (0 <= power <= 10000 and 0 < measured <= 50):
        return "温度或功率超出可用范围，暂停发令"
    power_max_age = POWER_REPORT_GAP_SECONDS if reversal else 600
    if not 0 <= adapter["from_power_s"] <= power_max_age:
        return "功率上报过期，暂停发令"
    if reversal:
        sampled_updated = sampled.get("power_state_updated_at")
        if sampled_updated is None or adapter["power_state_updated_at"] > sampled_updated + 1e-6:
            return "等待采样确认最新功率状态或重连"
    if (sampled["reported"] != adapter["setpoint"] or sampled["adapter_mode"] != adapter["mode"]
            or sampled["desired"] != target):
        return "伴侣模式、设温或目标已变化，等待重新采样"
    if sampled.get("temperature_source") is not None and sampled["temperature_source"] != live.get("temperature_source"):
        return "温度来源已变化，等待重新采样"
    if sampled["measured"] != measured:
        return "等待有效的近期室温观测"
    return None


def eligible(now, baseline, live, shadow, *, mode="heat", setpoints=(20, 22), target=24, easing=False):
    if live["signature"] != baseline:
        return "检测到伴侣或恒温器设置变化，结束本次试用", None
    if shadow["witness"] != "matched" or not shadow["followup_complete"]:
        return "等待已确认指令的跟踪记录", None
    sampled = shadow["sample"]
    advice = shadow["easing"] if easing else shadow["decision"]
    if advice is None:
        return "等待可用的减载建议", None
    if reason := matching_feedback(now, live, sampled, target):
        return reason, None
    if not advice["hypothetical_send"] or advice["command_kind"] != "temperature":
        return advice["reason"], None
    if advice["rapid_override"] and live["temperature_age"] > RAPID_REPORT_MAX_AGE:
        return "急变读数已过5分钟，等待新上报", None
    proposed = numeric(advice["setpoint"])
    if advice["mode"] != mode or proposed not in setpoints or proposed is None or not 16 <= proposed <= 30:
        return "未通过当前模式与自动设温限制", None
    adapter = live["adapter"]
    current = numeric(adapter["setpoint"])
    if current is None or proposed == current:
        return "当前设温不可用或无需调整", None
    increase = increases_output(mode, current, proposed)
    watts = numeric(adapter["power"])
    edges = shadow["edges"]
    high = bool(edges and watts is not None and watts > max(edges))
    if advice.get("direct_high_overshoot") and (not high or not current_power_evidence(now, live, shadow)):
        return "持续高功率条件已解除", None
    changed_at = advice.get("early_ramp_at")
    ramp = changed_at is not None and 0 <= now-changed_at <= 3600
    if easing:
        if increase:
            return "换向受阻时仅允许减弱输出", None
        low = shadow.get("reversal_low_cutoff")
        if low is None:
            low = shadow["low_power_cutoff"]
        if low is None or watts is None or watts <= low:
            return "当前功率无须减载", None
    elif (not increase and not high and not (advice.get("early_overshoot") and ramp)
          and not (advice.get("house_weakening") and credible_output(now, live, shadow))):
        return "功率不足以佐证空调活动，暂停发令", None
    return advice["reason"], proposed
