"""Coordinate connected rooms before any command enters the durable journal."""
from house_policy import ACTIVE, INDEPENDENT, POLICY, recovering, temperature_change, whole_house_need
from model import increases_output, numeric, setpoint_bounds
from pilot_gate import credible_output, current_power_evidence, eligible, matching_feedback


def acknowledged(actions):
    return [action for action in actions if action["status"] == "proxy_echo"]


def observation_ready(now, action, live, rooms, policy=POLICY):
    if action is None:
        return True
    return now - action["proxy_at"] >= policy["feedback_wait_seconds"] and all(
        action["proxy_at"] < (live[room].get("temperature_report_at") or 0) <= now for room in rooms)


def all_rooms_assessed(live, connected):
    return all(numeric(live[room]["temperature"]) is not None
               and 0 < numeric(live[room]["temperature"]) <= 50
               and numeric(live[room]["signature"][4]) is not None
               and 0 <= live[room]["temperature_age"] <= 1200 for room in connected)


def final_gate(candidate, now, live, readings, connected, state, configurations, *, policy=POLICY):
    room, kind, value, purpose = candidate
    feedback = live[room]
    shadow = readings[room]
    mode, setpoint, _, _, target, _, _ = feedback["signature"]
    if (not shadow["witness"] == "matched" or not shadow["followup_complete"]
            or room in state["manual_holds"]
            or list(feedback["signature"][:3]) != state["baselines"][room]):
        return "等待可信回报或人工恢复"
    if flag := matching_feedback(now, feedback, shadow["sample"], target, reversal=kind == "mode"):
        return flag
    if not current_power_evidence(now, feedback, shadow):
        return "等待本机新鲜功率上报"
    if kind == "temperature":
        bounds = setpoint_bounds(configurations[room]["soft_limits"][mode])
        if bounds is None or not bounds[0] <= value <= bounds[1]:
            return "超出设温边界"
        if purpose == "reversal_initial":
            return None
        decision = {"mode": mode, "setpoint": value, "hypothetical_send": True,
                    "command_kind": "temperature", "rapid_override": False,
                    "house_weakening": not increases_output(mode, setpoint, value), "reason": "house"}
        reason, permitted = eligible(now, feedback["signature"], feedback, dict(shadow, decision=decision),
                                     mode=mode, setpoints=(value,), target=target)
        return reason if permitted is None else None
    adapter = feedback["adapter"]
    if value not in adapter["modes"]:
        return "等待本机模式能力和连续有效功率报告"
    if value == "off":
        if shadow.get("reversal_low_cutoff") is None:
            return "缺少可安全恢复的本地停机低耗参考"
        if now - adapter["last_mode_change"] < policy["min_run_seconds"]:
            return "等待本机最短运行时间"
        return None
    for peer in connected - {room}:
        peer_mode = live[peer]["adapter"]["mode"]
        if peer_mode not in {value, "off", "fan_only"}:
            return "等待相反或未知模式退出"
    parked = state["parked"].get(room)
    proof = shadow.get("power_evidence") or {}
    low = shadow.get("reversal_low_cutoff")
    last = proof.get("last_report")
    since = proof.get("bands", {}).get("low", {}).get("since")
    if (not parked or mode != "off" or low is None or numeric(adapter["power"]) >= low
            or last is None or since is None
            or now - parked["off_at"] < policy["min_off_seconds"]
            or last - max(since, parked["off_at"]) < policy["min_off_seconds"]):
        return "等待本机停机时间和独立低耗证据"
    if parked["mode"] != value and now - parked["started_at"] < policy["reversal_run_seconds"]:
        return "等待本机冷热换向保护"
    return None


def coordinate(live, readings, ready, state, now, configurations, connected, policy=POLICY):
    zone_rooms = tuple(room for room in INDEPENDENT if room in connected)
    if not zone_rooms or not all_rooms_assessed(live, zone_rooms):
        return "unknown", {}, []
    zones = {room: {"measured": numeric(live[room]["temperature"]), "target": numeric(live[room]["signature"][4]),
                    "view": readings[room]["house_view"]} for room in zone_rooms}
    if any(zone["view"] is None for zone in zones.values()):
        return "unknown", {}, []
    need = whole_house_need(zones, policy)
    if need == "conflict":
        return need, {}, []
    actions = acknowledged([a for a in state["commands"] if a["room"] in connected])
    last_house = max(actions, key=lambda a: a["proxy_at"], default=None)
    house_ready = observation_ready(now, last_house, live, zone_rooms, policy)
    modes = {room: live[room]["adapter"]["mode"] for room in connected}
    active = [room for room, mode in modes.items() if mode in ACTIVE]
    attached = {room: ("living" if room == "study" else room) for room in connected}
    active_zones = [room for room in active if attached[room] in zones]
    mixed = len({modes[room] for room in active}) > 1
    reasons, candidates = {}, []
    if need == "hold" and mixed and active_zones:
        # When nobody needs a direction, keep the incumbent or the primary room.
        lead = state.get("house_lead")
        if lead not in active_zones:
            lead = min(active_zones, key=lambda room: (room == "study", room != "bedroom", room))
        direction = modes[lead]
    else:
        direction = need
    opposite = [room for room in active if direction in ACTIVE and modes[room] != direction]
    same = [room for room in active_zones if modes[room] == direction or direction == "hold"]
    lead = state.get("house_lead")
    effective = [room for room in same if need in ACTIVE
                 and (zones[attached[room]]["measured"]-zones[attached[room]]["target"])
                 * (1 if need == "heat" else -1) < 0
                 and recovering(zones[attached[room]]["measured"]-zones[attached[room]]["target"],
                                zones[attached[room]]["view"]["trend"], policy)]
    preferred = effective or same
    if lead not in preferred:
        def lead_priority(room):
            zone = zones[attached[room]]
            deficit = (zone["target"]-zone["measured"]) * (1 if direction == "heat" else -1)
            return (-max(0.0, deficit), room == "study", room != "bedroom", room)
        lead = min(preferred, key=lead_priority, default=None)
    if lead is not None:
        state["house_lead"] = lead
    lead_saturated = False
    if lead is not None and need in ACTIVE:
        lead_zone = zones[attached[lead]]
        lead_view = lead_zone["view"]
        lead_setting = live[lead]["adapter"]["setpoint"]
        limits = setpoint_bounds(configurations[lead]["soft_limits"][need])
        lead_progress = lead_view["trend"]
        lead_saturated = (limits is not None and lead_progress is not None
                          and not recovering(lead_zone["measured"]-lead_zone["target"], lead_progress, policy)
                          and (lead_setting >= limits[1] if need == "heat" else lead_setting <= limits[0]))
    locality_ok = need not in ACTIVE or all(
        z["view"]["trend"] is None or
        ((z["measured"]-z["target"]+z["view"]["trend"]) * (1 if need == "heat" else -1) <= policy["comfort_band"])
        for z in zones.values())
    stable = all(abs(z["measured"]-z["target"]) <= policy["ideal_band"]
                 and z["view"]["trend"] is not None
                 and abs(z["view"]["trend"]) < policy["minimum_progress_c"]
                 for z in zones.values())
    for room in sorted(connected):
        if not ready[room] or attached[room] not in zones:
            continue
        live_room = live[room]
        zone = zones[attached[room]]
        measured, target, view = zone["measured"], live_room["signature"][4], zone["view"]
        mode = modes[room]
        setting = live_room["adapter"]["setpoint"]
        last = max((a for a in actions if a["room"] == room), key=lambda a: a["proxy_at"], default=None)
        elapsed = observation_ready(now, last, live, (attached[room],), policy)
        if not elapsed:
            reasons[room] = "等待动作后观察时间和新室温"
            continue
        if mode == "off":
            if room not in state["parked"] or need not in ACTIVE or opposite or not house_ready or not locality_ok:
                continue
            dwell = view["cold_seconds" if need == "heat" else "warm_seconds"]
            required = policy["temperature_dwell_seconds"] if state["parked"][room]["mode"] == need else policy["mode_dwell_seconds"]
            if dwell < required:
                continue
            progress = view["trend"]
            if progress is None or recovering(measured-target, progress, policy):
                reasons[room] = "等待可信的未改善趋势"
                continue
            if lead is not None and not lead_saturated:
                continue
            if readings[room]["initial_setpoints"].get(need) is not None:
                reasons[room] = "需求持续且现有输出未能改善，按需启机"
                candidates.append((3, -abs(measured-target), room, "mode", need, "energy_start"))
            continue
        shedding = (need == "hold" and stable and house_ready and len(same) > 1 and room != lead)
        if room in opposite or shedding:
            already_reduced = any(a["room"] == room and a.get("purpose") == "coordinated_easing"
                                  and a["proxy_at"] >= live_room["adapter"]["last_mode_change"] for a in actions)
            output = credible_output(now, live_room, readings[room])
            proposed, reason = temperature_change(mode, setting, measured, target, view,
                                                  configurations[room]["soft_limits"][mode], has_output=output,
                                                  weakening=True, policy=policy)
            if proposed is not None and output and not already_reduced:
                reasons[room] = reason
                candidates.append((0, room != "study", room, "temperature", proposed, "coordinated_easing"))
            else:
                reasons[room] = "退出无需求方向或冗余设备"
                candidates.append((1, room != "study", room, "mode", "off", "energy_off"))
            continue
        output = credible_output(now, live_room, readings[room])
        proposed, reason = temperature_change(mode, setting, measured, target, view,
                                              configurations[room]["soft_limits"][mode], has_output=output,
                                              policy=policy)
        reasons[room] = reason
        if proposed is None:
            continue
        increasing = increases_output(mode, setting, proposed)
        if increasing and (need != mode or opposite or not house_ready or not locality_ok
                           or (room != lead and not lead_saturated)
                           or any(m not in {mode, "off", "fan_only"} for m in modes.values())):
            reasons[room] = "等待全屋同向需求、主机或动作后新反馈"
            continue
        candidates.append((2, -abs(measured-target), room, "temperature", proposed, "temperature"))
    return need, reasons, candidates
