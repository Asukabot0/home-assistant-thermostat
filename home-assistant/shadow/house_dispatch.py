"""Coordinate connected rooms before any command enters the durable journal."""
from house_policy import ACTIVE, INDEPENDENT, POLICY, recovering, temperature_change, whole_house_need
from model import increases_output, numeric, setpoint_bounds
from pilot_gate import credible_output, current_power_evidence, eligible, high_overshoot, matching_feedback


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


def excess_initial_output(mode, setting, bounds):
    return bounds is not None and setting is not None and (
        mode == "heat" and setting > bounds[1] or mode == "cool" and setting < bounds[0])


def initial_safety_setting(mode, setting, target, limits):
    bounds = setpoint_bounds(limits)
    if target is None or not excess_initial_output(mode, setting, bounds):
        return None
    return max(bounds[0], min(bounds[1], round(target)))


def final_gate(candidate, now, live, readings, connected, state, configurations, *, policy=POLICY):
    room, kind, value, purpose = candidate
    feedback = live[room]
    shadow = readings[room]
    mode, setpoint, _, _, target, _, _ = feedback["signature"]
    group = "living_study" if room in {"living","study"} else room
    if (feedback["signature"][3] != "auto" or feedback["signature"][5:] != ("标准", False)
            or feedback["native_authorized"] is not False or group in state["holds"]
            or not shadow["witness"] == "matched" or not shadow["followup_complete"]
            or room in state["manual_holds"]
            or list(feedback["signature"][:3]) != state["baselines"][room]):
        return "等待可信回报或人工恢复"
    if flag := matching_feedback(now, feedback, shadow["sample"], target, reversal=kind == "mode" and value != "off"):
        return flag
    if not current_power_evidence(now, feedback, shadow):
        return "等待本机新鲜功率上报"
    if kind == "temperature":
        bounds = setpoint_bounds(configurations[room]["soft_limits"][mode])
        if bounds is None or not bounds[0] <= value <= bounds[1]:
            return "超出设温边界"
        if purpose == "direct_high_easing" and not high_overshoot(now, feedback, shadow, configurations[room], shadow["house_view"], policy):
            return "持续高功率条件已解除"
        if purpose == "initial_safety_easing":
            batch = state.get("transitions", {}).get(group)
            expected = initial_safety_setting(mode, numeric(setpoint), numeric(target),
                                              configurations[room]["soft_limits"][mode])
            if (batch and batch["mode"] == mode and now >= batch["ready_at"]
                    and batch["members"].get(room, {}).get("phase") == "initial" and value == expected):
                return None
            return "继承设温归位条件已变化"
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


def observed_after(now, action, live, room, wait):
    if action is None:
        return True
    return (now-action["proxy_at"] >= wait
            and action["proxy_at"] < (live[room].get("temperature_report_at") or 0) <= now)


def attachment(room):
    return "living" if room == "study" else room


def guard_peers(zones, zone_room, mode, policy):
    direction = 1 if mode == "heat" else -1
    return any(name != zone_room and (
        direction*(zone["measured"]-zone["target"]) >= policy["ideal_band"]-policy["peer_ideal_guard_c"]
        or direction*(zone["measured"]-zone["target"]+(zone["view"]["trend"] or 0)) > policy["ideal_band"])
        for name, zone in zones.items())


def needs_local(zone, mode, policy):
    direction = 1 if mode == "heat" else -1
    delta = zone["measured"]-zone["target"]
    side = "cold" if mode == "heat" else "warm"
    return (direction*delta < -policy["comfort_band"]
            and zone["view"][f"{side}_seconds"] >= policy["temperature_dwell_seconds"]
            and (zone["view"]["trend"] is not None or zone["view"].get(f"{side}_confirmed"))
            and not recovering(delta, zone["view"]["trend"], policy))


def recovery_initial(room, mode, live, shadow, state, configurations, policy=POLICY):
    v = live[room]
    base = shadow["initial_setpoints"].get(mode)
    bounds = setpoint_bounds(configurations[room]["soft_limits"][mode])
    if base is None or bounds is None:
        return None
    target = v["signature"][4]
    measured = numeric(v["temperature"])
    if measured is None or abs(measured-target) > policy["mild_recovery_deviation"]:
        return base
    # Retain a milder comparable setting within one extra step of the
    # conservative baseline. A previous shedding setpoint or proxy memory
    # alone does not establish that an extreme setting can meet demand.
    direction = 1 if mode == "heat" else -1
    mild = min((base, round(target)-direction*policy["ordinary_step"]), key=lambda value: direction*value)
    choices = [mild]
    remembered = numeric(v["adapter"]["setpoint"])
    if v["adapter"]["mode"] == mode and remembered is not None and bounds[0] <= remembered <= bounds[1]:
        choices.append(remembered)
    if room != "study":
        history = state.get("last_mode_setting", {}).get(room, {}).get(mode)
        if history and history.get("target") == target:
            choices.append(history["value"])
        else:
            stopped = next((a for a in reversed(acknowledged(state["commands"]))
                            if a["room"] == room and a["kind"] == "mode" and a["value"] == "off"
                            and a.get("target_at_send") == target and a["before"][0] == mode), None)
            if stopped and numeric(stopped["before"][1]) is not None:
                choices.append(stopped["before"][1])
    from_memory = min(direction*value for value in choices)
    limited = direction*max(from_memory, direction*mild-policy["ordinary_step"])
    return int(max(bounds[0], min(bounds[1], limited)))


def coordinate(live, readings, ready, state, now, configurations, connected, policy=POLICY):
    zone_rooms = tuple(room for room in INDEPENDENT if room in connected)
    if not zone_rooms or not all_rooms_assessed(live, zone_rooms):
        return "unknown", {}, []
    zones = {room: {"measured": numeric(live[room]["temperature"]), "target": numeric(live[room]["signature"][4]),
                    "view": readings[room]["house_view"]} for room in zone_rooms}
    if any(zone["view"] is None for zone in zones.values()):
        return "unknown", {}, []
    need = whole_house_need(zones, policy)
    actions = acknowledged([a for a in state["commands"] if a["room"] in connected])
    last_house = max(actions, key=lambda a: a["proxy_at"], default=None)
    last_strengthening = max((a for a in actions if a["kind"] == "mode" and a["value"] in ACTIVE
                              or a["kind"] == "temperature" and increases_output(a["before"][0], a["before"][1], a["value"])),
                             key=lambda a: a["proxy_at"], default=None)
    last_exit = max((a for a in actions if a["kind"] == "mode" and a["value"] == "off"),
                    key=lambda a: a["proxy_at"], default=None)
    modes = {room: live[room]["adapter"]["mode"] for room in connected}
    active = {room for room, mode in modes.items() if mode in ACTIVE and attachment(room) in zones}
    mixed = len({modes[room] for room in active}) > 1
    lead = state.get("house_lead")
    direction = need
    if need == "hold" and mixed:
        if lead not in active:
            lead = min(active, key=lambda room: (room == "study", room != "bedroom"))
        direction = modes[lead]
    opposite = {room for room in active if direction in ACTIVE and modes[room] != direction}
    same = active - opposite
    demanded = {room for room in zones if need in ACTIVE and needs_local(zones[room], need, policy)}
    local_active = {room for room in same if attachment(room) in demanded and ready[room]}
    preferred = local_active or same
    if lead not in preferred:
        lead = min(preferred, key=lambda room: (room == "study", room != "bedroom"), default=None)
    state["house_lead"] = lead
    stable = all(abs(z["measured"]-z["target"]) <= policy["ideal_band"]
                 and z["view"]["trend"] is not None
                 and abs(z["view"]["trend"]) < policy["minimum_progress_c"] for z in zones.values())
    reasons, candidates = {}, []
    for room in sorted(connected):
        ref = attachment(room)
        if not ready[room] or ref not in zones:
            continue
        live_room = live[room]
        zone = zones[ref]
        measured, target, view = zone["measured"], live_room["signature"][4], zone["view"]
        mode, setting = modes[room], live_room["adapter"]["setpoint"]
        limits = configurations[room]["soft_limits"]
        last = max((a for a in actions if a["room"] == room), key=lambda a: a["proxy_at"], default=None)
        if mode == "off":
            if room not in state["parked"] or need not in ACTIVE or opposite or ref not in demanded:
                reasons[room] = "等待持续需求或原方向退出"
                continue
            same_local = {a for a in same if attachment(a) == ref}
            saturated = any(
                (bounds := setpoint_bounds(configurations[a]["soft_limits"][need])) is not None
                and (live[a]["adapter"]["setpoint"] >= bounds[1] if need == "heat"
                     else live[a]["adapter"]["setpoint"] <= bounds[0]) for a in same_local)
            if same_local and not saturated:
                reasons[room] = "同区域现有设备仍可承担需求"
                continue
            if room == "study" and guard_peers(zones, ref, need, policy):
                reasons[room] = "另一测点接近不利理想边界，优先直接测温区域"
                continue
            if (not observed_after(now, last, live, ref, 0)
                    or not active and not observed_after(now, last_exit, live, ref, 0)
                    or not observed_after(now, last_strengthening, live, ref, policy["feedback_wait_seconds"])
                    or last_strengthening is not None and not observed_after(
                        now, last_strengthening, live, attachment(last_strengthening["room"]), policy["feedback_wait_seconds"])):
                reasons[room] = "等待停机后新室温或前次加码响应"
                continue
            if readings[room]["initial_setpoints"].get(need) is not None:
                reasons[room] = "持续需求已由真实室温确认，按需启机"
                candidates.append((3, (room == "study", -abs(measured-target)), room, "mode", need, "energy_start"))
            continue

        direction_sign = 1 if mode == "heat" else -1
        delta = measured-target
        output = credible_output(now, live_room, readings[room])
        direct = high_overshoot(now, live_room, readings[room], configurations[room], view, policy)
        recent_reduction = (last is not None and last["kind"] == "temperature"
                            and last["before"][0] == mode
                            and last["proxy_at"] >= live_room["adapter"]["last_mode_change"]
                            and not increases_output(mode, last["before"][1], last["value"]))
        own_feedback = observed_after(now, last, live, ref, policy["reduction_wait_seconds"])
        peer_guard = guard_peers(zones, ref, mode, policy)
        adverse = direction_sign*delta > policy["ideal_band"]+1e-6 and not recovering(delta, view["trend"], policy)
        adverse_peer = peer_guard and direction_sign*delta >= -policy["ideal_band"] and output
        shedding = need == "hold" and stable and len(same)>1 and room != lead
        unused = room in opposite or shedding or (need != mode and (adverse or adverse_peer))
        if unused and own_feedback:
            if recent_reduction or not output:
                # A confirmed setpoint alone is not a confirmed reduction of load.
                # A dormant label is only cleared for a sustained opposite need,
                # or when verified ongoing output is worsening comfort.
                if output or room in opposite and demanded:
                    reasons[room] = "减载后仍无本方向需求，退出输出" if recent_reduction else "退出阻塞的相反模式"
                    candidates.append((1, (room != "study", 0), room, "mode", "off", "energy_off"))
                continue

        proposed, reason = temperature_change(mode, setting, measured, target, view, limits[mode],
                                              has_output=output, weakening=unused and output,
                                              direct_high=direct, policy=policy)
        reasons[room] = reason
        if proposed is None:
            if unused and output and own_feedback:
                reasons[room] = "已到软边界且仍有不需要的负荷，退出输出"
                candidates.append((1, (room != "study", 0), room, "mode", "off", "energy_off"))
            continue
        if increases_output(mode, setting, proposed):
            if (mode != need or opposite or ref not in demanded
                    or (lead != room and lead is not None and attachment(lead) == ref)
                    or not observation_ready(now, last_house, live, zone_rooms, policy)):
                reasons[room] = "等待全屋同向需求、主机或动作后新反馈"
                continue
            if peer_guard:
                proposed = min(proposed, setting+policy["ordinary_step"]) if mode=="heat" else max(proposed, setting-policy["ordinary_step"])
                reasons[room] = "另一测点接近不利理想边界，仅本区小步加码"
            candidates.append((4, (0, -abs(delta)), room, "temperature", proposed, "temperature"))
        elif own_feedback:
            purpose = ("direct_high_easing" if direct else "trend_easing" if reason == "预计越过理想带，提前减载"
                       else "coordinated_easing" if unused else "temperature")
            candidates.append((-2 if reason == "归位模式软限" else 0 if unused or direct else 2, (room != "study", -abs(delta)), room,
                               "temperature", proposed, purpose))
        else:
            reasons[room] = "等待本机动作后新室温再减载"
    return need, reasons, candidates
