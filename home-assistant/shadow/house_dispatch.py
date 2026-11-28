"""Coordinate connected rooms before any command enters the durable journal."""
from house_policy import ACTIVE, INDEPENDENT, POLICY, TEMP_EPSILON, recovering, temperature_change, whole_house_need
from model import automatic_setpoint, increases_output, numeric, setpoint_bounds
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


def lock_group_for(room, topology):
    return topology["actuators"][room]["lock_group"] if topology else "living_study" if room in {"living", "study"} else room


def final_gate(candidate, now, live, readings, connected, state, configurations, *, policy=POLICY, topology=None, trace=None):
    room, kind, value, purpose = candidate
    feedback = live[room]
    shadow = readings[room]
    mode, setpoint, _, _, target, _, _ = feedback["signature"]
    group = lock_group_for(room, topology)
    known_groups = {lock_group_for(r, topology) for r in live}
    if any(key not in known_groups for key in state["holds"]):
        return "存在未归属当前锁组的暂停记录，已暂停发令"
    if (feedback["signature"][3] != "auto" or feedback["signature"][5:] != ("标准", False)
            or feedback["native_authorized"] is not False or group in state["holds"]
            or not shadow["witness"] == "matched" or not shadow["followup_complete"]
            or room in state["manual_holds"]
            or now < state.get("external_overrides", {}).get(room, {}).get("until", 0)
            or now < max(state.get("start_failures", {}).get(room, {}).get(k, 0) for k in ("retry_at", "hold_until"))
            or list(feedback["signature"][:3]) != state["baselines"][room]):
        return "等待可信回报或人工恢复"
    if topology is None and (flag := matching_feedback(now, feedback, shadow["sample"], target, reversal=kind == "mode" and value != "off")):
        return flag
    if not current_power_evidence(now, feedback, shadow):
        return "等待本机新鲜功率上报"
    if topology is not None:
        home = topology["actuators"][room]["zone"]
        local = trace["zones"][home]
        if local["paused"] or local["quality"] != "fresh" or local["target"] is None:
            return "设备所在区域已暂停或观测失效"
        if kind == "temperature" and mode not in ACTIVE:
            return "当前模式不能调温"
        if kind == "mode" and value not in {*ACTIVE, "off"}:
            return "无效运行模式"
        stronger = (kind == "mode" and value in ACTIVE or kind == "temperature" and
                    purpose != "initial_safety_easing" and increases_output(mode, setpoint, value))
        plan = trace.get("handoff")
        if purpose in {"handoff_reduce", "handoff_off"}:
            phase = "reduce" if purpose == "handoff_reduce" else "exit"
            if not plan or plan["blocker"] != room or plan["phase"] != phase or plan["owner"] != trace["owner"]:
                return "协调退出阶段不再有效"
        if stronger:
            from zone_control import allows_assistance, exit_wait, opposing_allowed
            if (plan and plan["phase"] != "serve" or exit_wait(room,value if kind=="mode" else mode,now,live,readings,state,policy)):
                return "等待相反设备实测低耗和退出后反馈"
            scope = trace["action_scopes"].get(room)
            zone = trace["zones"].get(scope["zone"]) if scope else None
            direction = value if kind == "mode" else mode
            if (not zone or not zone["qualified"] or zone["paused"] or zone["direction"] != direction
                    or scope["zone"] != trace["owner"] or room not in trace["feasible"].get(scope["zone"], ())):
                return "区域需求或作用范围已变化"
            if scope["zone"] != home and not allows_assistance(local, direction):
                return "本区无已确认同向需求，暂停跨区加码"
            modes = {unit: live[unit]["adapter"]["mode"] for unit in connected}
            if any(m not in {direction, "off", "fan_only"} and not opposing_allowed(
                    room, scope["zone"], unit, modes, trace["zones"], topology, trace["output_evidence"])
                   for unit, m in modes.items() if unit != room):
                return "等待相反输出有序退出"
        if power_endpoint := shadow.get("live_power_endpoint"):
            if power_endpoint != feedback["adapter"].get("power_report_at"):
                return "实时功率证据与当前末端不一致"
    plan = state.get("gentle") if topology is None else None
    optional = purpose == "gentle_temperature" or (purpose in {"energy_start", "reversal_initial"} and plan is not None)
    if optional:
        index = tuple(ref for ref in INDEPENDENT if ref in connected)
        if (plan is None or plan["unit"] != room or plan["zone"] not in index
                or target != plan["target"] or live[plan["zone"]]["signature"][4] != plan["target"]
                or live[plan["zone"]]["temperature_source"] != plan["source"]
                or (value if kind == "mode" else mode) != plan["mode"]
                or not all_rooms_assessed(live, index)):
            return "温和需求或目标已变化"
        zones = zones_for(live, readings, index)
        if (any(z["view"] is None for z in zones.values()) or whole_house_need(zones, policy) != "hold"
                or gentle_block(plan, zones, policy)
                or live[plan["zone"]]["temperature_age"] > policy["sparse_report_age_seconds"]):
            return "温和加码条件或另一测点余量已变化"
        if any(live[peer]["adapter"]["mode"] not in {plan["mode"], "off", "fan_only"} for peer in connected-{room}):
            return "等待相反或未知模式退出"
        if purpose == "gentle_temperature" and (not increases_output(mode,setpoint,value)
                                                or abs(value-setpoint) > min(1, policy["gentle_step"])+TEMP_EPSILON):
            return "温和调温须按本方向且最多一步"
    if kind == "temperature":
        bounds = setpoint_bounds(configurations[room]["soft_limits"][mode])
        if bounds is None or not bounds[0] <= value <= bounds[1]:
            return "超出设温边界"
        if purpose == "direct_high_easing" and not high_overshoot(now, feedback, shadow, configurations[room], shadow["house_view"], policy):
            return "持续高功率条件已解除"
        if purpose == "initial_safety_easing":
            batch = state.get("transitions", {}).get(group)
            if batch is None or room not in batch.get("members", {}):
                batch = next((b for b in state.get("transitions", {}).values() if room in b.get("members", {})), None)
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
                                     mode=mode, setpoints=(value,), target=target, match_sample=topology is None)
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
    for peer in (connected - {room}) if topology is None else ():
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


def zones_for(live, readings, zone_rooms):
    return {room: {"measured": numeric(live[room]["temperature"]), "target": numeric(live[room]["signature"][4]),
                   "view": readings[room]["house_view"]} for room in zone_rooms}


def peer_comfort_risk(zones, zone, mode, policy):
    direction = 1 if mode == "heat" else -1
    for room, other in zones.items():
        if room == zone:
            continue
        adverse = direction*(other["measured"]-other["target"])
        progress = direction*(other["view"]["trend"] or 0)
        if adverse + max(0, progress) > policy["comfort_band"]+TEMP_EPSILON:
            return True
    return False


def gentle_deficit(zone, mode):
    direction = 1 if mode == "heat" else -1
    return direction*(zone["target"]-zone["measured"])


def forecast_returns_to_ideal(zone, mode, policy):
    deficit = gentle_deficit(zone, mode)
    direction = 1 if mode == "heat" else -1
    forecast = zone["view"]["trend"]
    return (deficit > policy["ideal_band"] and forecast is not None
            and direction*forecast >= deficit-policy["ideal_band"]-TEMP_EPSILON)


def gentle_block(plan, zones, policy):
    zone = zones[plan["zone"]]
    if peer_comfort_risk(zones, plan["zone"], plan["mode"], policy):
        return "peer_comfort_risk"
    if gentle_deficit(zone, plan["mode"]) <= policy["ideal_band"]+TEMP_EPSILON:
        return "inside_ideal_band"
    if forecast_returns_to_ideal(zone, plan["mode"], policy):
        return "returning_to_ideal"
    return None


def public_gentle(plan):
    return {key: plan[key] for key in ("zone", "unit", "mode", "target", "selected_at")}


def cancel_gentle(state, now, reason, policy=POLICY):
    plan = state.pop("gentle", None)
    if plan is None:
        return None
    state["gentle_guard"] = {"mode": plan["mode"], "until": now+policy["gentle_switch_wait_seconds"]}
    if reason in {"control_paused", "target_or_source_changed", "permit_inactive"}:
        state["gentle_requalify_at"] = now
    return dict(public_gentle(plan), reason=reason)


def gentle_unit(live, ready, configurations, zone, mode, target, policy):
    for room in (zone, "study"):
        if attachment(room) != zone or not ready.get(room):
            continue
        feedback = live[room]
        adapter = feedback["adapter"]
        if feedback["signature"][4] != target or mode not in adapter["modes"]:
            continue
        if adapter["mode"] == "off":
            return room
        if adapter["mode"] == mode:
            setting = numeric(adapter["setpoint"])
            limits = configurations[room]["soft_limits"][mode]
            bounds = setpoint_bounds(limits)
            step = min(1, policy["gentle_step"]) * (1 if mode=="heat" else -1)
            if (setting is not None and bounds and bounds[0] <= setting <= bounds[1]
                    and automatic_setpoint(setting, step, limits) is not None):
                return room
    return None


def choose_gentle(live, zones, control_ready, state, now, normal, connected, configurations, policy, *, acquire):
    trace = {"state": "idle", "reason": "no_qualified_zone", "zones": {}}
    plan = state.get("gentle")
    if normal != "hold":
        reason = "stale_temperature" if normal == "unknown" else "normal_demand_priority"
        trace["reason"] = reason
        if ended := cancel_gentle(state, now, reason, policy):
            trace.update(state="exited", exit=ended)
        return None, trace

    if plan is not None:
        zone, unit, mode = plan["zone"], plan["unit"], plan["mode"]
        if zone not in zones or not control_ready.get(zone) or not control_ready.get(unit):
            reason = "control_paused"
        elif (zones[zone]["target"] != plan["target"] or live[zone]["temperature_source"] != plan["source"]
              or live[unit]["signature"][4] != plan["target"]):
            reason = "target_or_source_changed"
        elif live[zone]["temperature_age"] > policy["sparse_report_age_seconds"]:
            reason = "stale_temperature"
        elif gentle_deficit(zones[zone], mode) <= policy["gentle_exit_band"]+TEMP_EPSILON:
            reason = "entered_exit_band"
        else:
            block = gentle_block(plan, zones, policy)
            trace.update(state="active", reason=block or "held_selection", **public_gentle(plan))
            return plan, trace
        trace.update(state="exited", reason=reason, exit=cancel_gentle(state, now, reason, policy))
        return None, trace

    if not acquire:
        return None, trace
    guard = state.get("gentle_guard")
    if guard and now >= guard["until"]:
        del state["gentle_guard"]
        guard = None
    options = []
    for name, zone in zones.items():
        evidence = zone["view"].get("gentle")
        result = trace["zones"][name] = {}
        if evidence is None or live[name]["temperature_age"] > policy["sparse_report_age_seconds"]:
            result["reason"] = ("stale_temperature" if live[name]["temperature_age"] > policy["sparse_report_age_seconds"]
                                else "no_ideal_band_excursion")
            continue
        mode = evidence["mode"]
        since = max(evidence["since"], state.get("gentle_requalify_at", 0))
        reports = [stamp for stamp in evidence["reports"] if stamp >= since]
        dwell = now-since
        span = reports[-1]-reports[0] if len(reports) >= 2 else 0
        result.update(mode=mode, seconds=dwell, reports=len(reports), report_span=span)
        unit = gentle_unit(live, control_ready, configurations, name, mode, zone["target"], policy)
        proposed = {"zone": name, "unit": unit, "mode": mode, "target": zone["target"],
                    "source": live[name]["temperature_source"], "selected_at": now}
        reason = gentle_block(proposed, zones, policy)
        if dwell < policy["gentle_dwell_seconds"]:
            result["reason"] = "waiting_dwell"
        elif span < policy["gentle_report_span_seconds"]:
            result["reason"] = "waiting_independent_reports"
        elif reason:
            result["reason"] = reason
        elif guard and guard["mode"] != mode:
            result["reason"] = "direction_hold"
        elif not control_ready.get(name) or unit is None:
            result["reason"] = "no_controllable_local_unit"
        elif unit == "study" and guard_peers(zones, name, mode, policy):
            result["reason"] = "prefer_independent_zone_near_peer_boundary"
        else:
            result["reason"] = "qualified"
            opposing = any(live[room]["adapter"]["mode"] in ACTIVE-{mode} for room in connected)
            priority = (opposing, unit=="study", -round(gentle_deficit(zone, mode), 6), -dwell, name)
            options.append((priority, proposed))
    if not options:
        return None, trace
    _, plan = min(options, key=lambda entry: entry[0])
    state["gentle"] = plan
    trace.update(state="selected", reason="qualified_local_demand", **public_gentle(plan))
    return plan, trace


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


def coordinate(live, readings, ready, state, now, configurations, connected, policy=POLICY, *,
               trace=None, control_ready=None, acquire_gentle=True):
    trace = {} if trace is None else trace
    if control_ready is None:
        control_ready = ready
    zone_rooms = tuple(room for room in INDEPENDENT if room in connected)
    zones = zones_for(live, readings, zone_rooms) if all_rooms_assessed(live, zone_rooms) else {}
    normal = (whole_house_need(zones, policy) if zones and all(z["view"] is not None for z in zones.values())
              else "unknown")
    gentle, trace["gentle"] = choose_gentle(live, zones, control_ready, state, now, normal, connected,
                                           configurations, policy, acquire=acquire_gentle)
    trace["normal_need"] = normal
    if normal == "unknown":
        return normal, {}, []
    need = gentle["mode"] if gentle else normal
    gentle_reason = gentle_block(gentle, zones, policy) if gentle else None
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
    demanded = ({gentle["zone"]} if gentle and gentle_reason is None else set()) if gentle else {
        room for room in zones if need in ACTIVE and needs_local(zones[room], need, policy)}
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
            if gentle and room == gentle["unit"] and gentle_reason is not None:
                reasons[room] = "温和恢复暂不加码：" + gentle_reason
                continue
            if gentle and room != gentle["unit"]:
                reasons[room] = "当前温和需求仅分配给一个本区设备"
                continue
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
                reasons[room] = ("理想带外持续偏差已由独立上报确认，温和启机" if gentle else
                                 "持续需求已由真实室温确认，按需启机")
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
        peer_risk = peer_comfort_risk(zones, ref, mode, policy)
        if gentle:
            # A mild request cannot displace opposite output that is still
            # helping its own zone. Only intervene for actual adverse output.
            unused = output and ((need != mode and (adverse or (peer_risk and adverse_peer)))
                                 or (room == gentle["unit"] and peer_risk))
        else:
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
        gentle_step = (proposed is None and gentle is not None and room == gentle["unit"] and mode == need
                       and gentle_reason is None and ref in demanded and not unused)
        if gentle_step:
            if bounds := setpoint_bounds(limits[mode]):
                if bounds[0] <= setting <= bounds[1]:
                    proposed = automatic_setpoint(setting, direction_sign*min(1,policy["gentle_step"]), limits[mode])
            reason = "理想带外持续偏差，温和调温" if proposed is not None else "温和调温已达本机软限"
        elif gentle and room == gentle["unit"] and gentle_reason is not None and proposed is None and not unused:
            reason = "温和恢复暂不加码：" + gentle_reason
        reasons[room] = reason
        if proposed is None:
            if unused and output and own_feedback:
                reasons[room] = "已到软边界且仍有不需要的负荷，退出输出"
                candidates.append((1, (room != "study", 0), room, "mode", "off", "energy_off"))
            continue
        if increases_output(mode, setting, proposed):
            if (mode != need or opposite or ref not in demanded
                    or (gentle and (room != gentle["unit"] or peer_risk or not gentle_step))
                    or (not gentle and lead != room and lead is not None and attachment(lead) == ref)
                    or not observation_ready(now, last_house, live, zone_rooms, policy)):
                reasons[room] = "等待全屋同向需求、主机或动作后新反馈"
                continue
            if peer_guard:
                proposed = min(proposed, setting+policy["ordinary_step"]) if mode=="heat" else max(proposed, setting-policy["ordinary_step"])
                reasons[room] = "另一测点接近不利理想边界，仅本区小步加码"
            candidates.append((4, (0, -abs(delta)), room, "temperature", proposed,
                               "gentle_temperature" if gentle_step else "temperature"))
        elif own_feedback:
            purpose = ("direct_high_easing" if direct else "trend_easing" if reason == "预计越过理想带，提前减载"
                       else "coordinated_easing" if unused else "temperature")
            candidates.append((-2 if reason == "归位模式软限" else 0 if unused or direct else 2, (room != "study", -abs(delta)), room,
                               "temperature", proposed, purpose))
        else:
            reasons[room] = "等待本机动作后新室温再减载"
    return need, reasons, candidates
