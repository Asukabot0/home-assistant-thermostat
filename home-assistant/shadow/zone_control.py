"""Estimate independent zones, track their targets and allocate safe equipment actions."""
from statistics import median

from house_policy import ACTIVE, POLICY, TEMP_EPSILON
from model import automatic_setpoint, increases_output, numeric, setpoint_bounds
from pilot_gate import credible_output, current_power_evidence
from topology import fingerprint, sensor_reading, zone_target
from service import metered_interval


DIRECTION = {"heat": 1, "cool": -1}


def trend(points, after, now, policy):
    recent = [p for p in points if p[0] > after and p[0] >= now-policy["sparse_trend_window_seconds"]]
    slopes = [(v2-v1)/(t2-t1) for i, (t1, v1) in enumerate(recent) for t2, v2 in recent[i+1:]
              if t2-t1 >= policy["minimum_trend_span_seconds"]]
    if not slopes or now-recent[-1][0] > policy["sparse_report_age_seconds"]:
        return None
    span = recent[-1][0]-recent[0][0]
    if len(recent) == 2 and span < policy["tracking_report_span_seconds"]:
        return None
    rate = median(slopes) * policy["forecast_seconds"]
    dispersion = (max(slopes)-min(slopes))*policy["forecast_seconds"]/2
    return {"delta": rate, "uncertainty": max(0.10*policy["forecast_seconds"]/span, dispersion),
            "from": recent[0][0], "to": recent[-1][0], "reports": len(recent)}


def add_points(row, observation, identity, now, policy):
    if row.get("identity") != identity:
        row.clear()
        row.update(identity=identity, changed_at=now, points=[])
    points = row["points"]
    at = observation["at"]
    if points and now-points[-1][0] > policy["tracking_gap_seconds"]:
        points.clear()
    if (observation["quality"] == "fresh" and (not points or at > points[-1][0])):
        points.append([at, observation["value"]])
    points[:] = [p for p in points if p[0] >= now-policy["tracking_history_seconds"]]
    return points


def track(row, observation, target, estimate, paused, now, policy):
    measured = observation["value"]
    result = {"measured": measured, "target": target, "at": observation["at"], "quality": observation["quality"],
              "trend": estimate, "paused": paused, "intent": "limited", "reason": "data_missing",
              "direction": None, "duration": 0, "report_span": 0, "reports": 0, "qualified": False}
    if target is None or observation["quality"] != "fresh":
        return result
    error = target-measured
    band = policy["tracking_band"]
    result["error"] = error
    if abs(error) <= band+TEMP_EPSILON:
        result.update(intent="maintain" if not paused else "limited", reason="near_target" if not paused else "zone_paused")
        return result
    mode = "heat" if error > 0 else "cool"
    sign = DIRECTION[mode]
    outside = []
    for at, value in reversed(row["points"]):
        if (at < row["changed_at"] or sign*(target-value) <= band+TEMP_EPSILON
                or outside and outside[-1]-at > policy["tracking_gap_seconds"]):
            break
        outside.append(at)
    span = outside[0]-outside[-1] if outside else 0
    duration = now-outside[-1] if outside else 0
    enough = (len(outside) >= 2 and duration >= policy["tracking_dwell_seconds"]
              and span >= policy["tracking_report_span_seconds"])
    returning = (estimate is not None and sign*estimate["delta"]-estimate["uncertainty"]
                 >= max(policy["minimum_progress_c"], (abs(error)-band)*policy["forecast_seconds"]/policy["recovery_horizon_seconds"]))
    result.update(direction=mode, duration=duration, report_span=span, reports=len(outside),
                  evidence_from=outside[-1] if outside else None, qualified=enough and not returning and not paused,
                  intent="limited" if paused else "recover" if enough and not returning else "observe",
                  reason="zone_paused" if paused else "returning" if returning else "confirmed_bias" if enough else "waiting_reports_or_dwell")
    return result


def setting_effect(actions, context, now, observations, topology, state, policy, readings):
    """Bind observed outcomes to a serialized action, target and observation definition."""
    seen = state.setdefault("effects", {})
    cutoff = now-86400
    seen = state["effects"] = {k: v for k, v in seen.items() if v["ended_at"] >= cutoff}
    for i, action in enumerate(actions):
        scope = action.get("control")
        at = action["proxy_at"]
        age = now-at
        key = str(action["sent_at"])
        if (scope is None or key in seen or age < policy["effect_sample_wait_seconds"]
                or age > policy["effect_max_seconds"] or scope["definition"] != context):
            continue
        zone, mode = scope["zone"], scope["mode"]
        after = observations.get(zone)
        if (after is None or after["paused"] or after["quality"] != "fresh" or after["at"] <= at
                or after["target"] != scope["target"] or after["source_version"] != scope["source_version"]):
            continue
        if any(zone in topology["actuators"][other["room"]]["influences"] for other in actions[i+1:] if other["room"] in topology["actuators"]):
            continue  # Effects with interacting actions stay unassigned.
        direction = DIRECTION.get(mode)
        if direction:
            improvement = abs(scope["before_c"]-after["target"]) - abs(after["measured"]-after["target"])
            row = readings.get(action["room"], {})
            recording = row.get("power_evidence") or {}
            kwh, covered = metered_interval(row.get("live_power_reports", ()), at, after["at"]) if (
                recording.get("context") is not None and recording.get("context") == scope.get("power_context")) else (0, 0)
            seen[key] = {"unit": action["room"], "zone": zone, "mode": mode, "target": after["target"],
                         "ended_at": after["at"], "start_c": scope["before_c"], "end_c": after["measured"],
                         "useful": improvement >= policy["minimum_progress_c"], "improvement": improvement,
                         "thermal_delta": direction*(after["measured"]-scope["before_c"]),
                         "command_at": action["sent_at"], "proxy_at": at, "kind": action["kind"],
                         "watt_hours": 1000*kwh, "covered_seconds": covered, "interval_seconds": after["at"]-at,
                         "weather_c": scope.get("weather_c"), "source_version": after["source_version"]}


def observations(topology, live, inputs, state, actions, paused, now, policy=POLICY, *, readings=None):
    by_sensor = {key: sensor_reading(s, live, inputs, now) for key, s in topology["sensors"].items()
                 if s.get("enabled", True)}
    context = topology["fingerprint"]
    if state.get("definition") != context:
        state.clear()
        state["definition"] = context
    trackers = state.setdefault("trackers", {})
    out = {}
    for key, zone in topology["zones"].items():
        primary = by_sensor[zone["primary"]]
        ref = dict(primary)
        sensors = {k: by_sensor[k] for k in zone.get("sensors", ()) if k in by_sensor}
        aux = {k: s for k, s in sensors.items() if topology["sensors"][k]["role"] == "auxiliary" and s["quality"] == "fresh"}
        if primary["quality"] == "fresh" and zone.get("fusion") == "weighted" and aux:
            weighted = [(primary["value"], 1), *((s["value"], topology["sensors"][k].get("weight", 1)) for k, s in aux.items())]
            ref["value"] = sum(value*w for value, w in weighted)/sum(w for _, w in weighted)
            # The primary clock drives control. Aux-only changes are retained as
            # inputs but do not create an independent zone temperature report.
            ref["source"] = [primary["source"], {k: s["source"] for k, s in aux.items()}]
        target = zone_target(zone, live, inputs)
        switched = inputs.get(zone.get("pause_entity"), {}).get("state")
        is_paused = key in paused or bool(zone.get("pause_entity") and switched != "off")
        identity = fingerprint([context, ref["source"], target, is_paused])
        memory = trackers.setdefault(key, {})
        points = add_points(memory, ref, identity, now, policy)
        changed = max((a["proxy_at"] for a in actions if key in topology["actuators"][a["room"]]["influences"]), default=0)
        forecast = trend(points, max(changed, memory["changed_at"]), now, policy) if ref["quality"] == "fresh" else None
        result = track(memory, ref, target, forecast, is_paused, now, policy)
        result.update(source_version=identity, source=ref["source"], primary=zone["primary"],
                      primary_reading=primary, inputs=sensors,
                      constraints={k: s for k, s in sensors.items() if topology["sensors"][k]["role"] == "constraint"})
        out[key] = result
    setting_effect(actions, context, now, out, topology, state, policy, readings or {})
    environment = {k: v for k, v in by_sensor.items() if topology["sensors"][k]["role"] == "environment"}
    return out, environment


def learned_initial(room, mode, zone, state, limits, policy=POLICY):
    bounds = setpoint_bounds(limits)
    if bounds is None or zone["target"] is None:
        return None
    learned = state.get("maintenance", {}).get(room, {}).get(mode)
    chosen = round(zone["target"])
    if learned and learned["source_version"] == zone["source_version"] and learned["target"] == zone["target"]:
        chosen = max(chosen-policy["ordinary_step"], min(chosen+policy["ordinary_step"], learned["setpoint"]))
    return max(bounds[0], min(bounds[1], chosen))


def peer_risk(room, zone_key, mode, zones, topology, policy):
    direction = DIRECTION[mode]
    unit = topology["actuators"][room]
    limits = []
    for peer, influence in unit["influences"].items():
        zone = zones[peer]
        checks = []
        if peer != zone_key:
            checks.append((peer, zone["measured"], zone["quality"], zone["trend"]))
        checks.extend((key, sensor["value"], sensor["quality"], None) for key, sensor in zone["constraints"].items())
        for key, measured, quality, predicted in checks:
            if quality != "fresh" or zone["target"] is None:
                limits.append({"point": key, "reason": "peer_observation_missing", "influence": influence})
                continue
            side = direction*(measured-zone["target"])
            projected = side + max(0, direction*predicted["delta"]+predicted["uncertainty"]) if predicted else side
            reason = ("peer_comfort_risk" if max(side, projected) > policy["comfort_band"]+TEMP_EPSILON else
                      "peer_near_boundary" if max(side, projected) >= policy["ideal_band"]-policy["peer_ideal_guard_c"] else None)
            if reason:
                limits.append({"point": key, "reason": reason, "influence": influence,
                               "deviation": side, "projected_adverse": projected})
    return limits


def opposing_allowed(room, zone_key, peer, modes, zones, topology, output):
    settings = topology.get("opposite_operation", {})
    if not settings.get("enabled", False) or not output.get(peer):
        return False
    units = topology["actuators"]
    peer_zone = units[peer]["zone"]
    ceiling = settings.get("max_cross_influence", 0.1)
    return (peer_zone != zone_key and zones[peer_zone]["qualified"] and zones[peer_zone]["direction"] == modes[peer]
            and topology["links"].get(tuple(sorted((peer_zone, zone_key))), 0) <= ceiling
            and units[room]["influences"].get(peer_zone, 0) <= ceiling
            and units[peer]["influences"].get(zone_key, 0) <= ceiling)


def response_ready(now, zone, action, wait):
    return action is None or (now-action["proxy_at"] >= wait and zone["at"] is not None and zone["at"] > action["proxy_at"])


def is_strengthening(action):
    return (action["value"] in ACTIVE if action["kind"] == "mode" else
            increases_output(action["before"][0], action["before"][1], action["value"]))


def allocation_scope(room, zone, zones, environment, topology):
    return {"unit": room, "zone": zone, "target": zones[zone]["target"], "source_version": zones[zone]["source_version"],
            "definition": topology["fingerprint"], "at": zones[zone]["at"], "before_c": zones[zone]["measured"],
            "weather_c": next((v["value"] for v in environment.values() if v["quality"] == "fresh"), None)}


def maintain_setting(now, room, mode, setting, zone, local, readings, other_output, policy):
    if other_output or zone["intent"] != "maintain" or zone["trend"] is None:
        return
    points = local["trackers"][zone["key"]]["points"]
    proof = readings.get("power_evidence") or {}
    last = proof.get("last_report")
    under_load = proof.get("bands", {}).get("above_low", {}).get("seconds", 0)
    start = max(readings.get("operation_since", 0), now-policy["maintain_confirmation_seconds"]-600)
    recent = [(t, v) for t, v in points if t > start and t >= now-policy["maintain_confirmation_seconds"]-600]
    if (len(recent) >= 2 and recent[-1][0]-recent[0][0] >= policy["maintain_confirmation_seconds"]
            and under_load >= policy["maintain_confirmation_seconds"] and last is not None
            and all(abs(v-zone["target"]) <= policy["tracking_band"]+TEMP_EPSILON for _, v in recent)
            and abs(zone["trend"]["delta"]) <= policy["minimum_progress_c"]):
        local.setdefault("maintenance", {}).setdefault(room, {})[mode] = {
            "setpoint": setting, "target": zone["target"], "source_version": zone["source_version"],
            "at": recent[-1][0], "observed_seconds": recent[-1][0]-recent[0][0]}


def comparison_rank(options, zone, mode, environment, local, policy):
    outdoor = next((v["value"] for v in environment.values() if v["quality"] == "fresh"), None)
    scores = {}
    for room in options:
        samples = [e for e in local.get("effects", {}).values() if e["unit"] == room and e["mode"] == mode and e["useful"]
                   and e["source_version"] == zone["source_version"] and e["target"] == zone["target"]
                   and abs(e["start_c"]-zone["measured"]) <= 0.4 and outdoor is not None
                   and e["weather_c"] is not None and abs(e["weather_c"]-outdoor) <= 3
                   and e["covered_seconds"] >= 0.8*e["interval_seconds"] and e["covered_seconds"] > 0]
        if len(samples) >= policy["effect_samples_min"]:
            scores[room] = median(e["watt_hours"]/e["improvement"] for e in samples)
    return scores if len(scores) >= 2 else {}


def coordinate(live, readings, ready, state, now, configurations, connected, topology, inputs, paused, trace,
               policy=POLICY, *, control_ready=None):
    local = state.setdefault("closed_loop", {})
    units = topology["actuators"]
    actions = [a for a in state["commands"] if a["status"] == "proxy_echo" and a["room"] in units]
    zones, environment = observations(topology, live, inputs, local, actions, paused, now, policy, readings=readings)
    for key, zone in zones.items():
        zone["key"] = key
    modes = {room: live[room]["adapter"]["mode"] for room in connected}
    output = {room: mode in ACTIVE and room in readings and credible_output(now, live[room], readings[room])
              for room, mode in modes.items()}
    duration = policy["peer_fairness_wait_seconds"]
    candidates, reasons, scopes = [], {}, {}
    demands = sorted((key for key, z in zones.items() if z["qualified"]), key=lambda key: (
        -int(abs(zones[key]["error"]) > policy["comfort_band"]+TEMP_EPSILON),
        -(abs(zones[key]["error"]) + min(zones[key]["duration"]/duration, 4)*0.2), key))
    # Check controllability independently from zone temperature. Foreign-zone
    # equipment can act only when its physical area is observed and protected.
    uncertain = {zone for a in state["commands"] if a["status"] == "unconfirmed" and a["room"] in units
                 and units[a["room"]]["lock_group"] in state["holds"] for zone in units[a["room"]]["influences"]}
    control_ready = ready if control_ready is None else control_ready
    feasible = {key: [r for r in sorted(connected) if control_ready.get(r) and key in units[r]["influences"] and key not in uncertain
                       and zones[units[r]["zone"]]["quality"] == "fresh" and not zones[units[r]["zone"]]["paused"]
                       and zones[key]["direction"] in live[r]["adapter"]["modes"]
                       and not any(x["reason"] != "peer_near_boundary" for x in peer_risk(r, key, zones[key]["direction"], zones, topology, policy))]
                for key in demands}
    owner = next((key for key in demands if feasible[key]), None)
    need = zones[owner]["direction"] if owner is not None else "hold" if all(z["quality"] == "fresh" for z in zones.values()) else "unknown"
    contender = local.get("owner")
    if owner and contender in demands and feasible[contender] and zones[contender]["direction"] != need:
        head, previous = zones[owner], zones[contender]
        if (abs(head["error"]) <= policy["comfort_band"]+TEMP_EPSILON
                and now-local.get("selected_at", 0) < duration and abs(head["error"]) < abs(previous["error"])+.3):
            owner, need = contender, previous["direction"]
    energy_scores = comparison_rank(feasible[owner], zones[owner], need, environment, local, policy) if owner else {}
    if owner != contender:
        local.update(owner=owner, selected_at=now)
    trace.update(topology=topology["observation_version"], definition=topology["fingerprint"], zones=zones,
                 environment=environment, owner=owner, demand_order=demands, feasible=feasible, output_evidence=output,
                 risks={}, equipment={}, observations_after={}, action_scopes=scopes, energy_scores=energy_scores,
                 recent_effects=list(local.get("effects", {}).values())[-8:], uncertain_zones=sorted(uncertain))
    for key in demands:
        if key != owner:
            zones[key].update(intent="limited", reason="unconfirmed_influencing_equipment" if key in uncertain else
                              "conflicting_priority" if feasible[key] else "no_safe_equipment")
    for room in sorted(connected):
        unit = units[room]
        home = unit["zone"]
        serve = owner if owner in unit["influences"] else home
        local_zone, zone = zones[home], zones[serve]
        mode, setting = modes[room], numeric(live[room]["adapter"]["setpoint"])
        if not ready.get(room) or local_zone["paused"] or local_zone["quality"] != "fresh" or local_zone["target"] is None:
            trace["equipment"][room] = "equipment_not_controllable" if not ready.get(room) else "local_zone_unavailable_or_paused"
            continue
        row = readings[room]
        scopes[room] = allocation_scope(room, serve, zones, environment, topology)
        scopes[room]["mode"] = need if mode == "off" else mode
        power = row.get("power_evidence") or {}
        scopes[room]["power_context"] = power.get("context")
        trace["equipment"][room] = {"power_context": power.get("context"), "power_at": power.get("last_report"),
                                    "bands": power.get("bands"), "served_zone": serve}
        own = next((a for a in reversed(actions) if a["room"] == room), None)
        last_up = next((a for a in reversed(actions) if serve in units[a["room"]]["influences"] and
                        ((a.get("control") or {}).get("zone", units[a["room"]]["zone"]) == serve or a["room"] == room)
                        and is_strengthening(a)), None)
        trace["observations_after"][room] = {"zone_report": zone["at"], "last_increase": last_up["proxy_at"] if last_up else None,
                                              "own_command": own["proxy_at"] if own else None}
        risks = peer_risk(room, serve, need if need in ACTIVE else mode, zones, topology, policy) if mode in ACTIVE or need in ACTIVE else []
        trace["risks"][room] = risks
        active_risk = peer_risk(room, home, mode, zones, topology, policy) if mode in ACTIVE else []
        red_risk = any(r["reason"] == "peer_comfort_risk" for r in active_risk)
        opposite = {peer for peer, m in modes.items() if peer != room and m not in {"off", "fan_only", need}
                    and not opposing_allowed(room, serve, peer, modes, zones, topology, output)}
        own_wait = response_ready(now, zone, own, policy["feedback_wait_seconds"])
        up_wait = response_ready(now, zone, last_up, policy["feedback_wait_seconds"])
        can_strengthen = (owner == serve and room in feasible.get(serve, ()) and not opposite and own_wait and up_wait)
        if last_up is not None and not up_wait and owner == serve:
            zone["intent"] = "observe"
        priority = unit.get("priority", 0)
        rank = (int(home != serve), round(energy_scores.get(room, float("inf")), 4),
                priority, -unit["influences"][serve])
        if mode == "off":
            live_same = [r for r in feasible.get(serve, ()) if modes[r] == need and r != room]
            if live_same:
                # A local primary that is responding gets a full measured window.
                # A saturated, inactive, or ineffective unit need not hold allocation forever.
                def capable(peer):
                    setting = numeric(live[peer]["adapter"]["setpoint"])
                    if setting is None:
                        return False
                    space = automatic_setpoint(setting, DIRECTION[need], configurations[peer]["soft_limits"][need])
                    last_effect = next((v for v in reversed(list(local.get("effects", {}).values()))
                                        if v["unit"] == peer and v["zone"] == serve and v["target"] == zone["target"]), None)
                    return space is not None and (last_effect is None or last_effect["useful"])
                if any(capable(peer) for peer in live_same):
                    reasons[room] = "maintain_existing_zone_assignment"
                    continue
            if can_strengthen and room in state["parked"] and need in live[room]["adapter"]["modes"]:
                candidates.append((3, rank, room, "mode", need, "energy_start"))
                reasons[room] = "start_for_confirmed_zone_error"
            else:
                reasons[room] = "wait_opposite_or_observation" if opposite or not own_wait or not up_wait else "no_admissible_start"
            continue
        if mode not in ACTIVE or setting is None:
            continue
        bounds = setpoint_bounds(configurations[room]["soft_limits"][mode])
        if bounds is None:
            continue
        direction = DIRECTION[mode]
        current_excess = direction*(local_zone["measured"]-local_zone["target"])
        local_prediction = local_zone["trend"]
        expected = (current_excess+direction*local_prediction["delta"]-local_prediction["uncertainty"]
                    if local_prediction else current_excess)
        high_band = power.get("bands", {}).get("high", {})
        direct = (output[room] and high_band.get("seconds", 0) >= configurations[room].get("overshoot_high_power_seconds", 180)
                  and current_excess >= policy["ideal_band"]-TEMP_EPSILON
                  and (local_prediction is None or direction*local_prediction["delta"] > -policy["minimum_progress_c"]))
        blocks_needed = owner is not None and need != mode and not any(
            z["qualified"] and z["direction"] == mode and name in unit["influences"] for name,z in zones.items())
        redundant = (owner is None and priority > 0 and local_zone["intent"] == "maintain" and local_prediction
                     and abs(local_prediction["delta"])+local_prediction["uncertainty"] <= policy["minimum_progress_c"]
                     and any(output[p] and p != room and modes[p] == mode and units[p]["zone"] == home for p in output))
        reducing = direct or red_risk or expected > policy["tracking_band"]+TEMP_EPSILON or blocks_needed or redundant
        low = power.get("bands", {}).get("low", {})
        quiet = current_power_evidence(now, live[room], row) and low.get("seconds", 0) >= policy["lower_output_observe_seconds"]
        eased = own is not None and own["kind"] == "temperature" and not is_strengthening(own)
        return_after = response_ready(now, local_zone, own, policy["reduction_wait_seconds"])
        if own and quiet and DIRECTION.get(mode) and low.get("since") is not None and low["since"] > own["proxy_at"]:
            return_after = return_after or now-own["proxy_at"] >= policy["reduction_wait_seconds"]
        if reducing:
            if local_zone["intent"] == "maintain":
                local_zone["intent"] = "weaken"
            if not return_after:
                reasons[room] = "wait_new_local_feedback_after_reduction"
                continue
            proposed = (bounds[1] if mode == "heat" and setting > bounds[1] else bounds[0]
                        if mode == "cool" and setting < bounds[0] else
                        automatic_setpoint(setting, -direction*policy["ordinary_step"], configurations[room]["soft_limits"][mode]))
            if ((blocks_needed and quiet) or (output[room] and (eased or proposed is None) and
                                             (blocks_needed or red_risk or redundant or current_excess > policy["ideal_band"]))):
                candidates.append((0, rank, room, "mode", "off", "energy_off"))
                reasons[room] = "exit_dormant_opposition" if quiet else "exit_contrary_or_risky_output"
            elif output[room] and proposed is not None:
                candidates.append((1, rank, room, "temperature", proposed, "zone_reduce"))
                reasons[room] = "high_power_overrun" if direct else "reduce_projected_or_neighbor_overrun"
            else:
                reasons[room] = "wait_verified_output_or_off_proof"
            continue
        if mode == need and can_strengthen:
            magnitude = policy["ordinary_step"]
            pred = zone["trend"]
            if (not risks and home == serve and abs(zone["error"]) >= policy["large_deviation"] and pred
                    and direction*pred["delta"] <= 0):
                magnitude = policy["large_step"]
            proposed = automatic_setpoint(setting, direction*magnitude, configurations[room]["soft_limits"][mode])
            if proposed is not None and bounds[0] <= setting <= bounds[1]:
                candidates.append((4, rank, room, "temperature", proposed, "zone_strengthen"))
                reasons[room] = "bounded_zone_correction"
                continue
        reasons[room] = "observe_local_response" if not own_wait or not up_wait else "hold_effective_setting"
        row = dict(row, operation_since=max(live[room]["adapter"]["last_mode_change"], own["proxy_at"] if own else 0))
        maintain_setting(now, room, mode, setting, local_zone, local, row,
                         any(output[other] and other != room and home in units[other]["influences"] for other in output), policy)
    trace["unmet"] = {key: zones[key]["reason"] for key in demands if key != owner or not candidates}
    trace["need"] = need
    trace["normal_need"] = need
    trace["gentle"] = {"state": "replaced_by_zone_tracking"}
    state["house_lead"] = min(candidates)[2] if candidates else next((r for r in sorted(output) if output[r]), None)
    return need, reasons, candidates
