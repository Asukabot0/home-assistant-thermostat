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
    endpoint = at if observation["quality"] == "fresh" and points and at > points[-1][0] else now
    if points and endpoint-points[-1][0] > policy["tracking_gap_seconds"]:
        points.clear()
        row.pop("recovery", None)
        row["changed_at"] = endpoint
    if observation["quality"] == "fresh" and (not points or at > points[-1][0]):
        points.append([at, observation["value"]])
    points[:] = [p for p in points if p[0] >= now-policy["tracking_history_seconds"]]
    return points


def track(row, observation, target, estimate, paused, now, policy):
    measured = observation["value"]
    episode = row.get("recovery") or {}
    result = {"measured": measured, "target": target, "at": observation["at"], "quality": observation["quality"],
              "trend": estimate, "paused": paused, "intent": "limited", "reason": "data_missing",
              "direction": episode.get("direction"), "duration": 0, "report_span": 0, "reports": 0,
              "qualified": False, "recovery_active": bool(episode), "settling_reports": 0, "settling_span": 0}
    if target is None or observation["quality"] != "fresh":
        return result
    error = target-measured
    band = policy["tracking_band"]
    result["error"] = error
    if abs(error) <= band+TEMP_EPSILON:
        if episode:
            inside = []
            for at, value in reversed(row["points"]):
                if at <= episode["last_outside"] or abs(value-target) > band+TEMP_EPSILON:
                    break
                inside.append(at)
            span = inside[0]-inside[-1] if inside else 0
            result.update(settling_reports=len(inside), settling_span=span, duration=now-episode["since"],
                          evidence_from=episode["since"], intent="observe" if not paused else "limited",
                          reason="verify_stable_recovery" if not paused else "zone_paused")
            projected = measured+(estimate["delta"] if estimate else 0)
            moving_out = (estimate is not None and abs(estimate["delta"])-estimate["uncertainty"] >= policy["minimum_progress_c"]
                          and abs(projected-target) > band+TEMP_EPSILON)
            if len(inside) < 2 or span < policy["recovery_settle_seconds"] or moving_out:
                return result
            row.pop("recovery", None)
            result.update(direction=None, recovery_active=False, recovery_completed_at=inside[0])
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
    if episode.get("direction") != mode:
        row.pop("recovery", None)
        episode = {}
    if enough and not episode and not paused:
        episode = {"direction":mode, "since":outside[-1], "report_span":span, "reports":len(outside),
                   "last_outside":outside[0]}
        row["recovery"] = episode
    if episode and outside:
        episode["last_outside"] = max(episode["last_outside"], outside[0])
    if episode:
        duration = now-episode["since"]
    returning = (estimate is not None and sign*estimate["delta"]-estimate["uncertainty"]
                 >= max(policy["minimum_progress_c"], (abs(error)-band)*policy["forecast_seconds"]/policy["recovery_horizon_seconds"]))
    result.update(direction=mode, duration=duration, report_span=span, reports=len(outside),
                  evidence_from=episode["since"] if episode else outside[-1] if outside else None,
                  recovery_active=bool(episode), recovery_evidence={k:episode[k] for k in ("report_span", "reports")} if episode else None,
                  qualified=bool(episode and outside and not returning and not paused),
                  intent="limited" if paused else "recover" if episode and not returning else "observe",
                  reason="zone_paused" if paused else "returning" if returning else "confirmed_bias" if episode else "waiting_reports_or_dwell")
    return result


def setting_effect(actions, now, observations, topology, state, policy, readings):
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
                or age > policy["effect_max_seconds"] or scope["definition"] != topology["action_versions"][action["room"]]
                or scope.get("learning_version") != topology["learning_versions"][action["room"]]):
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
                         "weather_c": scope.get("weather_c"), "source_version": after["source_version"],
                         "learning_version": topology["learning_versions"][action["room"]]}


def observations(topology, live, inputs, state, actions, paused, now, policy=POLICY, *, readings=None):
    by_sensor = {key: sensor_reading(s, live, inputs, now) for key, s in topology["sensors"].items()
                 if s.get("enabled", True)}
    previous = state.get("definition")
    state["definition"] = topology["fingerprint"]
    trackers = state.setdefault("trackers", {})
    for removed in trackers.keys()-topology["zones"].keys():
        del trackers[removed]
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
        definition = topology["zone_versions"][key]
        identity = fingerprint([definition, ref["source"], target, is_paused])
        memory = trackers.setdefault(key, {})
        if memory and "zone_version" not in memory and previous == topology["fingerprint"]:
            legacy = fingerprint([previous, ref["source"], target, is_paused])
            if memory["identity"] == legacy:
                memory["identity"] = identity
                for room, modes in state.get("maintenance", {}).items():
                    if topology["actuators"].get(room, {}).get("zone") == key:
                        for entry in modes.values():
                            if entry["source_version"] == legacy:
                                entry.update(source_version=identity, learning_version=topology["learning_versions"][room])
                for entry in state.get("effects", {}).values():
                    if entry["zone"] == key and entry["source_version"] == legacy and entry["unit"] in topology["actuators"]:
                        entry.update(source_version=identity, learning_version=topology["learning_versions"][entry["unit"]])
        points = add_points(memory, ref, identity, now, policy)
        memory["zone_version"] = definition
        changed = max((a["proxy_at"] for a in actions if key in topology["actuators"][a["room"]]["influences"]), default=0)
        forecast = trend(points, max(changed, memory["changed_at"]), now, policy) if ref["quality"] == "fresh" else None
        result = track(memory, ref, target, forecast, is_paused, now, policy)
        result.update(source_version=identity, observation_definition=definition, source=ref["source"], primary=zone["primary"],
                      primary_reading=primary, inputs=sensors,
                      constraints={k: s for k, s in sensors.items() if topology["sensors"][k]["role"] == "constraint"})
        out[key] = result
    for room, modes in list(state.get("maintenance", {}).items()):
        home = topology["actuators"].get(room, {}).get("zone")
        state["maintenance"][room] = {mode:entry for mode,entry in modes.items() if home in out
            and entry.get("learning_version") == topology["learning_versions"][room]
            and entry["source_version"] == out[home]["source_version"]}
    if "effects" in state:
        state["effects"] = {key: entry for key,entry in state["effects"].items() if entry["zone"] in out
            and entry["source_version"] == out[entry["zone"]]["source_version"]
            and entry.get("learning_version") == topology["learning_versions"].get(entry["unit"])}
    setting_effect(actions, now, out, topology, state, policy, readings or {})
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


def allows_assistance(home_zone, mode):
    return (home_zone["quality"] == "fresh" and not home_zone["paused"]
            and home_zone["qualified"] and home_zone["direction"] == mode)


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
            "definition": topology["action_versions"][room], "learning_version": topology["learning_versions"][room],
            "at": zones[zone]["at"], "before_c": zones[zone]["measured"],
            "weather_c": next((v["value"] for v in environment.values() if v["quality"] == "fresh"), None)}


def maintain_setting(now, room, mode, setting, zone, local, readings, other_output, policy, learning_version):
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
            "at": recent[-1][0], "observed_seconds": recent[-1][0]-recent[0][0], "learning_version":learning_version}


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


def quiet_power(now, live, shadow, duration, *, after=None):
    proof = shadow.get("power_evidence") or {}
    low = proof.get("bands", {}).get("low", {})
    watts = numeric(live["adapter"]["power"])
    cutoff = shadow.get("reversal_low_cutoff")
    if (not current_power_evidence(now, live, shadow) or watts is None or cutoff is None or watts >= cutoff
            or (since := low.get("since")) is None):
        return False
    return proof["last_report"]-max(since, after or since) >= duration


def unpowered(now, peer, need, zone, live, readings, actions, policy):
    """A unit that stayed at off-level power since it was last asked to run does not hold a zone assignment."""
    shadow = readings.get(peer) or {}
    setting = numeric(live[peer]["adapter"]["setpoint"])
    asked = max([live[peer]["adapter"]["last_mode_change"],
                 *(a["proxy_at"] for a in actions if a["room"] == peer and is_strengthening(a))])
    grace = policy.get("assignment_power_grace_seconds", 240)
    since = (shadow.get("power_evidence") or {}).get("bands", {}).get("low", {}).get("since")
    # Past the window a quiet unit is strengthened first, which starts a new window.
    return (setting is not None and zone["measured"] is not None
            and DIRECTION[need]*(setting-zone["measured"]) > policy["tracking_band"]+TEMP_EPSILON
            and grace <= now-asked <= grace+policy["feedback_wait_seconds"]
            and since is not None and since <= asked
            and quiet_power(now, live[peer], shadow, grace, after=asked))


def exit_wait(room, direction, now, live, readings, state, policy):
    """An opposite stop echo alone does not free the next mode to strengthen."""
    return any(peer != room and peer in live and live[peer]["adapter"]["mode"] == "off"
               and parked.get("handoff") and parked["mode"] != direction
               and ((live[room].get("temperature_report_at") or 0) <= parked["off_at"]
                    or (live[peer].get("temperature_report_at") or 0) <= parked["off_at"]
                    or not quiet_power(now,live[peer],readings.get(peer,{}),policy["lower_output_observe_seconds"],
                                       after=parked["off_at"]))
               for peer, parked in state["parked"].items())


def yielding_safe(zone, direction, policy):
    # Consider deterioration in the direction the yielding unit is currently
    # counteracting. A genuine opposite demand may wait only inside comfort.
    if zone["paused"] or zone["quality"] != "fresh" or zone["target"] is None:
        return False
    deficit = DIRECTION[direction]*(zone["target"]-zone["measured"])
    forecast = zone["trend"]
    risk = max(0, -DIRECTION[direction]*forecast["delta"]+forecast["uncertainty"]) if forecast else 0
    return deficit+risk < policy["comfort_band"]-TEMP_EPSILON


def conflict_options(demands, modes, live, readings, feasible, zones, output, control_ready, topology, configurations, parked, now, policy):
    options, blocked = {}, {}
    units = topology["actuators"]
    for zone in demands:
        direction = zones[zone]["direction"]
        blocked[zone] = "opposing_output_cannot_yield"
        for room in sorted(feasible[zone], key=lambda r:(units[r]["zone"] != zone, modes[r] != direction,
                                                        units[r].get("priority",0), r)):
            mode, setting = modes[room], numeric(live[room]["adapter"]["setpoint"])
            if not (mode == direction and setting is not None and automatic_setpoint(setting, DIRECTION[mode],
                        configurations[room]["soft_limits"][mode]) is not None or
                    mode == "off" and room in parked):
                continue
            blockers = [r for r, m in modes.items() if r != room and m not in {direction, "off", "fan_only"}
                        and not opposing_allowed(room, zone, r, modes, zones, topology, output)]
            if any(modes[r] not in ACTIVE or not control_ready.get(r) or r not in readings or
                   readings[r].get("reversal_low_cutoff") is None or
                   not current_power_evidence(now,live[r],readings[r]) or
                   not yielding_safe(zones[units[r]["zone"]],modes[r],policy) for r in blockers):
                continue
            options[zone] = (room, sorted(blockers))
            blocked.pop(zone)
            break
        if not feasible[zone]:
            blocked[zone] = "no_comfort_safe_assignment"
    return options, blocked


def contest_owner(demands, options, local, actions, zones, now, policy):
    served = {key: max((a["proxy_at"] for a in actions if a.get("control",{}).get("zone") == key
                        and is_strengthening(a) and a["proxy_at"] >= (zones[key]["evidence_from"] or 0)), default=0)
              for key in options}
    owner = local.get("owner")
    plan = local.get("handoff")
    if owner in options:
        elected_at = local.get("selected_at",0)
        turn = min((a["proxy_at"] for a in actions if a.get("control",{}).get("zone") == owner and
                    is_strengthening(a) and a["proxy_at"] >= elected_at),default=elected_at)
        if (plan and plan["owner"] == owner) or now-turn < policy["peer_fairness_wait_seconds"]:
            return owner
    if owner in options and len(options) > 1 and not served.get(owner):
        # The selected area already had a feasible opportunity. Yield the
        # next arbitration slot if it produced no successful increase.
        return next(key for key in demands if key in options and key != owner)
    return min(options, key=lambda k: (abs(zones[k]["error"]) <= policy["comfort_band"]+TEMP_EPSILON,
                                       max(zones[k]["evidence_from"] or 0, served[k]), demands.index(k)), default=None)


def handoff_step(owner, choice, zones, modes, live, readings, actions, ready, topology, local, now, policy):
    plan = local.get("handoff")
    room = choice[0] if choice else None
    blockers = choice[1] if choice else []
    direction = zones[owner]["direction"] if owner is not None else None
    # A plan keys off the actual observation and equipment dependencies.
    binding = topology["action_versions"][room] if room is not None else None
    if plan and (plan["owner"] != owner or plan["receiver"] != room or plan["direction"] != direction or
                 plan["definition"] != binding or plan["source_version"] != zones[owner]["source_version"] or
                 plan["blocker"] not in modes or plan["blocker_definition"] != topology["action_versions"][plan["blocker"]] or
                 plan["blocker_version"] != zones[topology["actuators"][plan["blocker"]]["zone"]]["source_version"]):
        plan = None
        local.pop("handoff",None)
    if plan is None and blockers:
        blocker = blockers[0]
        plan = {"owner":owner,"receiver":room,"blocker":blocker,"direction":direction,"selected_at":now,
                "definition":binding,"source_version":zones[owner]["source_version"],
                "blocker_definition":topology["action_versions"][blocker],
                "blocker_version":zones[topology["actuators"][blocker]["zone"]]["source_version"]}
        local["handoff"] = plan
    if plan is None:
        return None
    blocker = plan["blocker"]
    off = next((a for a in reversed(actions) if a["room"] == blocker and a["purpose"] == "handoff_off" and
                a.get("control",{}).get("handoff_at") == plan["selected_at"]),None)
    reduce = next((a for a in reversed(actions) if a["room"] == blocker and a["purpose"] == "handoff_reduce" and
                   a.get("control",{}).get("handoff_at") == plan["selected_at"]),None)
    peer_zone = zones[topology["actuators"][blocker]["zone"]]
    if off:
        if (not quiet_power(now,live[blocker],readings.get(blocker,{}),policy["lower_output_observe_seconds"],after=off["proxy_at"]) or
                modes[blocker]!="off" or not all(response_ready(now,z,off,policy["reduction_wait_seconds"])
                                                 for z in (zones[owner],peer_zone))):
            return dict(plan,phase="verify_exit",reason="observe_both_sensors_and_low_power_after_off")
        local.pop("handoff",None)
        if blockers:
            return handoff_step(owner,choice,zones,modes,live,readings,actions,ready,topology,local,now,policy)
        return dict(plan,phase="serve",reason="exit_verified")
    if blocker not in blockers:
        local.pop("handoff",None)
        return None
    if not ready.get(blocker):
        return dict(plan,phase="limited",reason="yielding_equipment_locked")
    if reduce and not all(response_ready(now,z,reduce,policy["reduction_wait_seconds"]) for z in (zones[owner],peer_zone)):
        return dict(plan,phase="verify_reduction",reason="observe_both_sensors_after_reduction")
    if reduce or quiet_power(now,live[blocker],readings.get(blocker,{}),policy["lower_output_observe_seconds"]):
        return dict(plan,phase="exit",reason="reduced_or_proven_dormant_opposition")
    return dict(plan,phase="reduce",reason="yield_one_bounded_step")


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
                       and (units[r]["zone"] == key or allows_assistance(
                           zones[units[r]["zone"]], zones[key]["direction"]))
                       and zones[key]["direction"] in live[r]["adapter"]["modes"]
                       and not any(x["reason"] != "peer_near_boundary" for x in peer_risk(r, key, zones[key]["direction"], zones, topology, policy))]
                for key in demands}
    contest = len({zones[key]["direction"] for key in demands}) > 1
    blocked = {}
    contender = local.get("owner")
    if contest:
        options, blocked = conflict_options(demands,modes,live,readings,feasible,zones,output,control_ready,topology,
                                            configurations,state["parked"],now,policy)
        owner = contest_owner(demands,options,local,actions,zones,now,policy)
        handoff = handoff_step(owner,options.get(owner),zones,modes,live,readings,actions,ready,topology,local,now,policy)
        elected = options[owner][0] if owner is not None else None
    else:
        owner = next((key for key in demands if feasible[key]),None)
        # Once an off is in the journal its low-power gate remains in force,
        # even when the two-sided intent no longer qualifies for this plan.
        local.pop("handoff",None)
        handoff = elected = None
    need = zones[owner]["direction"] if owner is not None else "hold" if all(z["quality"] == "fresh" for z in zones.values()) else "unknown"
    energy_scores = comparison_rank(feasible[owner], zones[owner], need, environment, local, policy) if owner else {}
    if owner != contender:
        local.update(owner=owner, selected_at=now)
    trace.update(topology=topology["observation_version"], definition=topology["fingerprint"], zones=zones,
                 environment=environment, owner=owner, demand_order=demands, feasible=feasible, output_evidence=output,
                 risks={}, equipment={}, observations_after={}, action_scopes=scopes, energy_scores=energy_scores,
                 recent_effects=list(local.get("effects", {}).values())[-8:], uncertain_zones=sorted(uncertain),
                 handoff=handoff, conflict=contest, yielding_blocks=blocked, elected=elected)
    for key in demands:
        if key != owner:
            zones[key].update(intent="limited", reason="unconfirmed_influencing_equipment" if key in uncertain else
                              blocked.get(key, "conflicting_priority" if feasible[key] else "no_safe_equipment"))
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
        own_critical_need = (mode in ACTIVE and local_zone.get("error") is not None and
                             DIRECTION[mode]*local_zone["error"] >= policy["comfort_band"]-TEMP_EPSILON)
        red_risk = any(r["reason"] == "peer_comfort_risk" for r in active_risk) and not own_critical_need
        opposite = {peer for peer, m in modes.items() if peer != room and m not in {"off", "fan_only", need}
                    and not opposing_allowed(room, serve, peer, modes, zones, topology, output)}
        own_wait = response_ready(now, local_zone, own, policy["feedback_wait_seconds"])
        up_wait = response_ready(now, zone, last_up, policy["feedback_wait_seconds"])
        handoff_wait = (handoff is not None and handoff["phase"] != "serve"
                        or need in ACTIVE and exit_wait(room,need,now,live,readings,state,policy))
        can_strengthen = (owner == serve and room in feasible.get(serve, ()) and not opposite and own_wait and up_wait
                          and not handoff_wait and (elected is None or room == elected))
        if last_up is not None and not up_wait and owner == serve:
            zone["intent"] = "observe"
        priority = unit.get("priority", 0)
        rank = (int(home != serve), round(energy_scores.get(room, float("inf")), 4),
                priority, -unit["influences"][serve])
        if handoff and room == handoff["blocker"] and handoff["phase"] != "serve":
            reasons[room] = handoff["reason"]
            # Bind each candidate to the local observation of the yielding
            # equipment as well as the elected plan kept in the trace.
            scopes[room] = allocation_scope(room,home,zones,environment,topology)
            scopes[room].update(mode=mode,power_context=power.get("context"),handoff_at=handoff["selected_at"],
                                handoff_phase=handoff["phase"])
            if handoff["phase"] == "exit":
                candidates.append((0,rank,room,"mode","off","handoff_off"))
            elif handoff["phase"] == "reduce" and mode in ACTIVE and setting is not None and output.get(room):
                if response_ready(now,local_zone,own,policy["reduction_wait_seconds"]):
                    step = automatic_setpoint(setting,-DIRECTION[mode]*policy["ordinary_step"],configurations[room]["soft_limits"][mode])
                    if step is None:
                        handoff.update(phase="exit",reason="yield_at_safe_boundary")
                        scopes[room]["handoff_phase"] = "exit"
                        candidates.append((0,rank,room,"mode","off","handoff_off"))
                    else:
                        candidates.append((1,rank,room,"temperature",step,"handoff_reduce"))
                else:
                    reasons[room]="verify_previous_yielding_action"
            continue
        if mode == "off":
            live_same = [r for r in feasible.get(serve, ()) if modes[r] == need and r != room]
            idle = {peer for peer in live_same if unpowered(now, peer, need, zone, live, readings, actions, policy)}
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
                    return (space is not None and (last_effect is None or last_effect["useful"])
                            and peer not in idle)
                if any(capable(peer) for peer in live_same):
                    reasons[room] = "maintain_existing_zone_assignment"
                    continue
            if can_strengthen and room in state["parked"] and need in live[room]["adapter"]["modes"]:
                candidates.append((3, rank, room, "mode", need, "energy_start"))
                reasons[room] = "start_beside_unpowered_unit" if idle else "start_for_confirmed_zone_error"
            else:
                reasons[room] = "wait_opposite_or_observation" if opposite or handoff_wait or not own_wait or not up_wait else "no_admissible_start"
            continue
        if mode not in ACTIVE or setting is None:
            continue
        bounds = setpoint_bounds(configurations[room]["soft_limits"][mode])
        if bounds is None:
            continue
        direction = DIRECTION[mode]
        current_excess = direction*(local_zone["measured"]-local_zone["target"])
        if (current_excess > policy["comfort_band"] + TEMP_EPSILON and output[room]
                and local_zone["at"] > live[room]["adapter"]["last_mode_change"]):
            candidates.append((0, rank, room, "mode", "off", "energy_off"))
            reasons[room] = "exit_local_comfort_overrun"
            continue
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
        previous_zone = (own.get("control") or {}).get("zone", home) if own else home
        assistance_limited = previous_zone != home and current_excess >= policy["tracking_band"] - TEMP_EPSILON
        reducing = (direct or red_risk or assistance_limited or expected > policy["tracking_band"]+TEMP_EPSILON
                    or blocks_needed or redundant)
        low = power.get("bands", {}).get("low", {})
        quiet = quiet_power(now, live[room], row, policy["lower_output_observe_seconds"])
        eased = own is not None and own["kind"] == "temperature" and not is_strengthening(own)
        return_after = response_ready(now, local_zone, own, policy["reduction_wait_seconds"])
        if own and quiet and low["since"] > own["proxy_at"]:
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
                reasons[room] = ("high_power_overrun" if direct else "reduce_at_assistance_limit" if assistance_limited
                                 else "reduce_projected_or_neighbor_overrun")
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
        reasons[room] = (blocked.get(home,"unmet_conflicting_demand") if contest and owner is None and local_zone["recovery_active"] else
                         "wait_coordinated_exit" if (handoff_wait or opposite) and zone["recovery_active"] else
                         "unmet_conflicting_demand" if contest and any(zones[k]["direction"] == mode for k in demands) else
                         "observe_local_response" if not own_wait or not up_wait else "hold_effective_setting")
        row = dict(row, operation_since=max(live[room]["adapter"]["last_mode_change"], own["proxy_at"] if own else 0))
        maintain_setting(now, room, mode, setting, local_zone, local, row,
                         any(output[other] and other != room and home in units[other]["influences"] for other in output), policy,
                         topology["learning_versions"][room])
    scheduled = {scopes[item[2]]["zone"] for item in candidates if item[-1] in {"zone_strengthen","energy_start"}}
    trace["unmet"] = {key: ("handoff_pending" if handoff and key == owner else zones[key]["reason"])
                      for key in demands if key not in scheduled}
    trace["need"] = need
    trace["normal_need"] = need
    trace["gentle"] = {"state": "replaced_by_zone_tracking"}
    state["house_lead"] = min(candidates)[2] if candidates else next((r for r in sorted(output) if output[r]), None)
    return need, reasons, candidates
