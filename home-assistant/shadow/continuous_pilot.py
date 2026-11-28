"""Run low-frequency Xiaomi Home thermostat control with a durable journal."""
from collections import defaultdict
from hashlib import sha256
import json
import logging
from logging.handlers import RotatingFileHandler
from math import isfinite
import os
from pathlib import Path
import signal
from sqlite3 import Error as DatabaseError
from tempfile import NamedTemporaryFile
from threading import Event
from time import time
from traceback import extract_tb
from urllib.error import HTTPError

from cadence import SHARED_ROOMS, save_receipt
from house_dispatch import acknowledged, cancel_gentle, coordinate, final_gate, initial_safety_setting, public_gentle, recovery_initial
from house_policy import ACTIVE, INDEPENDENT, POLICY
from heating_pilot import DATA_DIR, ha_client, publish
from model import increases_output, numeric
from pilot_gate import ROOMS, matching_feedback, room_advice, room_feedback
from service import CONFIG, POWER_REPORT_GAP_SECONDS, encode, local
from live_evidence import attach as attach_power
from topology import extra_inputs, read_topology
from zone_control import allows_assistance, assistance_initial, coordinate as coordinate_zones, learned_initial


PERMIT = DATA_DIR / "continuous-permit.json"
LIVE_PERMIT = DATA_DIR / "live-permit.json"
MAX_STEPS = 3
ECHO_TIMEOUT = 180
BATCH_SPACING = 5
TOPOLOGY_PATH = CONFIG.with_name("topology.json")
TOPOLOGY = read_topology(TOPOLOGY_PATH, ROOMS)


AUDIT = logging.getLogger("thermostat.live")
RELEASE = {}
# Written automatically by earlier releases; operator-written holds keep other text.
LEGACY_EXTERNAL_HOLD = "检测到外部修改了空调设置，已暂停该机"
RECONCILE_DEFAULTS = {"external_override_seconds": 3600, "start_verify_seconds": 900, "start_power_grace_seconds": 240,
                      "start_retry_limit": 2, "start_retry_backoff_seconds": 900, "start_failure_hold_seconds": 7200,
                      "start_failure_reset_seconds": 86400}
RUNNING_WATTS_FLOOR = 30


def reconcile_setting(key):
    return POLICY.get(key, RECONCILE_DEFAULTS[key])


def configure_audit():
    directory = Path(__file__).parent
    marker = directory / ".release"
    commit = marker.read_text().strip() if marker.exists() else None
    if commit is not None and (len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)):
        raise ValueError("Invalid installed release marker")
    RELEASE.update(commit=commit, config_sha256=sha256(CONFIG.read_bytes() + b"\0" +
                   (directory / "house_policy.json").read_bytes() +
                   (b"\0" + TOPOLOGY_PATH.read_bytes() if TOPOLOGY is not None else b"")).hexdigest(),
                   code_sha256=sha256(b"\0".join(p.read_bytes() for p in sorted(directory.glob("*.py")))).hexdigest())
    handler = RotatingFileHandler(DATA_DIR / "live-audit.jsonl", maxBytes=8*1024*1024, backupCount=23, encoding="utf-8")
    AUDIT.setLevel(logging.INFO)
    AUDIT.propagate = False
    AUDIT.addHandler(handler)
    now = time()
    metadata = {"observation_version": TOPOLOGY["observation_version"] if TOPOLOGY else None,
                "observation_fingerprint": TOPOLOGY["fingerprint"] if TOPOLOGY else None}
    startup = dict(RELEASE, at=now, event="startup", **metadata)
    put_state(DATA_DIR/"startup.json", startup)
    put_state(DATA_DIR/f"startup-{int(now*1_000_000)}.json", startup)
    if TOPOLOGY is not None:
        put_state(DATA_DIR/f"configuration-{RELEASE['config_sha256']}.json",
                  {"rooms": ROOMS, "policy": POLICY, "topology": json.loads(TOPOLOGY_PATH.read_bytes())})
    audit("startup", **metadata, startup_at=now)


def audit(event, **fields):
    if AUDIT.isEnabledFor(logging.INFO):
        AUDIT.info(encode(dict(at=time(), event=event, **RELEASE, **fields)))


def audit_error(event, error, **fields):
    # No raw exception text, HTTP body, URL, source line or local variables.
    frames = [{"file": Path(f.filename).name, "function": f.name, "line": f.lineno}
              for f in extract_tb(error.__traceback__)]
    audit(event, error=type(error).__name__, stack=frames, **fields)


def group(room):
    return TOPOLOGY["actuators"][room]["lock_group"] if TOPOLOGY else "living_study" if room in SHARED_ROOMS else room


def active_lock_groups(rooms=None):
    return {group(room) for room in (ROOMS if rooms is None else rooms)}


def transition_for(state, room):
    transitions = state.get("transitions", {})
    key = group(room)
    if key in transitions and room in transitions[key].get("members", {}):
        return key, transitions[key]
    for candidate, batch in transitions.items():
        if room in batch.get("members", {}):
            return candidate, batch
    return None, None


def read_permit(now):
    try:
        raw = json.loads(LIVE_PERMIT.read_text(encoding="utf-8"))
    except FileNotFoundError:
        try:
            raw = json.loads(PERMIT.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if raw.get("end") is None:
            return None
    if raw.get("enabled") is not True:
        return None
    start, end = raw["start"], raw["end"]
    if not isfinite(start) or start > now or (end is not None and (not isfinite(end) or now >= end)):
        return None
    if set(raw["rooms"]) != set(ROOMS) or any(
        len(sig) != 3 or sig[0] not in {"cool", "heat"} or not isinstance(sig[1], (int,float)) or not 16 <= sig[1] <= 30
        for sig in raw["rooms"].values()
    ):
        return None
    isolated = raw.get("isolated_rooms", [])
    if not isinstance(isolated, list) or any(not isinstance(room, str) or room not in ROOMS for room in isolated):
        return None
    zones = raw.get("paused_zones", [])
    if not isinstance(zones, list) or any(TOPOLOGY is None or key not in TOPOLOGY["zones"] for key in zones):
        return None
    return raw


def put_state(path, state):
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as out:
        out.write(encode(state) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(out.name, path)


def running_watts(room):
    reference = ROOMS[room].get("verified_off_power")
    matched = isinstance(reference, dict) and reference.get("entity") == ROOMS[room]["power"]
    ceiling = numeric(reference.get("ceiling_w")) if matched else None
    return max(RUNNING_WATTS_FLOOR, 2*ceiling) if ceiling is not None and ceiling > 0 else RUNNING_WATTS_FLOOR


def is_start(action):
    return action["kind"] == "mode" and action["value"] in ACTIVE


def clock(stamp):
    return local(stamp).strftime("%H：%M")


def start_power(action, values, now):
    """Return a power report taken after the command while the unit reports the started mode."""
    adapter = values["adapter"]
    watts = numeric(adapter["power"])
    age = numeric(adapter["from_power_s"])
    if (watts is None or not 0 <= watts <= 10000 or age is None or not 0 <= age <= POWER_REPORT_GAP_SECONDS
            or now-age < action["sent_at"] or adapter["mode"] != action["value"]):
        return None
    return watts


def park_from_report(state, room, values, previous_mode, started_at=None):
    off_at = values["adapter"]["last_mode_change"]
    state.setdefault("parked", {})[room] = {"off_at": off_at, "mode": previous_mode,
                                            "started_at": off_at if started_at is None else started_at}


def reconcile_report(state, room, values, now):
    """Adopt a reported state that no pending command explains, without a standing pause."""
    signature = list(values["signature"][:3])
    previous = state["baselines"][room]
    state["baselines"][room] = signature
    parked = state.setdefault("parked", {})
    if previous[0] == "off" and signature[0] == "off":
        # A stopped unit can refresh its setpoint or fan attributes without any operation.
        if room not in parked:
            park_from_report(state, room, values, "off")
        audit("stopped_report_drift", room=room, previous=previous, reported=signature)
        return
    failures = state.setdefault("start_failures", {})
    last = next((a for a in reversed(state["commands"])
                 if a["room"] == room and a["status"] == "proxy_echo" and a["kind"] == "mode"), None)
    # A stop before a working compressor would draw power is treated as a person's choice.
    failed_start = (last is not None and is_start(last) and previous[0] == last["value"] and signature[0] == "off"
                    and last["sent_at"] != failures.get(room, {}).get("command_at")
                    and reconcile_setting("start_power_grace_seconds") <= now-last["proxy_at"]
                    <= reconcile_setting("start_verify_seconds")
                    and last.get("peak_watts") is not None and last["peak_watts"] < running_watts(room))
    overrides = state.setdefault("external_overrides", {})
    _, batch = transition_for(state, room)
    if batch and room in batch["members"]:
        batch["members"][room]["phase"] = "done"
    if failed_start:
        overrides.pop(room, None)
        earlier = failures.get(room, {})
        recent = now-earlier.get("at", now) <= reconcile_setting("start_failure_reset_seconds")
        count = earlier.get("count", 0) + 1 if recent else 1
        entry = {"count": count, "at": now, "mode": last["value"], "command_at": last["sent_at"],
                 "peak_watts": last.get("peak_watts")}
        if count > reconcile_setting("start_retry_limit"):
            entry["hold_until"] = now + reconcile_setting("start_failure_hold_seconds")
        else:
            entry["retry_at"] = now + reconcile_setting("start_retry_backoff_seconds") * 2**(count-1)
        failures[room] = entry
        park_from_report(state, room, values, last["value"], last["proxy_at"])
        audit("start_unverified", room=room, previous=previous, reported=signature, **entry)
        return
    overrides[room] = {"at": now, "until": now + reconcile_setting("external_override_seconds"),
                       "previous": previous, "reported": signature}
    if signature[0] == "off":
        park_from_report(state, room, values, previous[0] if previous[0] in ACTIVE else "off")
    else:
        parked.pop(room, None)
        if signature[0] in ACTIVE:
            failures.pop(room, None)
    audit("external_override", room=room, previous=previous, reported=signature, until=overrides[room]["until"])


def release_legacy_holds(state, live):
    for room in [r for r, reason in state["manual_holds"].items() if reason == LEGACY_EXTERNAL_HOLD and r in live
                 and live[r]["signature"][0] not in {"unavailable", "unknown"}]:
        del state["manual_holds"][room]
        values = live[room]
        signature = list(values["signature"][:3])
        previous = state["baselines"][room]
        state["baselines"][room] = signature
        if signature[0] == "off" and room not in state["parked"]:
            park_from_report(state, room, values, previous[0] if previous[0] in ACTIVE else "off")
        audit("legacy_external_hold_released", room=room, previous=previous, reported=signature)


def yielded(state, room, now):
    """Return why a room temporarily yields control, expiring stale entries."""
    override = state.get("external_overrides", {}).get(room)
    if override is not None:
        if now < override["until"]:
            return f"检测到外部调整，暂时让出调度至 {clock(override['until'])}"
        del state["external_overrides"][room]
        audit("external_override_expired", room=room, reported=override["reported"])
    failure = state.get("start_failures", {}).get(room)
    if failure is not None:
        if now < failure.get("hold_until", 0):
            return f"开机多次未生效，暂停重试至 {clock(failure['hold_until'])}"
        if now < failure.get("retry_at", 0):
            return f"开机未生效，{clock(failure['retry_at'])} 后重试"
    return None


def settle(state, live, now, *, trial=True):
    """Check proxy echoes and collect observations without claiming physical verification."""
    holds = state["holds"]
    for action in state["commands"]:
        room = action["room"]
        values = live[room]
        signature = values["signature"][:3]
        age = now - action["sent_at"]
        previous_status = action["status"]
        if action["status"] == "prepared":
            action["status"] = "unconfirmed"
            holds[group(room)] = "发令结果未写入，已暂停复发"
        elif action["status"] == "accepted":
            mode, setpoint, _ = signature
            echoed = (signature == tuple(action["expected"]) if action["kind"] == "temperature" else
                      mode == action["value"] and (mode == "off" or numeric(setpoint) is not None and 16 <= setpoint <= 30))
            if echoed and action["sent_at"]-1 <= values["adapter"]["updated_at"] <= action["sent_at"]+ECHO_TIMEOUT:
                action.update(status="proxy_echo", proxy_at=values["adapter"]["updated_at"])
                state["baselines"][room] = list(signature)
                if TOPOLOGY is None and action["kind"] == "temperature" and action.get("target_at_send") is not None and mode in ACTIVE:
                    state.setdefault("last_mode_setting", {}).setdefault(room, {})[mode] = {
                        "target": action["target_at_send"], "value": setpoint, "at": action["proxy_at"]}
                if action["purpose"] in {"energy_off", "handoff_off"}:
                    if TOPOLOGY is None and action.get("target_at_send") is not None and numeric(action["before"][1]) is not None:
                        state.setdefault("last_mode_setting", {}).setdefault(room, {})[action["before"][0]] = {
                            "target": action["target_at_send"], "value": action["before"][1], "at": action["proxy_at"]}
                    state.setdefault("parked", {})[room] = {"off_at": action["proxy_at"],
                        "mode": action["before"][0], "started_at": action["before_mode_at"]}
                    if action["purpose"] == "handoff_off":
                        state["parked"][room]["handoff"] = True
                elif action["purpose"] == "energy_start":
                    state.setdefault("parked", {}).pop(room, None)
                _, batch = transition_for(state, room)
                if batch and action.get("transaction") == batch["id"]:
                    member = batch["members"][room]
                    batch["ready_at"] = max(action["proxy_at"], action["sent_at"] + BATCH_SPACING)
                    if action["purpose"] == "reversal_off":
                        duration = ROOMS[room]["transition_off_seconds"]
                        if not 180 <= duration <= 300:
                            raise ValueError("transition_off_seconds must be between 180 and 300")
                        member.update(phase="off", off_echo_at=action["proxy_at"],
                                      off_until=max(now, action["proxy_at"]) + duration)
                    elif action["purpose"] in {"reversal_mode", "energy_start"}:
                        member["phase"] = "initial"
                    else:
                        member["phase"] = "done"
                elif action["kind"] == "mode" and trial:
                    holds[group(room)] = "换向已收到伴侣回报，待核对室内机实际模式"
            elif age >= ECHO_TIMEOUT or (signature not in (tuple(action["before"]), (None, None, None))
                                         and mode not in {"unavailable", "unknown"} and not echoed):
                action["status"] = "unconfirmed"
                holds[group(room)] = "伴侣回报超时或与指令不一致，已暂停复发"
        if previous_status != action["status"]:
            audit("receipt", room=room, command_at=action["sent_at"], status=action["status"], http=action.get("http"))
        if action["status"] == "proxy_echo":
            watts = start_power(action, values, now) if is_start(action) else None
            if watts is not None and now-action["proxy_at"] <= reconcile_setting("start_verify_seconds"):
                action["peak_watts"] = max(action.get("peak_watts") or 0, watts)
                if action["peak_watts"] >= running_watts(room) and state.get("start_failures", {}).pop(room, None):
                    audit("start_verified_by_power", room=room, command_at=action["sent_at"], watts=action["peak_watts"])
            for minute in (5, 15):
                key = str(minute)
                if age >= minute*60 and key not in action["followup"]:
                    action["followup"][key] = {"at": now, "room_c": numeric(values["temperature"]),
                        "room_age_s": round(values["temperature_age"]), "watts": numeric(values["adapter"]["power"]),
                        "power_age_s": round(values["adapter"]["from_power_s"]), "reported": list(signature),
                        "temperature_report_at": values.get("temperature_report_at"),
                        "power_report_at": values["adapter"].get("power_report_at")}


def archive_actions(path, state, now):
    archives = defaultdict(dict)
    recent = []
    for action in state["commands"]:
        if action["sent_at"] < now-86400 and action["status"] not in {"prepared", "accepted"}:
            archives[local(action["sent_at"]).date()][str(action["sent_at"])] = action
        else:
            recent.append(action)
    for day, actions in archives.items():
        target = path.with_name(f"{path.stem}-actions-{day}.json")
        saved = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        saved.update(actions)
        put_state(target, saved)
    state["commands"] = recent


def assess(live, state, now, advise_room=room_advice, *, trial=True, isolated=(), trace=None, inputs=None, paused=()):
    trace = {} if trace is None else trace
    pending = {a["room"] for a in state["commands"] if a["status"] in {"prepared", "accepted"}}
    transitions = state.setdefault("transitions", {})
    manual = state.setdefault("manual_holds", {})
    parked = state.setdefault("parked", {})
    connected = set(live) - set(isolated)
    if TOPOLOGY is not None:
        state.pop("gentle", None)
    release_legacy_holds(state, live)
    for room, values in live.items():
        mode, setpoint, fan = values["signature"][:3]
        if (room not in pending and mode not in {"unavailable", "unknown"}
                and [mode, setpoint, fan] != state["baselines"][room]):
            reconcile_report(state, room, values, now)
    echoes = acknowledged(state["commands"])
    details, ready, members = {}, {}, {}
    known_groups = active_lock_groups(live)
    orphan_holds = {k: v for k, v in state["holds"].items() if k not in known_groups}
    for key, batch in list(transitions.items()):
        member_rooms = batch.get("members", {})
        held = key in state["holds"] or any(group(r) in state["holds"] for r in member_rooms if r in live)
        for room, member in member_rooms.items():
            if room not in live:
                if member.get("phase") != "done":
                    orphan_holds.setdefault(key, "存在未完成且未归属当前设备的事务记录")
                continue
            if (room not in pending and not held and member["phase"] == "off"
                    and live[room]["signature"][0] == "off"):
                off = next((a for a in reversed(echoes) if a["room"] == room
                            and a.get("transaction") == batch["id"] and a["purpose"] == "reversal_off"), None)
                if off:
                    parked[room] = {"off_at": off["proxy_at"], "mode": off["before"][0],
                                    "started_at": off.get("before_mode_at", off["proxy_at"])}
                    member["phase"] = "done"
            if member["phase"] == "queued" and room not in pending and not held:
                member["phase"] = "done"  # Recompute unsent plans from the current whole-house demand.
            if member["phase"] != "done":
                members[room] = (batch, member)
        if (not held and not (pending & member_rooms.keys()) and set(member_rooms) <= set(live)
                and all(member["phase"] == "done" for member in member_rooms.values())):
            del transitions[key]
    orphan_reason = (f"存在未归属当前锁组的暂停记录（{'、'.join(sorted(orphan_holds))}），请核对后清理或启用新许可"
                     if orphan_holds else None)
    for room, values in live.items():
        mode, setpoint, fan, requested, target, preset, native_on = values["signature"]
        power = numeric(values["adapter"]["power"])
        yielding = yielded(state, room, now)
        hold = state["holds"].get(group(room)) or orphan_reason or manual.get(room) or yielding
        ready[room] = (room in connected and not hold and room not in pending
                       and requested == "auto" and preset == "标准" and native_on is False
                       and values["native_authorized"] is False and power is not None and 0 <= power <= 10000
                       and ((mode == "off" and room in parked)
                            or (mode in ACTIVE and numeric(setpoint) is not None and 16 <= setpoint <= 30)))
        details[room] = {"room_c": numeric(values["temperature"]), "target_c": target, "mode_reported": mode,
                         "setpoint_reported": setpoint, "power_w": power, "ready": ready[room],
                         "temperature_age_s": numeric(values["temperature_age"]),
                         "power_age_s": numeric(values["adapter"]["from_power_s"]),
                         "temperature_report_at": values.get("temperature_report_at"),
                         "status": hold or ("手动隔离" if room in isolated else
                                            "等待自动状态和有效数据" if not ready[room] else "等待全屋分配")}
    if pending:
        if plan := state.get("gentle"):
            zone, unit = plan["zone"], plan["unit"]
            if any(r not in connected or r in manual or group(r) in state["holds"]
                   or live[r]["signature"][3] != "auto" or live[r]["signature"][5:] != ("标准", False)
                   or live[r]["native_authorized"] is not False for r in (zone, unit)):
                trace["gentle_exit"] = cancel_gentle(state, now, "control_paused")
            elif (live[zone].get("temperature_source") != plan["source"]
                  or any(live[r]["signature"][4] != plan["target"] for r in (zone, unit))):
                trace["gentle_exit"] = cancel_gentle(state, now, "target_or_source_changed")
        trace["pending_rooms"] = sorted(pending)
        for room in pending:
            details[room]["status"] = "等待米家指令回报"
        return details, None
    last_control = max((a["proxy_at"] for a in echoes if a["room"] in connected), default=0)
    references = set(INDEPENDENT) & connected
    readings = {}
    for room, values in live.items():
        if room not in (connected if TOPOLOGY is not None else references) and not ready[room]:
            continue
        try:
            readings[room] = advise_room(room, values["signature"][0], offsets=values["offsets"],
                                         feedback=values, house_since=last_control)
        except (ValueError, DatabaseError) as error:
            if TOPOLOGY is None:
                raise
            ready[room] = False
            details[room].update(ready=False, status="设备见证或功率证据暂不可读")
            audit_error("evidence_error", error, room=room)
    if TOPOLOGY is not None:
        meter_memory = state.setdefault("power_live", {})
        readings = {room: attach_power(now, live[room], row, meter_memory.setdefault(room, {}))
                    for room, row in readings.items()}
        for room in details:
            proof = readings.get(room, {}).get("power_evidence") or {}
            details[room].update(power_report_at=live[room]["adapter"].get("power_report_at"),
                                 temperature_source=live[room].get("temperature_source"),
                                 power_context=live[room].get("power_context"),
                                 power_bands=proof.get("bands"), power_evidence_at=proof.get("last_report"))

    def allocate(available, record, *, acquire_gentle=True):
        if TOPOLOGY is not None:
            return coordinate_zones(live, readings, available, state, now, ROOMS, connected,
                                    TOPOLOGY, inputs or {}, set(paused), record, control_ready=ready)
        return coordinate(live, readings, available, state, now, ROOMS, connected, trace=record,
                          control_ready=ready, acquire_gentle=acquire_gentle)
    # Finish an already confirmed mode change before making any unrelated allocation.
    incomplete = {room for room, (_, member) in members.items() if member["phase"] == "initial"}
    busy_groups = set(transitions) | {group(r) for r in members if r in live}
    allocation_ready = {room: enabled and room not in incomplete and group(room) not in busy_groups
                        for room, enabled in ready.items()}
    need, reasons, candidates = allocate(allocation_ready, trace)
    plan = state.get("gentle")
    invalidated, closed, safety = [], [], {}
    for room in sorted(incomplete):
        batch, member = members[room]
        feedback = live[room]
        mode, setpoint, _, _, target, _, _ = feedback["signature"]
        if not ready[room] or mode != batch["mode"]:
            continue
        if TOPOLOGY is not None:
            home = TOPOLOGY["actuators"][room]["zone"]
            zone_name = (member.get("control") or {}).get("zone", home)
            zone = trace["zones"].get(zone_name)
            if zone is None or zone["paused"] or zone["quality"] != "fresh":
                if zone is None or zone["paused"]:
                    member["phase"] = "done"
                    closed.append(room)
                    invalidated.append(room)
                continue
        elif need == "unknown" or matching_feedback(now, feedback, readings[room]["sample"], target):
            continue
        optional = member.get("gentle_at")
        invalid = need != mode or (optional is not None and trace["normal_need"] != mode and (
            plan is None or plan["unit"] != room or plan["selected_at"] != optional
            or trace["gentle"]["reason"] in {"inside_ideal_band", "peer_comfort_risk", "returning_to_ideal"}))
        if TOPOLOGY is not None:
            binding = member.get("control") or {}
            home_zone = trace["zones"][home]
            invalid = (zone_name != trace["owner"] or binding.get("definition") != TOPOLOGY["action_versions"][room]
                       or binding.get("source_version") != zone["source_version"] or binding.get("target") != zone["target"]
                       or not zone["qualified"] or zone["direction"] != mode or room not in trace["feasible"].get(zone_name, ())
                       or (zone_name != home and not allows_assistance(home_zone, mode)))
            target = home_zone["target"]
        if invalid:
            if not member.get("initial_invalidated") or member["target"] != target:
                invalidated.append(room)
            member.pop("initial", None)
            member.update(initial_invalidated=True, target=target)
        safe_setting = initial_safety_setting(mode, numeric(setpoint), numeric(target), ROOMS[room]["soft_limits"][mode])
        if safe_setting is not None:
            safety[room] = safe_setting
        elif invalid:
            # Only close unsent work. The echo, baseline and historical
            # commands remain in the durable journal.
            member["phase"] = "done"
            closed.append(room)
    if closed:
        incomplete.difference_update(closed)
        for room in closed:
            members.pop(room, None)
        for key, batch in list(transitions.items()):
            member_rooms = batch.get("members", {})
            held = key in state["holds"] or any(group(r) in state["holds"] for r in member_rooms if r in live)
            if not held and set(member_rooms) <= set(live) and all(m["phase"] == "done" for m in member_rooms.values()):
                del transitions[key]
        busy_groups = set(transitions) | {group(r) for r in members if r in live}
        allocation_ready = {room: enabled and room not in incomplete and group(room) not in busy_groups
                            for room, enabled in ready.items()}
        # Assess again to release allocation slots. Preserve the first
        # lifecycle event in the audit, and do not acquire a new gentle owner
        # during the same assessment that discarded an old continuation.
        followup = {}
        need, reasons, candidates = allocate(allocation_ready, followup, acquire_gentle=trace["gentle"]["state"] != "exited")
        if TOPOLOGY is not None:
            trace.update(followup)
        elif trace["gentle"]["state"] not in {"selected", "exited"}:
            trace["gentle"] = followup["gentle"]
    trace.update(need=need, lead=state.get("house_lead"), invalidated_initial=invalidated)
    for room, detail in details.items():
        detail["house_need"] = need
        if room in reasons:
            detail["status"] = reasons[room]
        if TOPOLOGY is not None:
            zone = trace["zones"][TOPOLOGY["actuators"][room]["zone"]]
            detail.update(zone=zone["key"], zone_intent=zone["intent"], zone_reason=zone["reason"],
                          zone_target_c=zone["target"], zone_temperature_c=zone["measured"])
        elif room in readings:
            view = readings[room]["house_view"]
            detail["prediction_10m_c"] = view["trend"] if view is not None else None
            detail["trend_valid"] = view is not None and view["trend"] is not None
            if view is not None:
                detail.update({key: view[key] for key in ("cold_seconds", "warm_seconds", "cold_confirmed", "warm_confirmed")})
    for room in sorted(incomplete):
        batch, member = members[room]
        if not ready[room] or now < batch["ready_at"] or need == "unknown":
            continue
        if room in safety:
            details[room]["status"] = "已确认模式继承超出软限的设温，按当前目标安全减载"
            candidates.append((-3, (0, 0), room, "temperature", safety[room], "initial_safety_easing"))
            continue
        if need == "conflict":
            continue
        mode = live[room]["adapter"]["mode"]
        current = live[room]["adapter"]["setpoint"]
        target = live[room]["signature"][4]
        if (mode != batch["mode"] or need != mode
                or (state.get("gentle") is not None and state["gentle"]["unit"] != room)):
            continue
        if TOPOLOGY is not None:
            binding = member.get("control") or {}
            zone_name = binding.get("zone")
            zone = trace["zones"].get(zone_name)
            if (zone is None or zone_name != trace["owner"] or not zone["qualified"]
                    or zone["source_version"] != binding.get("source_version") or zone["target"] != binding.get("target")):
                continue
            home_zone = trace["zones"][TOPOLOGY["actuators"][room]["zone"]]
            initial = (learned_initial(room, mode, home_zone, state["closed_loop"], ROOMS[room]["soft_limits"][mode])
                       if zone_name == home_zone["key"] else
                       assistance_initial(mode, home_zone, ROOMS[room]["soft_limits"][mode]))
            target = home_zone["target"]
            trace["feasible"].setdefault(zone_name, []).append(room)
            trace["action_scopes"][room] = dict(binding, before_c=zone["measured"], at=zone["at"])
        else:
            initial = recovery_initial(room, mode, live, readings[room], state, ROOMS)
            if matching_feedback(now, live[room], readings[room]["sample"], target):
                continue
        if initial is None:
            continue
        if initial != current and increases_output(mode, current, initial):
            if need != mode or TOPOLOGY is None and any(live[peer]["adapter"]["mode"] not in {mode, "off", "fan_only"}
                                   for peer in connected - {room}):
                details[room]["status"] = "等待本模式的全屋需求并确认无反向设备"
                continue
        member.update(initial=initial, target=target)
        if initial == current:
            member["phase"] = "done"
        else:
            details[room]["status"] = "按当前目标补发已确认模式的初值"
            candidates.append((-1, (0, 0), room, "temperature", initial, "reversal_initial"))
    if plan := state.get("gentle"):
        trace["gentle"]["allocation_reason"] = details[plan["unit"]]["status"]
    trace["candidates"] = [item[2:] for item in sorted(candidates)]
    trace["rejections"] = []
    for priority, ranking, *action in sorted(candidates):
        room, kind, _, purpose = action
        if (trial and kind == "temperature" and purpose not in {"reversal_initial", "initial_safety_easing"}
                and sum(c["room"] == room and c["kind"] == "temperature" and c["status"] != "canceled"
                        for c in state["commands"]) >= MAX_STEPS):
            details[room]["status"] = "已达本次试点的设温次数限制"
            trace["rejections"].append([action, details[room]["status"]])
            continue
        if flag := final_gate(action, now, live, readings, connected, state, ROOMS, topology=TOPOLOGY, trace=trace):
            details[action[0]]["status"] = flag
            trace["rejections"].append([action, flag])
        else:
            return details, tuple(action)
    return details, None


def start_batch(state, room, desired, details, sent, control=None):
    member = {"phase": "queued", "target": details[room]["target_c"]}
    if control:
        member["control"] = control
    if plan := state.get("gentle"):
        if plan["unit"] == room and plan["mode"] == desired:
            member["gentle_at"] = plan["selected_at"]
    batch = {"id": sent, "mode": desired, "members": {room: member}, "ready_at": sent}
    state["transitions"][group(room)] = batch
    return batch


def cancel_inactive_gentle(now):
    exits = []
    for path in DATA_DIR.glob("continuous-*.json"):
        if not path.stem.removeprefix("continuous-").isdigit():
            continue
        try:
            state = json.loads(path.read_text())
        except FileNotFoundError:
            continue
        if ended := cancel_gentle(state, now, "permit_inactive"):
            put_state(path, state)
            exits.append(ended)
    return exits


def display_progress(message, details, trace, selection=None):
    if TOPOLOGY is None:
        publish(message, details)
        return
    control = {key: trace.get(key) for key in ("topology", "definition", "zones", "environment", "season", "need", "owner",
                                               "lead", "risks", "candidates", "rejections", "pending_rooms", "unmet", "handoff", "conflict", "yielding_blocks", "elected")}
    control.update(at=time(), selected=selection)
    publish(message, details, control=control)


def cycle(call):
    permit = read_permit(time())
    if permit is None:
        exits = cancel_inactive_gentle(time())
        audit("cycle", outcome="permit_inactive", gentle_exits=exits)
        publish("实机控制已暂停或验证窗口已结束")
        return
    trial = permit["end"] is not None
    end = permit["end"] if trial else float("inf")
    path = DATA_DIR / f"continuous-{int(permit['start'])}.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {"start": permit["start"], "baselines": permit["rooms"], "holds": {}, "commands": []}
        put_state(path, state)
    live = {room: room_feedback(call, room) for room in ROOMS}
    before = encode(state)
    now = time()
    settle(state, live, now, trial=trial)
    if not trial:
        archive_actions(path, state, now)
    settled = encode(state)
    if before != settled:
        put_state(path, state)
    before = settled
    isolated = permit.get("isolated_rooms", ())
    trace = {"permit_sha256": sha256(encode(permit).encode()).hexdigest()}
    inputs = extra_inputs(call, TOPOLOGY)
    paused = permit.get("paused_zones", ())
    details, selection = assess(live, state, now, trial=trial, isolated=isolated, trace=trace, inputs=inputs, paused=paused)
    if before != encode(state):
        put_state(path, state)
    if selection is None or now >= end-30:
        audit("cycle", outcome="idle" if selection is None else "permit_expiring", assessment=trace, rooms=details)
        display_progress("实机持续试点运行中" if trial else "持续实机控制运行中", details, trace)
        return

    room, kind, value, purpose = selection
    refreshed = {code: room_feedback(call, code) for code in ROOMS}
    verification = {}
    refreshed_inputs = extra_inputs(call, TOPOLOGY)
    checked, confirmed = assess(refreshed, state, time(), trial=trial, isolated=isolated, trace=verification,
                                inputs=refreshed_inputs, paused=paused)
    scope_key = lambda t: tuple((t.get("action_scopes", {}).get(room) or {}).get(k)
                                for k in ("zone", "target", "definition", "source_version", "handoff_at", "handoff_phase"))
    if selection != confirmed or scope_key(trace) != scope_key(verification) or read_permit(time()) != permit:
        if selection == confirmed:
            ended = cancel_gentle(state, time(), "permit_inactive")
            if ended:
                verification["gentle_exit"] = ended
        put_state(path, state)
        audit("cycle", outcome="recheck_changed", assessment=trace, verification=verification,
              proposed=selection, final=None, rooms=checked)
        display_progress("发令前条件变化，本轮维持原设置", checked, verification)
        return
    previous = list(refreshed[room]["signature"][:3])
    sent = time()
    if sent >= end-25:
        audit("cycle", outcome="permit_expiring", assessment=trace, verification=verification, rooms=checked)
        return
    batch = None
    control = verification.get("action_scopes", {}).get(room)
    if purpose in {"energy_start", "reversal_initial", "initial_safety_easing"}:
        _, batch = transition_for(state, room)
        if batch is None and purpose == "energy_start":
            batch = start_batch(state, room, value, checked, sent, control)
    action = {"room": room, "kind": kind, "value": value, "before": previous, "sent_at": sent, "status": "prepared",
              "purpose": purpose, "transaction": batch["id"] if batch else None, "target_at_send": checked[room]["target_c"],
              "expected": [previous[0], value, previous[2]] if kind == "temperature" else [value],
              "before_room_c": numeric(refreshed[room]["temperature"]),
              "before_watts": numeric(refreshed[room]["adapter"]["power"]),
              "before_mode_at": refreshed[room]["adapter"]["last_mode_change"], "followup": {}, "physical_verified": False}
    if control is not None:
        action["control"] = control
    if ((plan := state.get("gentle")) and room == plan["unit"]
            and purpose in {"energy_start", "gentle_temperature", "reversal_initial"}):
        action["gentle"] = public_gentle(plan)
    state["commands"].append(action)
    put_state(path, state)
    if read_permit(time()) != permit or time() >= end-20:
        action.update(status="canceled")
        ended = cancel_gentle(state, time(), "permit_inactive")
        put_state(path, state)
        audit("cycle", outcome="permit_revoked", assessment=trace, verification=verification,
              proposed=selection, final=None, command_at=sent, rooms=checked, gentle_exits=[ended] if ended else [])
        return
    service, field = ("set_temperature", "temperature") if kind == "temperature" else ("set_hvac_mode", "hvac_mode")
    audit("cycle", outcome="dispatch", assessment=trace, verification=verification, final=selection,
          command_at=sent, rooms=checked)
    try:
        code = call("services/climate/" + service, {"entity_id": ROOMS[room]["adapter"], field: value})
        action.update(status="accepted", http=code)
    except HTTPError as error:
        action.update(status="unconfirmed" if error.code >= 500 else "rejected", http=error.code)
        audit_error("relay_error", error, room=room, command_at=sent, http=error.code)
        error.close()
    except OSError as error:
        action.update(status="unconfirmed")
        audit_error("relay_error", error, room=room, command_at=sent)
    received = time()
    action["relay_at"] = received
    if action["status"] != "accepted":
        state["holds"][group(room)] = "米家调用失败或未及时返回，已暂停复发"
    put_state(path, state)
    save_receipt(DATA_DIR, room, sent, action["status"], received, purpose=purpose)
    audit("relay", room=room, command_at=sent, status=action["status"], http=action.get("http"))
    display_progress("已记录一条米家实机指令，等待伴侣回报", checked, verification, selection)


def main():
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    configure_audit()
    try:
        call = ha_client()
    except Exception as error:
        audit_error("startup_error", error)
        raise
    while not stop.is_set():
        try:
            cycle(call)
        except Exception as error:
            audit_error("cycle_error", error)
            publish(f"数据或服务异常，暂停发令（{type(error).__name__}）")
        stop.wait(20)


if __name__ == "__main__":
    main()
