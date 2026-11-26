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
from service import CONFIG, encode, local


PERMIT = DATA_DIR / "continuous-permit.json"
LIVE_PERMIT = DATA_DIR / "live-permit.json"
MAX_STEPS = 3
ECHO_TIMEOUT = 180
BATCH_SPACING = 5


AUDIT = logging.getLogger("thermostat.live")
RELEASE = {}

def configure_audit():
    directory = Path(__file__).parent
    marker = directory / ".release"
    commit = marker.read_text().strip() if marker.exists() else None
    if commit is not None and (len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)):
        raise ValueError("Invalid installed release marker")
    RELEASE.update(commit=commit, config_sha256=sha256(CONFIG.read_bytes() + b"\0" +
                   (directory / "house_policy.json").read_bytes()).hexdigest(),
                   code_sha256=sha256(b"\0".join(p.read_bytes() for p in sorted(directory.glob("*.py")))).hexdigest())
    handler = RotatingFileHandler(DATA_DIR / "live-audit.jsonl", maxBytes=4*1024*1024, backupCount=7, encoding="utf-8")
    AUDIT.setLevel(logging.INFO)
    AUDIT.propagate = False
    AUDIT.addHandler(handler)
    audit("startup")


def audit(event, **fields):
    if AUDIT.isEnabledFor(logging.INFO):
        AUDIT.info(encode(dict(at=time(), event=event, **RELEASE, **fields)))


def audit_error(event, error, **fields):
    # No raw exception text, HTTP body, URL, source line or local variables.
    frames = [{"file": Path(f.filename).name, "function": f.name, "line": f.lineno}
              for f in extract_tb(error.__traceback__)]
    audit(event, error=type(error).__name__, stack=frames, **fields)


def group(room):
    return "living_study" if room in SHARED_ROOMS else room


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
    return raw


def put_state(path, state):
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as out:
        out.write(encode(state) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(out.name, path)


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
                if action["kind"] == "temperature" and action.get("target_at_send") is not None and mode in ACTIVE:
                    state.setdefault("last_mode_setting", {}).setdefault(room, {})[mode] = {
                        "target": action["target_at_send"], "value": setpoint, "at": action["proxy_at"]}
                if action["purpose"] == "energy_off":
                    if action.get("target_at_send") is not None and numeric(action["before"][1]) is not None:
                        state.setdefault("last_mode_setting", {}).setdefault(room, {})[action["before"][0]] = {
                            "target": action["target_at_send"], "value": action["before"][1], "at": action["proxy_at"]}
                    state.setdefault("parked", {})[room] = {"off_at": action["proxy_at"],
                        "mode": action["before"][0], "started_at": action["before_mode_at"]}
                elif action["purpose"] == "energy_start":
                    state.setdefault("parked", {}).pop(room, None)
                batch = state.get("transitions", {}).get(group(room))
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
            for minute in (5, 15):
                key = str(minute)
                if age >= minute*60 and key not in action["followup"]:
                    action["followup"][key] = {"at": now, "room_c": numeric(values["temperature"]),
                        "room_age_s": round(values["temperature_age"]), "watts": numeric(values["adapter"]["power"]),
                        "power_age_s": round(values["adapter"]["from_power_s"]), "reported": list(signature)}


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


def assess(live, state, now, advise_room=room_advice, *, trial=True, isolated=(), trace=None):
    trace = {} if trace is None else trace
    pending = {a["room"] for a in state["commands"] if a["status"] in {"prepared", "accepted"}}
    transitions = state.setdefault("transitions", {})
    manual = state.setdefault("manual_holds", {})
    parked = state.setdefault("parked", {})
    connected = set(live) - set(isolated)
    echoes = acknowledged(state["commands"])
    details, ready, members = {}, {}, {}
    for key, batch in list(transitions.items()):
        for room, member in batch["members"].items():
            if (room not in pending and key not in state["holds"] and member["phase"] == "off"
                    and live[room]["signature"][0] == "off"):
                off = next((a for a in reversed(echoes) if a["room"] == room
                            and a.get("transaction") == batch["id"] and a["purpose"] == "reversal_off"), None)
                if off:
                    parked[room] = {"off_at": off["proxy_at"], "mode": off["before"][0],
                                    "started_at": off.get("before_mode_at", off["proxy_at"])}
                    member["phase"] = "done"
            if member["phase"] == "queued" and room not in pending and key not in state["holds"]:
                member["phase"] = "done"  # Recompute unsent plans from the current whole-house demand.
            if member["phase"] != "done":
                members[room] = (batch, member)
        if key not in state["holds"] and not (pending & batch["members"].keys()) and all(
                member["phase"] == "done" for member in batch["members"].values()):
            del transitions[key]
    for room, values in live.items():
        mode, setpoint, fan, requested, target, preset, native_on = values["signature"]
        if (room not in pending and mode not in {"unavailable", "unknown"}
                and [mode, setpoint, fan] != state["baselines"][room]):
            manual.setdefault(room, "检测到外部修改了空调设置，已暂停该机")
        power = numeric(values["adapter"]["power"])
        hold = state["holds"].get(group(room)) or manual.get(room)
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
    readings = {room: advise_room(room, values["signature"][0], offsets=values["offsets"],
                                  feedback=values, house_since=last_control)
                for room, values in live.items() if ready[room] or room in references}
    # Finish an already confirmed mode change before making any unrelated allocation.
    incomplete = {room for room, (_, member) in members.items() if member["phase"] == "initial"}
    allocation_ready = {room: enabled and room not in incomplete and group(room) not in transitions
                        for room, enabled in ready.items()}
    need, reasons, candidates = coordinate(live, readings, allocation_ready, state, now, ROOMS, connected,
                                           trace=trace, control_ready=ready)
    plan = state.get("gentle")
    invalidated, closed, safety = [], [], {}
    for room in sorted(incomplete):
        batch, member = members[room]
        feedback = live[room]
        mode, setpoint, _, _, target, _, _ = feedback["signature"]
        if (not ready[room] or need == "unknown" or mode != batch["mode"]
                or matching_feedback(now, feedback, readings[room]["sample"], target)):
            continue
        optional = member.get("gentle_at")
        invalid = need != mode or (optional is not None and trace["normal_need"] != mode and (
            plan is None or plan["unit"] != room or plan["selected_at"] != optional
            or trace["gentle"]["reason"] in {"inside_ideal_band", "peer_comfort_risk", "returning_to_ideal"}))
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
        for key, batch in list(transitions.items()):
            if key not in state["holds"] and all(m["phase"] == "done" for m in batch["members"].values()):
                del transitions[key]
        allocation_ready = {room: enabled and room not in incomplete and group(room) not in transitions
                            for room, enabled in ready.items()}
        # Assess again to release allocation slots. Preserve the first
        # lifecycle event in the audit, and do not acquire a new gentle owner
        # during the same assessment that discarded an old continuation.
        followup = {}
        need, reasons, candidates = coordinate(live, readings, allocation_ready, state, now, ROOMS, connected,
                                               trace=followup, control_ready=ready,
                                               acquire_gentle=trace["gentle"]["state"] != "exited")
        if trace["gentle"]["state"] not in {"selected", "exited"}:
            trace["gentle"] = followup["gentle"]
    trace.update(need=need, lead=state.get("house_lead"), invalidated_initial=invalidated)
    for room, detail in details.items():
        detail["house_need"] = need
        if room in reasons:
            detail["status"] = reasons[room]
        if room in readings:
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
        initial = recovery_initial(room, mode, live, readings[room], state, ROOMS)
        if initial is None or matching_feedback(now, live[room], readings[room]["sample"], target):
            continue
        if initial != current and increases_output(mode, current, initial):
            if need != mode or any(live[peer]["adapter"]["mode"] not in {mode, "off", "fan_only"}
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
        if flag := final_gate(action, now, live, readings, connected, state, ROOMS):
            details[action[0]]["status"] = flag
            trace["rejections"].append([action, flag])
        else:
            return details, tuple(action)
    return details, None


def start_batch(state, room, desired, details, sent):
    member = {"phase": "queued", "target": details[room]["target_c"]}
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
    details, selection = assess(live, state, now, trial=trial, isolated=isolated, trace=trace)
    if before != encode(state):
        put_state(path, state)
    if selection is None or now >= end-30:
        audit("cycle", outcome="idle" if selection is None else "permit_expiring", assessment=trace, rooms=details)
        publish("实机持续试点运行中" if trial else "持续实机控制运行中", details)
        return

    room, kind, value, purpose = selection
    refreshed = {code: room_feedback(call, code) for code in ROOMS}
    verification = {}
    checked, confirmed = assess(refreshed, state, time(), trial=trial, isolated=isolated, trace=verification)
    if selection != confirmed or read_permit(time()) != permit:
        if selection == confirmed:
            ended = cancel_gentle(state, time(), "permit_inactive")
            if ended:
                verification["gentle_exit"] = ended
        put_state(path, state)
        audit("cycle", outcome="recheck_changed", assessment=trace, verification=verification,
              proposed=selection, final=None, rooms=checked)
        publish("发令前条件变化，本轮维持原设置", checked)
        return
    previous = list(refreshed[room]["signature"][:3])
    sent = time()
    if sent >= end-25:
        audit("cycle", outcome="permit_expiring", assessment=trace, verification=verification, rooms=checked)
        return
    batch = None
    if purpose in {"energy_start", "reversal_initial", "initial_safety_easing"}:
        batch = state["transitions"].get(group(room))
        if batch is None and purpose == "energy_start":
            batch = start_batch(state, room, value, checked, sent)
    action = {"room": room, "kind": kind, "value": value, "before": previous, "sent_at": sent, "status": "prepared",
              "purpose": purpose, "transaction": batch["id"] if batch else None, "target_at_send": checked[room]["target_c"],
              "expected": [previous[0], value, previous[2]] if kind == "temperature" else [value],
              "before_room_c": numeric(refreshed[room]["temperature"]),
              "before_watts": numeric(refreshed[room]["adapter"]["power"]),
              "before_mode_at": refreshed[room]["adapter"]["last_mode_change"], "followup": {}, "physical_verified": False}
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
    publish("已记录一条米家实机指令，等待伴侣回报", checked)


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
