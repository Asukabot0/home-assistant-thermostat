"""Run low-frequency Xiaomi Home thermostat control with a durable journal."""
from collections import defaultdict
import json
from math import isfinite
import os
import signal
import sqlite3
from tempfile import NamedTemporaryFile
from threading import Event
from time import time
from urllib.error import HTTPError

from cadence import SHARED_ROOMS, save_receipt
from house_dispatch import acknowledged, coordinate, final_gate
from house_policy import ACTIVE, INDEPENDENT, POLICY
from heating_pilot import DATA_DIR, ha_client, publish
from model import increases_output, numeric, setpoint_bounds
from pilot_gate import ROOMS, matching_feedback, room_advice, room_feedback
from service import encode, local


PERMIT = DATA_DIR / "continuous-permit.json"
LIVE_PERMIT = DATA_DIR / "live-permit.json"
MAX_STEPS = 3
ECHO_TIMEOUT = 180
BATCH_SPACING = 5


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
                if action["purpose"] == "energy_off":
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


def assess(live, state, now, advise_room=room_advice, *, trial=True, isolated=()):
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
                         "status": hold or ("手动隔离" if room in isolated else
                                            "等待自动状态和有效数据" if not ready[room] else "等待全屋分配")}
    if pending:
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
    need, reasons, candidates = coordinate(live, readings, allocation_ready, state, now, ROOMS, connected)
    for room, detail in details.items():
        detail["house_need"] = need
        if room in reasons:
            detail["status"] = reasons[room]
        if room in readings:
            view = readings[room]["house_view"]
            detail["prediction_10m_c"] = view["trend"] if view is not None else None
    for room in sorted(incomplete):
        batch, member = members[room]
        if not ready[room] or now < batch["ready_at"] or need in {"unknown", "conflict"}:
            continue
        mode = live[room]["adapter"]["mode"]
        current = live[room]["adapter"]["setpoint"]
        target = live[room]["signature"][4]
        if mode != batch["mode"]:
            continue
        initial = readings[room]["initial_setpoints"].get(mode)
        if initial is None or matching_feedback(now, live[room], readings[room]["sample"], target):
            continue
        bounds = setpoint_bounds(ROOMS[room]["soft_limits"][mode])
        view = readings[room]["house_view"]
        stable = (need == "hold" and view and view["trend"] is not None
                  and abs(view["trend"]) < POLICY["minimum_progress_c"]
                  and abs(details[room]["room_c"]-target) <= POLICY["ideal_band"])
        if stable and bounds is not None and bounds[0] <= current <= bounds[1]:
            initial = current
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
            candidates.append((-1, 0, room, "temperature", initial, "reversal_initial"))
    for priority, ranking, *action in sorted(candidates):
        room, kind, _, purpose = action
        if (trial and kind == "temperature" and purpose != "reversal_initial"
                and sum(c["room"] == room and c["kind"] == "temperature" and c["status"] != "canceled"
                        for c in state["commands"]) >= MAX_STEPS):
            details[room]["status"] = "已达本次试点的设温次数限制"
            continue
        if flag := final_gate(action, now, live, readings, connected, state, ROOMS):
            details[action[0]]["status"] = flag
        else:
            return details, tuple(action)
    return details, None


def start_batch(state, room, desired, details, sent):
    member = {"phase": "queued", "target": details[room]["target_c"]}
    batch = {"id": sent, "mode": desired, "members": {room: member}, "ready_at": sent}
    state["transitions"][group(room)] = batch
    return batch


def cycle(call):
    permit = read_permit(time())
    if permit is None:
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
    details, selection = assess(live, state, now, trial=trial, isolated=isolated)
    if before != encode(state):
        put_state(path, state)
    if selection is None or now >= end-30:
        publish("实机持续试点运行中" if trial else "持续实机控制运行中", details)
        return

    room, kind, value, purpose = selection
    refreshed = {code: room_feedback(call, code) for code in ROOMS}
    checked, confirmed = assess(refreshed, state, time(), trial=trial, isolated=isolated)
    if selection != confirmed or read_permit(time()) != permit:
        put_state(path, state)
        publish("发令前条件变化，本轮维持原设置", checked)
        return
    previous = list(refreshed[room]["signature"][:3])
    sent = time()
    if sent >= end-25:
        return
    batch = None
    if purpose in {"energy_start", "reversal_initial"}:
        batch = state["transitions"].get(group(room))
        if batch is None:
            batch = start_batch(state, room, value, checked, sent)
    action = {"room": room, "kind": kind, "value": value, "before": previous, "sent_at": sent, "status": "prepared",
              "purpose": purpose, "transaction": batch["id"] if batch else None,
              "expected": [previous[0], value, previous[2]] if kind == "temperature" else [value],
              "before_room_c": numeric(refreshed[room]["temperature"]),
              "before_watts": numeric(refreshed[room]["adapter"]["power"]),
              "before_mode_at": refreshed[room]["adapter"]["last_mode_change"], "followup": {}, "physical_verified": False}
    state["commands"].append(action)
    put_state(path, state)
    if read_permit(time()) != permit or time() >= end-20:
        action.update(status="canceled")
        put_state(path, state)
        return
    service, field = ("set_temperature", "temperature") if kind == "temperature" else ("set_hvac_mode", "hvac_mode")
    try:
        code = call("services/climate/" + service, {"entity_id": ROOMS[room]["adapter"], field: value})
        action.update(status="accepted", http=code)
    except HTTPError as error:
        action.update(status="unconfirmed" if error.code >= 500 else "rejected", http=error.code)
        error.close()
    except OSError:
        action.update(status="unconfirmed")
    received = time()
    action["relay_at"] = received
    if action["status"] != "accepted":
        state["holds"][group(room)] = "米家调用失败或未及时返回，已暂停复发"
    put_state(path, state)
    save_receipt(DATA_DIR, room, sent, action["status"], received, purpose=purpose)
    publish("已记录一条米家实机指令，等待伴侣回报", checked)


def main():
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    call = ha_client()
    while not stop.is_set():
        try:
            cycle(call)
        except (ValueError, TypeError, KeyError, OSError, OverflowError, sqlite3.Error) as error:
            publish(f"数据或服务异常，暂停发令（{type(error).__name__}）")
        stop.wait(20)


if __name__ == "__main__":
    main()
