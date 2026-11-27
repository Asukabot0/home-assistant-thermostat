"""Run a single witnessed heating adjustment behind a time-limited local permit."""
from datetime import datetime
import json
from math import isfinite
import os
from pathlib import Path
import signal
import sqlite3
from threading import Event
from time import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from pilot_gate import ROOMS, eligible, room_advice, room_feedback
from service import LOCAL, atomic, encode


DATA_DIR = Path("/pilot")
PERMIT = DATA_DIR / "permit.json"
ATTEMPT = DATA_DIR / "first-adjustment.json"
MANUAL_STOP = DATA_DIR / "manual-stop.json"
ACCESS = Path("/access/home-assistant.env")
API = os.getenv("ROOM_CLIMATE_HA_URL", "http://homeassistant:8123").rstrip("/") + "/api/"


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def ha_client():
    token = next(value.strip().strip('"\'') for key, _, value in
                 (line.partition("=") for line in ACCESS.read_text().splitlines())
                 if key == "HOMEASSISTANT_TOKEN" and value.strip())
    opener = build_opener(ProxyHandler({}), NoRedirects())

    def call(path, fields=None):
        body = None if fields is None else encode(fields).encode()
        request = Request(API + path, data=body, headers={"Authorization": "Bearer " + token,
                                                          "Content-Type": "application/json"})
        with opener.open(request, timeout=8 if body is None else 20) as response:
            return json.load(response) if body is None else response.status

    return call


def read_permit(now):
    try:
        raw = json.loads(PERMIT.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if raw.get("enabled") is not True:
        return None
    start = float(raw["start"])
    end = float(raw["end"])
    if not all(isfinite(item) for item in (start, end)) or not start <= now < end:
        return None
    local_start = datetime.fromtimestamp(start, LOCAL)
    quiet_start = local_start.replace(hour=23, minute=0, second=0, microsecond=0)
    if not 7 <= local_start.hour < 23 or end > quiet_start.timestamp():
        return None
    rooms = {room: tuple(signature) for room, signature in raw["rooms"].items()}
    if not rooms or set(rooms).difference(ROOMS) or any(
        len(s) != 7 or s[:2] != ("heat", 21) or s[3:] != ("heat", 24, "标准", False)
        for s in rooms.values()
    ):
        return None
    return (start, end, rooms)


def publish(status, rooms=None, *, control=None):
    now = datetime.now(LOCAL).isoformat(timespec="seconds")
    atomic(DATA_DIR / "status.json", encode({"at": now, "status": status, "rooms": rooms or {}, "control": control}) + "\n")


def reserve(record):
    with ATTEMPT.open("x", encoding="utf-8") as output:
        output.write(encode(record) + "\n")
        output.flush()
        os.fsync(output.fileno())


def assess(call, rooms):
    readings = {}
    selected = None
    feedback = {room: room_feedback(call, room) for room in rooms}
    for room, baseline in rooms.items():
        live = feedback[room]
        if live["signature"] != baseline:
            atomic(MANUAL_STOP, encode({"at": time(), "room": room}) + "\n")
            return {}, None
        reason, value = eligible(time(), baseline, live, room_advice(room))
        readings[room] = {"reason": reason, "setpoint": value}
        if selected is None and value is not None:
            selected = room, value
    return readings, selected


def cycle(call):
    permit = read_permit(time())
    if permit is None:
        call("states/climate.bedroom_room_thermostat")
        publish("授权时段已结束或试用已停用" if PERMIT.exists() else "待确认启用")
        return
    if MANUAL_STOP.exists():
        publish("已检测到手动操作，试用暂停")
        return
    if ATTEMPT.exists():
        result = json.loads(ATTEMPT.read_text(encoding="utf-8"))
        message = ("首条自动调节已现场确认，试用额度已用完" if result.get("physical_verified") is True
                   else "本次试用的一条发令额度已锁定，等待现场反馈")
        publish(message)
        return
    _, end, rooms = permit
    rooms = {room: rooms[room] for room in ROOMS if room in rooms}
    details, selected = assess(call, rooms)
    if selected is None:
        publish("试用中，等待合格的调温条件" if details else "已检测到手动操作，试用暂停", details)
        return
    room, value = selected
    now = time()
    if now >= end - 30 or read_permit(now) != permit:
        publish("试用授权即将结束或已变化", details)
        return
    confirmed, refreshed = assess(call, rooms)
    if refreshed != selected:
        publish("复核后不再符合发令条件", confirmed)
        return
    sent = time()
    if sent >= end - 25 or read_permit(sent) != permit:
        publish("复核后发现授权时段已变化", details)
        return
    record = {"at": sent, "room": room, "setpoint": value, "status": "prepared", "physical_verified": False}
    reserve(record)
    now = time()
    if now >= end - 20 or read_permit(now) != permit:
        record["status"] = "canceled"
        atomic(ATTEMPT, encode(record) + "\n")
        publish("发令前授权条件变化，已取消", details)
        return
    try:
        response = call("services/climate/set_temperature", {"entity_id": ROOMS[room]["adapter"], "temperature": value})
        record.update(status="accepted", http=response, relay_at=time())
    except HTTPError as error:
        record.update(status="rejected", http=error.code)
    except OSError:
        record.update(status="unconfirmed")
    atomic(ATTEMPT, encode(record) + "\n")
    publish("已锁定本次试用的一条发令额度，等待现场反馈", details)


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
