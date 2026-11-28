"""Qualified steady episodes used to choose the next mode's starting temperature."""
import json
from statistics import median

from cadence import SHARED_ROOMS
from math import ceil, floor
from house_policy import POLICY
from model import clip, numeric, setpoint_bounds


EPISODE_SECONDS = 20 * 60
LEARNING_WINDOW = 7 * 86400
EPISODE_LIMIT = 20
SCHEMA = """
CREATE TABLE IF NOT EXISTS stable_episodes (
    room TEXT NOT NULL, mode TEXT NOT NULL, started INTEGER NOT NULL,
    ended INTEGER NOT NULL, offset REAL NOT NULL,
    PRIMARY KEY (room, mode, ended)
);
CREATE TABLE IF NOT EXISTS stable_episode_quality (
    room TEXT NOT NULL, mode TEXT NOT NULL, ended INTEGER NOT NULL,
    target REAL NOT NULL, reports INTEGER NOT NULL,
    PRIMARY KEY (room, mode, ended)
);
"""


def learning_session(journals, tick):
    """Read the active controller's durable journal without mutating it."""
    for name in ("live-permit.json", "continuous-permit.json"):
        try:
            permit = json.loads((journals / name).read_text(encoding="utf-8"))
            break
        except FileNotFoundError:
            continue
    else:
        return None
    start, end = permit["start"], permit["end"]
    if permit["enabled"] is not True or tick < start or (end is not None and tick + 59 >= end):
        return None
    try:
        session = json.loads((journals / f"continuous-{int(start)}.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    return dict(session, isolated_rooms=permit.get("isolated_rooms", [])) if session["start"] == start else None


def learning_active(session, room, virtual, now=None, *, lock_group=None):
    if session is None or virtual is None or virtual["state"] != "auto":
        return None
    attrs = json.loads(virtual["shared_attrs"] or "{}")
    group = lock_group if lock_group is not None else ("living_study" if room in SHARED_ROOMS else room)
    override = session.get("external_overrides", {}).get(room)
    if (room in session.get("isolated_rooms", ()) or room in session.get("manual_holds", {})
            or override is not None and (now is None or now < override["until"])
            or room in session["holds"] or group in session["holds"]
            or attrs.get("preset_mode") != "标准" or attrs.get("control_hold_latched", False)):
        return None
    return session["start"]


def action_responded(action, commands):
    if action["status"] != "proxy_echo":
        return False
    if action.get("purpose") == "reversal_off":
        # The following mode action owns the response window for an internal stop.
        return True
    before = numeric(action.get("before_watts"))
    if action["kind"] == "mode":
        mode, setting = action["value"], None
        stronger = mode != "off"
    elif action["kind"] == "temperature":
        mode, old, _ = action["before"]
        setting = action["value"]
        if action.get("purpose") == "reversal_initial":
            mode_action = next((cmd for cmd in reversed(commands) if cmd["room"] == action["room"]
                                and cmd["kind"] == "mode" and cmd["value"] == mode
                                and 0 <= action["sent_at"]-cmd["sent_at"] <= 180), None)
            before = numeric(mode_action.get("before_watts")) if mode_action else None
            stronger = True
        else:
            stronger = setting > old if mode == "heat" else setting < old
    else:
        return False
    if before is None:
        return False
    for point in action.get("followup", {}).values():
        power = numeric(point.get("watts"))
        age = numeric(point.get("power_age_s"))
        reported = point.get("reported")
        if (power is not None and age is not None and reported
                and 0 <= age <= min(600, point["at"] - action["sent_at"])
                and reported[0] == mode and (setting is None or reported[1] == setting)
                and (power > before if stronger else power < before)):
            return True
    return False


def record_episode(db, room, tick, history, session, active_since, power_span):
    """Insert one full, non-overlapping 20-minute episode when the end is observed."""
    if active_since is None or power_span < EPISODE_SECONDS:
        return
    start = tick - EPISODE_SECONDS
    window = history[-(EPISODE_SECONDS//60+1):]
    if not window:
        return
    mode, setpoint = window[-1]["adapter_mode"], window[-1]["reported"]
    target = numeric(window[-1]["desired"])
    if (mode not in {"heat", "cool"} or setpoint is None or target is None
            or [row["tick"] for row in window] != list(range(start, tick+1, 60))):
        return
    previous = db.execute("SELECT max(ended) FROM stable_episodes WHERE room=? AND mode=?", (room, mode)).fetchone()[0]
    if previous is not None and start < previous:
        return
    if any(row["adapter_mode"] != mode or row["reported"] != setpoint or row["measured"] is None
           or row["desired"] != target
           or abs(row["measured"]-target) > POLICY["ideal_band"] + 1e-6 or row["temp_report"] is None
           or not -59 <= row["tick"]-row["temp_report"] <= 1200 for row in window):
        return
    reports = sorted({row["temp_report"]: row["measured"] for row in window
                      if start <= row["temp_report"] <= tick+59}.items())
    if len(reports) < 2:
        return
    first, last = reports[0], reports[-1]
    if (first[0]-start > 300 or tick-last[0] > 300
            or any(b[0]-a[0] > POLICY["sparse_report_age_seconds"] for a,b in zip(reports,reports[1:]))
            or abs(last[1]-first[1]) > .2+1e-6
            or max(v for _,v in reports)-min(v for _,v in reports) > .3+1e-6):
        return
    prior = db.execute(
        "SELECT tick,value FROM decisions WHERE room=? AND tick>=? AND tick<? ORDER BY tick", (room, start, tick)
    ).fetchall()
    if ([row[0] for row in prior] != list(range(start, tick, 60))
            or any(json.loads(row[1]).get("learning_session") != active_since for row in prior)):
        return
    commands = session["commands"]
    if any(action["room"] == room and start-900 <= action["sent_at"] <= tick+59
           and action["status"] != "canceled" and not action_responded(action, commands) for action in commands):
        return
    offset = clip(median(setpoint - measured for _, measured in reports), -4, 4)
    db.execute("INSERT OR IGNORE INTO stable_episodes VALUES (?,?,?,?,?)", (room, mode, start, tick, offset))
    db.execute("INSERT OR IGNORE INTO stable_episode_quality VALUES (?,?,?,?,?)", (room, mode, tick, target, len(reports)))


def learned_offset(db, room, mode, tick):
    offsets = db.execute(
        """SELECT offset FROM stable_episodes JOIN stable_episode_quality USING(room,mode,ended)
           WHERE room=? AND mode=? AND ended BETWEEN ? AND ?
           ORDER BY ended DESC LIMIT ?""",
        (room, mode, tick-LEARNING_WINDOW, tick, EPISODE_LIMIT)
    ).fetchall()
    if not offsets:
        return None
    return {"offset_c": median(row[0] for row in offsets), "episodes": len(offsets),
            "confidence": "实测稳态片段", "verified_physical_setpoint": False}


def initial_setpoint(db, room, mode, tick, target, bounds):
    bounds = setpoint_bounds(bounds)
    target = numeric(target)
    if bounds is None or target is None or ceil(bounds[0]) > floor(bounds[1]):
        return None
    estimate = learned_offset(db, room, mode, tick) if room != "study" else None
    setting = round(target + estimate["offset_c"]) if estimate else round(target)
    if estimate and estimate["episodes"] <= 2:
        setting = clip(setting, ceil(target - 1), floor(target + 1))
    return int(clip(setting, ceil(bounds[0]), floor(bounds[1])))
