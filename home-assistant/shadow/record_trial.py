"""Save evidence for an already authorized physical trial without sending a command."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import sqlite3
from statistics import median
import time
from uuid import uuid4

from cadence import TRIALS_SCHEMA
from service import CONFIG, DATA, SOURCE, encode, open_readonly, powers, room_source, state_history


def instant(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("An ISO 8601 timestamp with a UTC offset is required.")
    return parsed.timestamp()


def span(points):
    return {"count": len(points), "median": round(median(points), 2),
            "min": round(min(points), 2), "max": round(max(points), 2)} if points else {"count": 0}


def collect(args, recorder, observer, config):
    start = args.sent_at - 600
    end = args.sent_at + args.followup_minutes * 60
    watts = [(at, w) for at, w in powers(recorder, config["power"], start) if w is not None and at <= end]
    temperatures = [(r["temp_report"], r["measured"]) for r in observer.execute(
        "SELECT DISTINCT temp_report, measured FROM samples WHERE room=? AND temp_report BETWEEN ? AND ? "
        "AND measured IS NOT NULL ORDER BY temp_report", (args.room, start, end)
    )]
    proxy = []
    for event in state_history(recorder, config["adapter"], start):
        if event["reported"] is None or event["reported"] > end:
            continue
        attrs = json.loads(event["shared_attrs"] or "{}")
        proxy.append({"at": event["reported"], "mode": event["state"], "setpoint": attrs.get("temperature")})
    if args.physical != "unknown":
        confidence = "人工确认执行" if args.physical == "matched" else "人工确认与指令不符"
    elif watts:
        confidence = "有功率观测，实际命令结果未确认"
    else:
        confidence = "未知"
    return {"room": args.room, "action": args.action, "request": args.request, "sent_at": args.sent_at,
            "relay": args.relay, "relay_delay_s": args.relay_at - args.sent_at if args.relay_at else None,
            "physical": args.physical, "physical_note": args.physical_note,
            "observed_delay_s": args.observed_at - args.sent_at if args.observed_at else None,
            "beeps_observed": args.beeps, "beep_at": getattr(args, "beep_at", None), "confidence": confidence,
            "power_before": span([w for at,w in watts if at < args.sent_at]),
            "power_after": span([w for at,w in watts if at >= args.sent_at]),
            "power_timeline": watts, "temperature_timeline": temperatures, "proxy_timeline": proxy,
            "followup_seconds": args.followup_minutes * 60,
            "completed_followup": end <= time.time()}


def presenter():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("room", choices=("bedroom", "living", "study"))
    p.add_argument("action", choices=("power-on", "power-off", "mode", "temperature"))
    p.add_argument("--request", required=True)
    p.add_argument("--sent-at", required=True, type=instant)
    p.add_argument("--relay", choices=("accepted", "rejected", "timeout", "unknown"), default="unknown")
    p.add_argument("--relay-at", type=instant)
    p.add_argument("--physical", choices=("matched", "not_matched", "unknown"), default="unknown")
    p.add_argument("--physical-note")
    p.add_argument("--observed-at", type=instant, help="Actual observation time, when known.")
    p.add_argument("--beeps", type=int, choices=range(0, 10))
    p.add_argument("--beep-at", type=instant, help="Time of the last firsthand heard beep for this command.")
    p.add_argument("--followup-minutes", type=int, choices=range(1, 61), default=15)
    p.add_argument("--id", help="Update this record after a later observation.")
    args = p.parse_args()
    if args.physical != "unknown" and (not args.physical_note or not args.physical_note.strip()):
        p.error("Physical verification requires a firsthand evidence note.")
    if args.beep_at is not None and args.beeps == 0:
        p.error("A heard beep cannot be recorded with a zero beep count.")
    evidence_times = (args.relay_at, args.observed_at, args.beep_at)
    if any(x < args.sent_at for x in evidence_times if x is not None):
        p.error("Evidence times cannot precede the command.")
    now = time.time()
    if any(x > now for x in (args.sent_at, *evidence_times) if x is not None):
        p.error("Command and evidence times must have already occurred.")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with closing(open_readonly(SOURCE/"home-assistant_v2.db")) as recorder, room_source(
        SOURCE/".storage/room_climate_control.sqlite3"
    ) as observer:
        result = collect(args, recorder, observer, config[args.room])
    key = args.id or uuid4().hex
    with closing(sqlite3.connect(DATA/"shadow.sqlite3", timeout=8)) as db, db:
        db.execute(TRIALS_SCHEMA)
        if args.id:
            current = db.execute("SELECT room,sent_at,value FROM trials WHERE id=?", (key,)).fetchone()
            if not current or current[:2] != (args.room, args.sent_at):
                p.error("An update must use the original room and command timestamp.")
            saved = json.loads(current[2])
            if args.beep_at is None and args.beeps != 0:
                result["beep_at"] = saved.get("beep_at")
            if args.beeps is None and not (args.beep_at is not None and saved.get("beeps_observed") == 0):
                result["beeps_observed"] = saved.get("beeps_observed")
        db.execute("INSERT INTO trials VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET value=excluded.value",
                   (key, args.room, args.sent_at, encode(result)))
    print(json.dumps(dict(result, id=key, recorded_at=datetime.now(timezone.utc).isoformat()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    presenter()
