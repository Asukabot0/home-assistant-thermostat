"""Track the audible command interval from recorded evidence."""
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple


BEEP_INTERVAL = 30 * 60
SHARED_ROOMS = ("living", "study")
JOURNALS = Path("/pilot")
TRIALS_SCHEMA = "CREATE TABLE IF NOT EXISTS trials (id TEXT PRIMARY KEY, room TEXT, sent_at REAL, value TEXT)"
REVERSAL_EASING = "reversal_easing"


class Beep(NamedTuple):
    at: float
    source: str


def record_beep(record):
    """Distinguish a witnessed beep from the estimates used for one-way IR."""
    if record.get("beeps_observed") == 0:
        return None
    sent = record["sent_at"]
    if record.get("beep_at") is not None:
        return Beep(record["beep_at"], "现场听到")
    relay = record.get("relay")
    delay = record.get("relay_delay_s")
    if relay == "accepted" and delay is not None:
        return Beep(sent + delay, "按米家指令回执估算")
    if (relay in {"accepted", "timeout", "unconfirmed"} or record.get("physical") == "matched"
            or (record.get("beeps_observed") or 0) > 0):
        return Beep(sent, "按指令发送时刻估算")
    return None


def journal_actions(journals, since, now):
    """Recover possible beeps if a process stopped before it saved the receipt."""
    for path in journals.glob("continuous-*.json"):
        if not path.stem.removeprefix("continuous-").isdigit():
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        for action in state["commands"]:
            sent = action["sent_at"]
            status = action["status"]
            if not since <= sent <= now or status not in {"prepared", "accepted", "proxy_echo", "unconfirmed"}:
                continue
            received = action.get("relay_at")
            accepted = status in {"accepted", "proxy_echo"}
            yield {
                "room": action["room"], "sent_at": sent,
                "relay": "accepted" if accepted else "unconfirmed",
                "relay_delay_s": received - sent if received is not None else None,
                "purpose": action.get("purpose"),
            }


def latest_beeps(db, now, journals=JOURNALS, *, for_reversal=False, shared_groups=None):
    """Find the last beep, with an exception only for the same room's easing step."""
    since = now - 86400
    records = {(r["room"], round(r["sent_at"], 3)): r for r in journal_actions(journals, since, now)}
    for path in [journals / "first-adjustment.json", *journals.glob("receipt-*.json")]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        if "sent_at" not in record:  # Existing heating-pilot journal.
            sent = record["at"]
            received = record.get("relay_at")
            record = {**record, "sent_at": sent, "relay": record["status"],
                      "relay_delay_s": received - sent if received is not None else None}
        if since <= record["sent_at"] <= now:
            key = record["room"], round(record["sent_at"], 3)
            records[key] = {**records.get(key, {}), **record}
    for room, sent, value in db.execute("SELECT room,sent_at,value FROM trials WHERE sent_at BETWEEN ? AND ?", (since, now)):
        recorded = json.loads(value)
        key = room, round(sent, 3)
        if (key not in records or recorded.get("beeps_observed") == 0 or recorded.get("beep_at") is not None
                or recorded.get("relay_delay_s") is not None or record_beep(records[key]) is None):
            purpose = records.get(key, {}).get("purpose")
            records[key] = {"purpose": purpose, **recorded}
    by_room = {}
    reversal = {}
    for (room, _), record in records.items():
        event = record_beep(record)
        if event is None or event.at > now:
            continue
        if room not in by_room or event.at > by_room[room].at:
            by_room[room] = event
        if record.get("purpose") != REVERSAL_EASING or record.get("relay") != "accepted":
            if room not in reversal or event.at > reversal[room].at:
                reversal[room] = event
    groups = (SHARED_ROOMS,) if shared_groups is None else tuple(g for g in shared_groups if len(g) >= 2)
    if for_reversal:
        for group in groups:
            for room in group:
                for peer in group:
                    if peer != room and peer in by_room and (room not in reversal or by_room[peer].at > reversal[room].at):
                        reversal[room] = by_room[peer]
        return reversal
    for group in groups:
        shared = max((by_room[room] for room in group if room in by_room), key=lambda event: event.at, default=None)
        if shared is not None:
            by_room.update(dict.fromkeys(group, shared))
    return by_room


def save_receipt(folder, room, sent, status, received, *, purpose=None):
    """Expose a completed manual trial dispatch to the controller immediately."""
    if status not in {"accepted", "timeout", "unconfirmed"}:
        return
    record = {"room": room, "sent_at": sent, "relay": status, "relay_delay_s": received - sent}
    if purpose is not None:
        record["purpose"] = purpose
    target = folder / f"receipt-{room}-{sent:.6f}.json"
    with NamedTemporaryFile(mode="w", dir=folder, delete=False, encoding="utf-8") as out:
        json.dump(record, out)
        out.flush()
        os.fsync(out.fileno())
    Path(out.name).replace(target)
