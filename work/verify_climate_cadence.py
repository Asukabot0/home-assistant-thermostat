"""Focused checks for cooldown exemptions and restart recovery."""
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "home-assistant/shadow"))

from cadence import REVERSAL_EASING, TRIALS_SCHEMA, latest_beeps, save_receipt


def check_easing_and_batch():
    with TemporaryDirectory() as directory, sqlite3.connect(":memory:") as db:
        journal = Path(directory)
        db.execute(TRIALS_SCHEMA)
        save_receipt(journal, "living", 100, "accepted", 101)
        save_receipt(journal, "living", 200, "accepted", 201, purpose=REVERSAL_EASING)
        normal = latest_beeps(db, 250, journal)
        reversal = latest_beeps(db, 250, journal, for_reversal=True)
        assert normal["study"].at == normal["living"].at == 201
        assert reversal["living"].at == 101
        assert reversal["study"].at == 201

        # An annotation supplies better timing and retains the automatic action's purpose.
        evidence = {"sent_at": 200, "beep_at": 200.5, "relay": "accepted"}
        db.execute("INSERT INTO trials VALUES (?,?,?,?)", ("manual", "living", 200, json.dumps(evidence)))
        assert latest_beeps(db, 250, journal)["living"].at == 200.5
        assert latest_beeps(db, 250, journal, for_reversal=True)["living"].at == 101

        save_receipt(journal, "living", 250, "accepted", 251, purpose="reversal_off")
        save_receipt(journal, "living", 500, "accepted", 501, purpose="reversal_mode")
        save_receipt(journal, "study", 510, "accepted", 511, purpose="reversal_mode")
        save_receipt(journal, "study", 520, "accepted", 521, purpose="reversal_initial")
        anchors = latest_beeps(db, 550, journal, for_reversal=True)
        assert latest_beeps(db, 550, journal)["living"].at == anchors["living"].at == anchors["study"].at == 521


def check_restart_receipt_recovery():
    with TemporaryDirectory() as directory, sqlite3.connect(":memory:") as db:
        journal = Path(directory)
        db.execute(TRIALS_SCHEMA)
        actions = [
            {"room": "living", "sent_at": 100, "status": "proxy_echo", "relay_at": 101, "purpose": REVERSAL_EASING},
            {"room": "bedroom", "sent_at": 300, "status": "prepared", "purpose": REVERSAL_EASING},
            {"room": "study", "sent_at": 310, "status": "canceled"},
        ]
        (journal / "continuous-100.json").write_text(json.dumps({"commands": actions}))
        ordinary = latest_beeps(db, 350, journal)
        reversal = latest_beeps(db, 350, journal, for_reversal=True)
        assert ordinary["living"].at == ordinary["study"].at == 101
        assert "living" not in reversal and reversal["study"].at == 101
        assert ordinary["bedroom"].at == reversal["bedroom"].at == 300


if __name__ == "__main__":
    check_easing_and_batch()
    check_restart_receipt_recovery()
    print("Verified own-room cooling exemptions, shared batch reanchoring, and receipt recovery")
