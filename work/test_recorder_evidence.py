"""Exercise durable power evidence through the real two-database shadow cycle."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "home-assistant/shadow"))
import service

ROOMS = ("bedroom", "living", "study")
TICK = 2_000_040


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


class RecorderEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        (self.root/".storage").mkdir()
        self.enterContext(patch.object(service, "SOURCE", self.root))
        self.enterContext(patch.object(service, "JOURNALS", self.root))
        self.recorder = self.enterContext(closing(sqlite3.connect(self.root/"home-assistant_v2.db")))
        self.rooms = self.enterContext(closing(sqlite3.connect(self.root/".storage/room_climate_control.sqlite3")))
        self.db = self.enterContext(closing(sqlite3.connect(self.root/"shadow.sqlite3")))
        self.db.row_factory = sqlite3.Row
        service.init(self.db, TICK-60)
        self.config = {}
        self.recorder.executescript("""
            CREATE TABLE states_meta(metadata_id INTEGER, entity_id TEXT);
            CREATE TABLE states(state_id INTEGER PRIMARY KEY, state TEXT, last_reported_ts REAL,
                                last_updated_ts REAL, metadata_id INTEGER, attributes_id INTEGER);
            CREATE TABLE state_attributes(attributes_id INTEGER PRIMARY KEY, shared_attrs TEXT);
        """)
        self.rooms.execute("""CREATE TABLE samples (room TEXT,tick INTEGER,watts REAL,adapter_mode TEXT,reported REAL,
                             temp_report INTEGER,measured REAL,desired REAL,outdoor REAL)""")
        self.boot = stamp(TICK-60)
        for i, room in enumerate(ROOMS, 1):
            power = f"sensor.{room}_power"
            self.config[room] = {"power": power, "power_fallback": None, "adapter": f"climate.{room}_unit",
                                 "reference": room, "verified_off_power": {"entity": power, "ceiling_w": 8}}
            self.recorder.executemany("INSERT INTO states_meta VALUES (?,?)",
                                      [(i, power), (i+10, f"climate.{room}_room_thermostat")])
            self.db.execute("INSERT INTO power_models VALUES (?,?,?,?)", (room, TICK, "calibrated",
                            json.dumps({"centers_w": [10, 250], "edges_w": [90]})))
        self.db.commit()
        self.recorder.commit()

    def samples(self, tick, watts):
        with self.rooms:
            self.rooms.executemany("INSERT INTO samples VALUES (?,?,?,'off',24,?,24,24,14)",
                                   [(room, tick, watts, tick) for room in ROOMS])

    def reports(self, tick, watts, *, changed=False, observer=True, power_stamp=None):
        report_at = tick+20 if power_stamp is None else power_stamp
        with self.recorder:
            for i, room in enumerate(ROOMS, 1):
                if changed:
                    self.recorder.execute("INSERT INTO states VALUES (NULL,?,?,?,?,NULL)",
                                          (watts, report_at, report_at, i))
                if observer:
                    attrs = {"power_entity_id": self.config[room]["power"], "power_observer_started_at": self.boot,
                             "temperature_entity_id": f"sensor.{room}_temp", "power_reported_at": stamp(report_at)}
                    key = self.recorder.execute("INSERT INTO state_attributes(shared_attrs) VALUES (?)",
                                                (json.dumps(attrs),)).lastrowid
                    self.recorder.execute("INSERT INTO states VALUES (NULL,'auto',?,?,?,?)",
                                          (tick+59, tick+59, i+10, key))

    def cycle(self, tick, age=65):
        return service.cycle(self.db, self.config, tick, release_commit="test", now=tick+age)

    def value(self, tick, room="living"):
        r = self.db.execute("SELECT value FROM decisions WHERE room=? AND tick=?", (room,tick)).fetchone()
        return json.loads(r[0]) if r else None

    def seed_low(self):
        for i in range(6):
            tick = TICK+i*60
            self.samples(tick, 1)
            self.reports(tick, "1", changed=i==0)
            self.assertTrue(self.cycle(tick))
        self.assertEqual(self.value(TICK+300)["low_power_seconds"], 240)

    def test_240_to_300_survives_recorder_rewriting_same_value_timestamp(self):
        self.seed_low()
        with self.recorder:
            self.recorder.execute("UPDATE states SET last_reported_ts=? WHERE metadata_id<=3", (TICK+320,))
        tick = TICK+360
        self.samples(tick, 2)
        self.reports(tick, "2", changed=True)
        self.assertTrue(self.cycle(tick))
        self.assertEqual(self.value(tick)["low_power_seconds"], 300)
        proof = self.db.execute("SELECT report_at,watts FROM power_reports WHERE source='sensor.living_power' ORDER BY report_at").fetchall()
        self.assertEqual([row["watts"] for row in proof], [1,1,1,1,1,1,2])
        self.assertEqual(len({row["report_at"] for row in proof}), 7)

    def test_late_observer_fills_once_and_preserves_existing_low_power_segment(self):
        self.seed_low()
        tick = TICK+360
        self.samples(tick, 1)
        self.reports(tick, "1", observer=False)
        before = self.db.execute("SELECT count(*) FROM power_reports").fetchone()[0]
        self.assertFalse(self.cycle(tick))
        self.assertIsNone(self.value(tick))
        self.assertEqual(self.db.execute("SELECT count(*) FROM power_reports").fetchone()[0], before)
        self.reports(tick, "1")
        # Recorder has already coalesced the next same-valued report into its row.
        with self.recorder:
            self.recorder.execute("UPDATE states SET last_reported_ts=? WHERE metadata_id<=3", (tick+80,))
        self.assertTrue(self.cycle(tick, age=85))
        self.assertEqual(self.value(tick)["power_evidence_status"], "ready")
        self.assertEqual(self.value(tick)["low_power_seconds"], 300)
        self.assertFalse(self.cycle(tick, age=95))
        self.assertEqual(self.db.execute("SELECT count(*) FROM power_reports").fetchone()[0], before+3)
        self.assertEqual(self.db.execute("SELECT count(*) FROM decisions WHERE tick=?", (tick,)).fetchone()[0], 3)
        # The shadow cycle has no dispatch path or command journal.
        self.assertFalse(list(self.root.glob("continuous-*.json")))
        self.samples(tick+60, 1)
        self.reports(tick+60, "1")
        self.assertTrue(self.cycle(tick+60))
        self.assertEqual(self.value(tick+60)["low_power_seconds"], 360)

    def test_missing_observer_is_distinct_from_invalid_report_and_gaps(self):
        self.seed_low()
        tick = TICK+360
        self.samples(tick, None)
        self.reports(tick, "unavailable", changed=True)
        self.assertTrue(self.cycle(tick))
        self.assertEqual(self.value(tick)["power_evidence_status"], "no_fresh_report")
        self.assertIsNone(self.value(tick)["power_evidence"])
        self.assertEqual(self.db.execute("SELECT watts FROM power_reports WHERE source=? AND report_at=?",
                                        ("sensor.living_power", tick+20)).fetchone()[0], None)
        next_tick = tick+180
        self.samples(next_tick, 1)
        self.reports(next_tick, "1", changed=True, observer=False)
        self.assertTrue(self.cycle(next_tick, age=151))
        self.assertEqual(self.value(next_tick)["power_evidence_status"], "recorder_snapshot_missing")
        self.assertIsNone(self.value(next_tick)["power_evidence"])
        self.samples(next_tick+60, 1)
        self.reports(next_tick+60, "1")
        self.assertTrue(self.cycle(next_tick+60))
        self.assertEqual(self.value(next_tick+60)["low_power_seconds"], 0)

    def test_pending_changed_watts_does_not_contaminate_pair_ledger(self):
        self.seed_low()
        tick = TICK+360
        self.samples(tick, 2)
        self.reports(tick, "2")
        self.assertFalse(self.cycle(tick))
        self.assertIsNone(self.value(tick))
        self.reports(tick, "2", changed=True, observer=False)
        self.assertTrue(self.cycle(tick, age=80))
        self.assertEqual(self.value(tick)["low_power_seconds"], 300)
        self.assertEqual(self.value(tick)["power_report_w"], 2)


if __name__ == "__main__":
    unittest.main()
