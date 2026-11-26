"""Exercise the Recorder cutover using an isolated SQLite fixture."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "home-assistant/shadow"))
from model import calibrate
from service import (HISTORY, init, latest_model, load_room, low_power_span, power_evidence, powers,
                     recent_power_reports, reversal_low_cutoff)


PRIMARY = "sensor.local_power"
FALLBACK = "sensor.official_power"


class PowerHistoryCutoverTests(unittest.TestCase):
    def setUp(self):
        self.recorder = self.enterContext(closing(sqlite3.connect(":memory:")))
        self.recorder.row_factory = sqlite3.Row
        self.recorder.executescript("""
            CREATE TABLE states_meta(metadata_id INTEGER, entity_id TEXT);
            CREATE TABLE states(state_id INTEGER PRIMARY KEY, state TEXT, last_reported_ts REAL,
                                last_updated_ts REAL, metadata_id INTEGER, attributes_id INTEGER);
            CREATE TABLE state_attributes(attributes_id INTEGER, shared_attrs TEXT);
        """)
        self.recorder.executemany("INSERT INTO states_meta VALUES (?,?)", ((1, PRIMARY), (2, FALLBACK)))

    def add_points(self, entity, points):
        self.recorder.executemany(
            "INSERT INTO states VALUES (NULL,?,?,?,?,NULL)",
            ((str(watts), stamp, stamp, entity) for stamp, watts in points),
        )

    def test_only_fallback_before_first_valid_primary(self):
        self.add_points(1, [(100, "unavailable"), (110, -1), (160, 25), (180, "unknown")])
        self.add_points(2, [(100, 10), (150, 20), (160, 999), (175, 999)])
        self.assertEqual(powers(self.recorder, PRIMARY, 90, end=200, fallback=FALLBACK),
                         [(100, 10), (150, 20), (160, 25), (180, None)])
        self.assertEqual(powers(self.recorder, PRIMARY, 90, end=150, fallback=FALLBACK), [(100, 10)])

    def test_primary_older_than_window_does_not_mix_sources(self):
        self.add_points(1, [(60, 10), (160, 25)])
        self.add_points(2, [(100, 99), (130, 99), (170, 99)])
        self.assertEqual(powers(self.recorder, PRIMARY, 90, end=200, fallback=FALLBACK), [(160, 25)])

    def test_newer_local_poll_cannot_replace_existing_observer_evidence(self):
        observed = 1_800_000_000.25
        recorded = observed + 0.5e-6
        self.add_points(1, [(recorded, 25), (observed + 15, 90)])
        self.assertEqual(recent_power_reports(self.recorder, PRIMARY, "climate.room", observed - 60,
                                              observed + 60, as_of=observed), [(recorded, 25)])

    def test_same_value_new_reports_are_recovered_only_for_the_matching_source_and_epoch(self):
        self.add_points(1, [(100, 25)])
        self.recorder.execute("INSERT INTO states_meta VALUES (3, 'climate.room')")
        epoch = datetime.fromtimestamp(10, timezone.utc).isoformat()
        for key, report, source, boot in ((1, 160, PRIMARY, epoch), (2, 160, PRIMARY, epoch),
                                           (3, 220, FALLBACK, epoch), (4, 230, PRIMARY, "old")):
            attrs = json.dumps({"power_entity_id": source, "power_observer_started_at": boot,
                                "power_reported_at": datetime.fromtimestamp(report, timezone.utc).isoformat()})
            self.recorder.execute("INSERT INTO state_attributes VALUES (?,?)", (key, attrs))
            self.recorder.execute("INSERT INTO states VALUES (NULL,'heat',?,?,3,?)", (report+1, report+1, key))
        self.assertEqual(powers(self.recorder, PRIMARY, 90, end=250), [(100, 25)])
        reports = recent_power_reports(self.recorder, PRIMARY, "climate.room", 90, 250, context=[PRIMARY, epoch])
        self.assertEqual(reports, [(100, 25), (160, 25)])

    def test_shadow_uses_same_value_reports_and_restarts_proof_after_observer_boot(self):
        tick0 = HISTORY + 9000
        config = {"power": PRIMARY, "power_fallback": FALLBACK, "adapter": "climate.adapter", "reference": "test"}
        self.add_points(1, [(tick0+20, 220)])
        self.recorder.execute("INSERT INTO states_meta VALUES (3, 'climate.bedroom_room_thermostat')")
        with closing(sqlite3.connect(":memory:")) as db, closing(sqlite3.connect(":memory:")) as rooms:
            db.row_factory = rooms.row_factory = sqlite3.Row
            init(db, tick0)
            db.execute("INSERT INTO power_models VALUES (?,?,?,?)", ("bedroom", tick0, "calibrated",
                       json.dumps({"centers_w": [12, 250], "edges_w": [90]})))
            db.execute("INSERT INTO settings VALUES (?,?)", ("power_model_source:bedroom", json.dumps([PRIMARY, FALLBACK])))
            rooms.execute("""CREATE TABLE samples (room TEXT,tick INTEGER,watts REAL,adapter_mode TEXT,reported REAL,
                             temp_report INTEGER,measured REAL,desired REAL,outdoor REAL)""")
            for i in range(6):
                tick = tick0 + i*60
                boot = tick0 if i < 5 else tick
                attrs = {"power_entity_id": PRIMARY,
                         "power_observer_started_at": datetime.fromtimestamp(boot, timezone.utc).isoformat(),
                         "power_reported_at": datetime.fromtimestamp(tick+20, timezone.utc).isoformat()}
                self.recorder.execute("INSERT INTO state_attributes VALUES (?,?)", (i, json.dumps(attrs)))
                self.recorder.execute("INSERT INTO states VALUES (NULL,'auto',?,?,3,?)", (tick+30, tick+30, i))
                rooms.execute("INSERT INTO samples VALUES ('bedroom',?,220,'heat',24,?,24.5,24,14)", (tick, tick))
                value = load_room(db, rooms, self.recorder, "bedroom", config, tick, None, None)
                self.assertEqual(value["above_high_seconds"], max(0, i-1)*60 if i < 5 else 0)
                self.assertEqual(value["low_power_seconds"], 0)
                db.execute("INSERT INTO decisions VALUES ('bedroom',?,'1970-01-01',0,?)", (tick, json.dumps(value)))

    def test_fresh_cutover_immediately_recalibrates_from_joined_history(self):
        tick = HISTORY + 9000
        loads = [12, 150, 480]
        self.add_points(2, [(tick - 7200 + 100 * i, loads[i % 3]) for i in range(72)])
        self.add_points(1, [(tick - 180, 12), (tick - 120, 150), (tick - 60, 480)])
        self.assertIsNone(calibrate(powers(self.recorder, PRIMARY, tick - HISTORY)))

        with closing(sqlite3.connect(":memory:")) as db:
            db.row_factory = sqlite3.Row
            init(db, tick - 60)
            old_model, old_tick = latest_model(db, "bedroom", tick - 60, self.recorder, FALLBACK, True)
            self.assertEqual(latest_model(db, "bedroom", tick, self.recorder, PRIMARY, False, fallback=FALLBACK),
                             (old_model, old_tick))
            new_model, trained = latest_model(db, "bedroom", tick, self.recorder, PRIMARY, True, fallback=FALLBACK)
            self.assertEqual(trained, tick)
            self.assertEqual(low_power_span(tick, [], new_model)[0], low_power_span(tick, [], old_model)[0])
            self.assertEqual(new_model["edges_w"][-1], old_model["edges_w"][-1])
            self.assertEqual(db.execute("SELECT status FROM power_models WHERE trained=?", (tick,)).fetchone()[0],
                             "calibrated")
            self.assertEqual(latest_model(db, "bedroom", tick + 60, self.recorder, PRIMARY, True, fallback=FALLBACK),
                             (new_model, trained))


class PowerEvidenceTests(unittest.TestCase):
    context = [PRIMARY, "1970-01-01T00:00:00+00:00"]
    thresholds = {"low": 23, "above_low": 23, "high": 190}

    def evidence(self, at, reports, prior=None, *, thresholds=None, context=None):
        return power_evidence(at, reports, context or self.context, thresholds or self.thresholds, prior)

    def test_duration_uses_distinct_reports_and_real_elapsed_time(self):
        for band, watts, boundary in (("high", 220, 180), ("low", 12, 300)):
            with self.subTest(band=band):
                first = self.evidence(100, [(100, watts)])
                early = [(100 + seconds, watts) for seconds in range(0, boundary, 60)]
                for seconds in (boundary-1, boundary):
                    reports = early + [(100 + seconds, watts)]
                    result = self.evidence(100 + seconds, reports, first)
                    self.assertEqual(result["bands"][band]["seconds"], seconds)
                self.assertEqual(self.evidence(200, [(100, watts)]*4, first)["bands"][band]["seconds"], 0)

    def test_source_restart_gap_and_invalid_report_break_the_run(self):
        reports = [(t, 12) for t in range(100, 401, 60)]
        first = self.evidence(100, reports[:1])
        current = self.evidence(400, reports, first)
        self.assertEqual(current["bands"]["low"]["seconds"], 300)
        for context in ([FALLBACK, self.context[1]], [PRIMARY, "1970-01-01T00:05:00+00:00"]):
            self.assertEqual(self.evidence(400, reports, current, context=context)["bands"]["low"]["seconds"], 0)
        for interruption in ([], [(510, None)]):
            delayed = reports + interruption + [(530, 12), (590, 12)]
            result = self.evidence(590, delayed, current)
            self.assertEqual(result["bands"]["low"]["seconds"], 60)
        self.assertIsNone(self.evidence(521, reports, current)["last_report"])
        self.assertEqual(self.evidence(460, reports + [(460, None)], current)["bands"]["low"]["seconds"], 0)

    def test_reclassification_does_not_backfill_and_compatible_models_keep_credit(self):
        reports = [(t, 61) for t in range(100, 1961, 60)]
        first = self.evidence(100, reports[:1])
        old = self.evidence(1840, reports, first)
        new = self.evidence(1900, reports, old, thresholds={"low": 86.25, "above_low": 22, "high": 191})
        self.assertEqual(new["bands"]["low"]["seconds"], 0)
        self.assertEqual(new["bands"]["above_low"]["seconds"], 1800)
        self.assertEqual(self.evidence(1960, reports, new,
                                      thresholds={"low": 86.25, "above_low": 22, "high": 191})
                         ["bands"]["low"]["seconds"], 60)

    def test_lowest_cluster_is_not_off_evidence(self):
        actual = {"power": PRIMARY}
        self.assertIsNone(reversal_low_cutoff(actual, 86.25))
        actual["verified_off_power"] = {"entity": FALLBACK, "ceiling_w": 30}
        self.assertIsNone(reversal_low_cutoff(actual, 86.25))
        actual["verified_off_power"]["entity"] = PRIMARY
        self.assertEqual(reversal_low_cutoff(actual, 86.25), 30)
        self.assertEqual(reversal_low_cutoff(actual, 21.45), 21.45)


if __name__ == "__main__":
    unittest.main()
