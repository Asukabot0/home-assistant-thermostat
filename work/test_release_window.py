"""Release-window metrics must ignore populated historical rows."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "home-assistant/shadow"))
import service


class ReleaseWindowTests(unittest.TestCase):
    def test_fresh_release_does_not_inherit_old_days_or_other_commit_minutes(self):
        with TemporaryDirectory() as tmp, closing(sqlite3.connect(":memory:")) as db:
            db.row_factory = sqlite3.Row
            service.init(db, 600000)
            commit = "test"
            deploy = 600005
            tick = 600120
            scores = {"measured": 24, "target": 24, "source_fresh": True, "comfort_half_band": .95,
                      "hypothetical_send": False, "metered_seconds": 60, "metered_kwh": .1,
                      "confidence": "推测", "prediction_15m_delta": None, "outdoor": None,
                      "activity": "fixture", "model": None, "reason": "fixture",
                      "mode": "hold", "send_after": None, "setpoint": None}
            db.execute("INSERT INTO settings VALUES ('policy_started_1', '10000')")
            for room in ("bedroom", "living", "study"):
                for at, revision, ref, measured in ((599940, 7, None, 24),
                         (600000, service.POLICY_REVISION, commit, 24),
                         (600060, service.POLICY_REVISION, "another", 24),
                         (tick, service.POLICY_REVISION, commit, 24.8)):
                    value = dict(scores, policy_revision=revision, release_commit=ref, measured=measured)
                    db.execute("INSERT INTO decisions VALUES (?,?,?,?,?)", (room,at,"day",0,json.dumps(value)))
            config = {room: {"name": room} for room in ("bedroom","living","study")}
            with patch.object(service, "OUTPUT", Path(tmp)), patch.object(service, "live_actuations", return_value={}):
                service.display(db, config, tick, release=(commit,deploy))
                result = json.loads((Path(tmp)/"shadow-status.json").read_text())
            self.assertEqual(result["assessment_completed_minutes"], 2)
            self.assertFalse(result["data_ready_for_24h_review"])
            self.assertEqual(result["rooms"]["bedroom"]["totals"]["minutes"], 1)
            self.assertEqual(result["rooms"]["bedroom"]["totals"]["within_comfort_pct"], 0)
            self.assertAlmostEqual(result["observed_home_kwh"], .3)
            self.assertEqual(result["independent_temperature_rooms"], ["bedroom", "living"])


if __name__ == "__main__":
    unittest.main()
