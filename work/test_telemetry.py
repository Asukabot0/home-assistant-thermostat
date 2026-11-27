"""Check evidence integrity, privacy, bounded integrations and durable delivery."""
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"home-assistant/telemetry"))
import evidence
import worker


class TelemetryTests(unittest.TestCase):
    def test_sqlite_read_transaction_includes_uncheckpointed_wal(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/"source.db"
            source=sqlite3.connect(path)
            self.addCleanup(source.close)
            source.execute("PRAGMA journal_mode=WAL")
            source.execute("PRAGMA wal_autocheckpoint=0")
            source.execute("CREATE TABLE reports (stamp REAL)")
            source.execute("INSERT INTO reports VALUES (1234)")
            source.commit()
            with evidence.readonly(path) as snapshot:
                self.assertEqual(snapshot.execute("SELECT stamp FROM reports").fetchone()[0],1234)
                self.assertRaises(sqlite3.OperationalError,snapshot.execute,"DELETE FROM reports")

    def test_gaps_and_observer_epoch_never_extend_stale_power(self):
        points=[dict(report_at=t,watts=w,observer_epoch=e) for t,w,e in [(0,100,"1"),(60,100,"1"),(61,90,"2"),(121,90,"2"),(300,90,"2")] ]
        self.assertEqual(evidence.power_ranges(points,0,600),[(0,60,100),(61,121,90)])
        stat=evidence.summarize([],[],{},["zone1","zone2"],[],[],0,3600,0.3)
        self.assertEqual(stat["joint_coverage_pct"],0)
        self.assertIsNone(stat["joint_temperature_pct"]["0.3"])
        self.assertEqual(stat["minute_coverage"]["missing"],[[0,3600]])

    def test_redacted_archive_verification_and_identical_logical_hash(self):
        snap={"raw":{"live-audit":[{"event":"cycle","at":50}]},"runtime":{},"summary":{"window":{
            "joint_coverage_pct":0,"joint_temperature_minutes":0,"minute_coverage":{}}},"effective":{"observation_version":"v1"},
            "status":{"ok":True},"captured_at":60,"watermarks":{},"private_values":["sensor.fixture_living"]}
        first,files=evidence.bundle((0,60),"state",snap,None,None)
        snap["captured_at"]=65
        second,_=evidence.bundle((0,60),"state",snap,None,None)
        self.assertEqual(first["logical_sha256"],second["logical_sha256"])
        with TemporaryDirectory() as tmp:
            file=Path(tmp)/"artifact.zip"
            evidence.write_zip(file,files)
            digest=worker.sha256(file.read_bytes()).hexdigest()
            self.assertEqual(evidence.verify_zip(file,digest),first)
            self.assertRaises(ValueError,evidence.verify_zip,file,"wrong")
        snap["status"]={"unsafe":"sensor.fixture_living"}
        self.assertRaises(ValueError,evidence.bundle,(0,60),"state",snap,None,None)

    def test_pending_survives_network_failure_and_cursor_moves_after_remote_index(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"spool").mkdir()
            attempt={"file":"evidence-test.zip","start":0,"end":3600,"kind":"hourly","sha256":"dummy",
                     "commit":"a"*40,"advance":True}
            state={"cursor":0,"uploaded":[],"pending":attempt}
            gh=Mock()
            gh.upload.return_value={"verified_at":"ok"}
            gh.index.side_effect=worker.RemoteFailure("HTTP_503")
            with patch.object(worker,"verify_zip"),patch.object(worker,"sleep"):
                self.assertRaises(worker.RemoteFailure,worker.send_pending,root,state,gh)
                self.assertEqual(state["pending"],attempt)
                self.assertEqual(state["cursor"],0)
                gh.upload.return_value.update(url="https://github.com/example")
                gh.index.side_effect=None
                worker.send_pending(root,state,gh)
            self.assertIsNone(state["pending"])
            self.assertEqual(json.loads((root/"queue.json").read_bytes())["cursor"],3600)

    def test_schedule_uses_complete_utc_hours_and_waits_until_five(self):
        self.assertEqual(worker.last_full_end(3600+299),0)
        self.assertEqual(worker.last_full_end(3600+300),3600)
        self.assertEqual(worker.next_time(3600+300),7200+300)

    def test_public_projection_preserves_zone_and_removes_unrelated_fields(self):
        topo={"zones":{"area_private":{"primary":"s"}},"sensors":{"s":{"entity":"sensor.private"}}}
        public=evidence.PublicSchema("a"*64,{},topo)
        out=public.audit({"event":"cycle","at":1,"assessment":{"zones":{"area_private":{"measured":24,"target":24,
                         "reason":"near_target","source":["sensor.private", "ha_report"],"token":"leak"}}},"token":"leak"})
        blob=evidence.json_bytes(out)
        self.assertIn(b"near_target",blob)
        for secret in (b"area_private",b"sensor.private",b"leak",b"token"):
            self.assertNotIn(secret,blob)


if __name__=="__main__":
    unittest.main()
