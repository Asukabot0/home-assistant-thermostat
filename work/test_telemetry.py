"""Check evidence integrity, privacy, bounded integrations and durable delivery."""
import base64
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
        snap["summary"]["window"]["controller_errors"] = {"cycle_error": 1}
        fixed,_=evidence.bundle((0,60),"state",snap,None,"verified-previous.zip")
        self.assertNotEqual(first["logical_sha256"],fixed["logical_sha256"])
        self.assertEqual(fixed["supersedes"],"verified-previous.zip")
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

    def test_index_distinguishes_capture_commit_from_historical_window_versions(self):
        service=worker.GitHub({"gh":"fixture","repo":"owner/repo"})
        service.api=Mock(side_effect=[{"ref":"refs/heads/telemetry"},None,{}])
        old="a"*40
        new="b"*40
        span={"start":0,"end":3600,"version":{"commit":old}}
        service.index([{"kind":"hourly","revision":1,"start":0,"end":3600,"commit":new,
                        "observed_versions":evidence.observed_versions({"by_version":[span]})}],new)
        payload=service.api.call_args.kwargs["body"]
        row=json.loads(base64.b64decode(payload["content"]))["latest"]
        self.assertEqual(row["capture_commit"],new)
        self.assertEqual(row["observed_versions"][0]["commit"],old)

    def test_schedule_uses_complete_utc_hours_and_waits_until_five(self):
        self.assertEqual(worker.last_full_end(3600+299),0)
        self.assertEqual(worker.last_full_end(3600+300),3600)
        self.assertEqual(worker.next_time(3600+300),7200+300)

    def test_public_projection_keeps_season(self):
        public=evidence.PublicSchema("a"*64,{},{})
        out=public.audit({"event":"cycle","at":1,"assessment":{"season":{"mode":"cool","temperature_c":29.5,"at":100}}})
        self.assertEqual(out["assessment"]["season"],{"mode":"cool","temperature_c":29.5,"at":100})

    def test_collector_reconciles_errors_in_each_version_and_rolling_window(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"pilot-data").mkdir()
            (root/"shadow").mkdir()
            fixture={"rooms":{},"topology":{"sensors":{},"zones":{}},"policy":{"tracking_band":0.3}}
            for file,key in (("rooms.json","rooms"),("topology.json","topology"),("house_policy.json","policy")):
                (root/"shadow"/file).write_text(json.dumps(fixture[key]))
            a,b="a"*64,"b"*64
            first=dict(fixture,policy={"tracking_band":0.4})
            for ref,data in ((a,first),(b,fixture)):
                (root/"pilot-data"/f"configuration-{ref}.json").write_text(json.dumps(data))
            timepoints=[(60,"startup",a),(65,"cycle_error",a),(125,"evidence_error",a),
                        (160,"startup",b),(160,"startup_error",b),(170,"cycle_error",b)]
            data=[{"event":e,"at":t,"commit":v[:40],"config_sha256":v,
                   "error":"OperationalError" if e.endswith("error") else None} for t,e,v in timepoints]
            (root/"pilot-data/live-audit.jsonl").write_bytes(b"".join(evidence.json_bytes(x) for x in data))
            snap=evidence.collect(root,61,240,{"status":"unknown"},"a"*64)
        stat=snap["summary"]
        totals={"cycle_error":2,"evidence_error":1,"startup_error":1}
        self.assertEqual(stat["window"]["controller_errors"],totals)
        for section in (stat,stat["rolling_24h"]):
            first,second=section["by_version"]
            self.assertEqual(first["statistics"]["controller_errors"],{"cycle_error":1,"evidence_error":1})
            self.assertEqual(second["statistics"]["controller_errors"],{"cycle_error":1,"startup_error":1})
            self.assertIn("0.4",first["statistics"]["joint_temperature_pct"])
            self.assertIn("0.3",second["statistics"]["joint_temperature_pct"])
        self.assertEqual(stat["rolling_24h"]["overall"]["controller_errors"],totals)
        self.assertIsNone(stat["window"]["joint_kwh"])
        self.assertIsNone(stat["window"]["controller_error_total"])

    def test_unhealthy_runtime_collects_available_audit_even_if_configuration_is_missing(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"pilot-data").mkdir()
            (root/"pilot-data/live-audit.jsonl").write_bytes(evidence.json_bytes({"event":"cycle_error","error":"OSError","at":100}))
            data=evidence.collect(root,61,240,{"status":"unknown","problems":["pilot_stopped"]},"a"*64)
            manifest,files=evidence.bundle((61,240),"diagnostic",data,None,None)
        self.assertEqual(data["watermarks"]["config"]["status"],"unknown")
        self.assertEqual(manifest["version"]["status"],"unknown")
        self.assertIn(b"cycle_error",files["raw/live-audit.jsonl"])
        self.assertIsNone(data["summary"]["window"]["joint_kwh"])

    def test_pending_delivery_precedes_stopped_or_mismatched_runtime(self):
        for problem in ("pilot_stopped","shadow_mismatch","startup_missing_for_current_process"):
            with self.subTest(problem=problem), TemporaryDirectory() as tmp:
                root=Path(tmp)
                (root/"spool").mkdir()
                record={"file":"prior.zip","start":0,"end":3600,"kind":"hourly","sha256":"fixture","commit":"a"*40,"advance":True,
                        "summary_revision":evidence.SUMMARY_REVISION}
                state={"pending":record,"uploaded":[],"cursor":0}
                gh=Mock()
                gh.upload.return_value={"verified_at":"ok","url":"https://github.com/fixture"}
                def observe(*args):
                    self.assertIsNone(state["pending"])
                    self.assertEqual(state["cursor"],3600)
                    self.assertEqual(gh.upload.call_count,1)
                    return {"status":"mismatch" if "mismatch" in problem else "unknown","problems":[problem],"startup":None}
                with patch.object(worker,"verify_zip"), patch.object(worker,"runtime",side_effect=observe), patch.object(worker,"package") as package:
                    worker.perform({"production":str(root),"local_bytes":1000},root,state,gh,3900)
                self.assertEqual(state["runtime_status"]["problems"],[problem])
                self.assertEqual(package.call_args_list[0].args[-1],"diagnostic")

    def test_runtime_classifies_stopped_hash_mismatch_and_missing_startup(self):
        config={"docker":"docker","pilot_container":"pilot","shadow_container":"shadow"}
        running={"status":"unverified","running":True,"container_started_at":evidence.iso(100),
                 "code_sha256":"a"*64,"config_sha256":"b"*64}
        start={"event":"startup","at":101,"commit":"c"*40,"code_sha256":"a"*64,"config_sha256":"b"*64}
        for pilot,shadow,rows,expected in ((running,{"status":"stopped","running":False},[start],"unknown"),
            (running,dict(running,code_sha256="d"*64),[start],"mismatch"),(running,running,[],"unknown")):
            with self.subTest(expected=expected), patch.object(worker,"container_runtime",side_effect=[dict(pilot),dict(shadow)]), patch.object(worker,"audit_rows",return_value=rows):
                self.assertEqual(worker.runtime(config,Path("."),140)["status"],expected)

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
