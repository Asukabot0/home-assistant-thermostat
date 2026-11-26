"""Persist observations and reversible model versions in a local SQLite file."""
from contextlib import closing
from datetime import datetime, timedelta
import json
from math import sqrt
from pathlib import Path
import sqlite3
from statistics import median

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    room TEXT, tick INTEGER, localday TEXT, measured REAL, humidity REAL,
    desired REAL, reported REAL, adapter_mode TEXT, observed TEXT, confidence TEXT,
    watts REAL, outdoor REAL, temp_report INTEGER, quality TEXT, verified_target REAL,
    PRIMARY KEY(room, tick)
);
CREATE TABLE IF NOT EXISTS commands (
    room TEXT, sent INTEGER, kind TEXT, value TEXT, status TEXT,
    completed INTEGER, PRIMARY KEY(room, sent)
);
CREATE TABLE IF NOT EXISTS models (
    room TEXT, mode TEXT, generation INTEGER, parent INTEGER,
    offset REAL, metrics TEXT, PRIMARY KEY(room,mode,generation)
);
CREATE TABLE IF NOT EXISTS daily (
    room TEXT, date TEXT, metrics TEXT, PRIMARY KEY(room,date)
);
CREATE TABLE IF NOT EXISTS maintenance (
    name TEXT PRIMARY KEY, completed TEXT
);
"""

class Telemetry:
    def __init__(self,path: Path):
        self.path=path

    def connect(self):
        connection=sqlite3.connect(self.path,timeout=8)
        connection.row_factory=sqlite3.Row
        return connection

    def initialize(self):
        with closing(self.connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            if "verified_target" not in {row["name"] for row in db.execute("PRAGMA table_info(samples)")}:
                db.execute("ALTER TABLE samples ADD COLUMN verified_target REAL")

    def checkpoint(self):
        with closing(self.connect()) as db:
            busy, _, _ = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if busy:
                raise RuntimeError("Room climate journal is busy")

    def record(self, observations):
        with closing(self.connect()) as db, db:
            db.executemany("INSERT OR REPLACE INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",observations)

    def command(self, room, timestamp, kind, value, status, completed=None):
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO commands VALUES(?,?,?,?,?,?) ON CONFLICT(room,sent) DO UPDATE SET status=excluded.status, completed=excluded.completed",
                       (room,timestamp,kind,str(value),status,completed))

    def last_command(self, room):
        with closing(self.connect()) as db:
            row = db.execute(
                "SELECT sent,kind,value,status FROM commands WHERE room=? AND status!='canceled' "
                "ORDER BY sent DESC LIMIT 1", (room,),
            ).fetchone()
        return dict(row) if row else None

    def current_models(self, room, defaults):
        with closing(self.connect()) as db:
            return {mode: dict(row) if (row:=db.execute(
                "SELECT offset,generation FROM models WHERE room=? AND mode=? ORDER BY generation DESC LIMIT 1",
                (room,mode)).fetchone()) else {"offset":baseline,"generation":None}
                for mode,baseline in defaults.items()}

    def save_manual_offsets(self, room, offsets, current):
        with closing(self.connect()) as db, db:
            for mode, offset in offsets.items():
                previous = current[mode]
                if offset == previous["offset"]:
                    continue
                parent = previous["generation"]
                metadata = {"source": "manual", "previous_offset": previous["offset"]}
                db.execute("INSERT INTO models VALUES(?,?,?,?,?,?)",
                           (room, mode, (parent or 0) + 1, parent, offset, json.dumps(metadata)))

    @staticmethod
    def evaluate(rows, commands):
        valid_qualities = {"observing", "ready", "in_band", "rate_limit", "deduplicated"}
        confirmed_dispatch = {"sent", "verified", "inferred_response", "unconfirmed"}
        uncertain_dispatch = {"prepared", "transport_failed"}
        errors = []
        kwh=0.0
        power_seconds=0
        starts=0
        cycles=0
        reaches=[]
        error_start=None
        previous_valid=None
        for i,row in enumerate(rows):
            if row["measured"] is None or row["quality"] not in valid_qualities:
                error_start=None
                previous_valid=None
            else:
                if (previous_valid is None or row["tick"]-previous_valid["tick"]>120
                    or row["desired"]!=previous_valid["desired"]):
                    error_start=None
                previous_valid=row
                error=row["measured"]-row["desired"]
                errors.append(error)
                if abs(error)>0.5 and error_start is None: error_start=row["tick"]
                if abs(error)<=0.5 and error_start is not None:
                    reaches.append((row["tick"]-error_start)/60)
                    error_start=None
            if i==0: continue
            previous=rows[i-1]
            delta=row["tick"]-previous["tick"]
            if not 0<delta<=120:
                continue
            if row["watts"] is not None and previous["watts"] is not None:
                kwh+=(row["watts"]+previous["watts"])*delta/7_200_000
                power_seconds+=delta
            if row["confidence"]==previous["confidence"]=="verified":
                if previous["observed"]=="off" and row["observed"] in {"heat","cool"}: starts+=1
                if previous["observed"] in {"heat","cool"} and row["observed"]=="off": cycles+=1
        sent = sum(c["status"] in confirmed_dispatch for c in commands)
        ambiguous = sum(c["status"] in uncertain_dispatch for c in commands)
        verified = sum(c["status"]=="verified" for c in commands)
        return {
            "valid_samples":len(errors),
            "mae":sum(map(abs,errors))/len(errors) if errors else None,
            "rmse":sqrt(sum(x*x for x in errors)/len(errors)) if errors else None,
            "within_0_5":sum(abs(x)<=0.5 for x in errors)/len(errors) if errors else None,
            "within_0_8":sum(abs(x)<=0.8 for x in errors)/len(errors) if errors else None,
            "maximum_absolute_deviation":max(map(abs,errors),default=None),
            "median_recovery_minutes":median(reaches) if reaches else None,
            "verified_starts":starts,"verified_stops":cycles,
            "command_count":sent,"possible_beeps":sent+ambiguous,
            "ambiguous_dispatches":ambiguous,"physically_verified_commands":verified,
            "verified_command_rate":verified/(sent+ambiguous) if sent+ambiguous else None,
            "estimated_kwh":round(kwh,3) if power_seconds else None,
            "power_coverage_minutes":round(power_seconds/60,1),
        }

    @staticmethod
    def quality_model(rows, commands, mode):
        # Only independent confirmation may produce a model update.
        validated=[r for r in rows if r["confidence"]=="verified" and r["measured"] is not None
                   and r["quality"] in {"ready","in_band","rate_limit","deduplicated","off"}]
        active=[r for r in validated if r["observed"]==mode and r["verified_target"] is not None]
        readings={r["temp_report"]:r for r in active if r["temp_report"]}
        if len(readings)<30 or len({r["localday"] for r in readings.values()})<3:
            return None
        distinct=sorted(readings.values(),key=lambda r:r["tick"])
        deviations=[r["verified_target"]-r["measured"] for r in distinct]

        def interval_rates(series, expected):
            for first,second in zip(series,series[1:]):
                duration=second["tick"]-first["tick"]
                if (first["observed"]==second["observed"]==expected and 120<=duration<=1200
                    and first["temp_report"]!=second["temp_report"]):
                    yield first,(second["measured"]-first["measured"])*3600/duration

        passive=list(interval_rates(validated,"off"))
        active_rates=list(interval_rates(validated,mode))
        natural=[slope for _,slope in passive]
        factors=[slope/(r["outdoor"]-r["measured"]) for r,slope in passive
                 if r["outdoor"] is not None and abs(r["outdoor"]-r["measured"])>=3
                 and slope*(r["outdoor"]-r["measured"])>0]
        leak=median(factors) if len(factors)>=10 else None
        corrected=[slope-leak*(r["outdoor"]-r["measured"]) for r,slope in active_rates
                   if leak and r["outdoor"] is not None]
        delays=[(c["completed"]-c["sent"])/60 for c in commands
                if c["completed"] and c["status"]=="verified"]
        return {"samples":len(distinct),"requested_offset":median(deviations),
                "median_room_change_c_per_hour":median([v for _,v in active_rates]) if active_rates else None,
                "median_natural_drift_c_per_hour":median(natural) if len(natural)>=5 else None,
                "outdoor_leak_per_hour":leak,"thermal_time_constant_hours":1/leak if leak else None,
                "corrected_capacity_c_per_hour":median(corrected) if len(corrected)>=5 else None,
                "median_confirmation_delay_minutes":median(delays) if delays else None,
                "mae":sum(abs(r["desired"]-r["measured"]) for r in distinct)/len(distinct)}

    def current_summary(self, room):
        with closing(self.connect()) as db:
            daily=db.execute("SELECT metrics FROM daily WHERE room=? ORDER BY date DESC LIMIT 1",(room,)).fetchone()
            return json.loads(daily["metrics"]) if daily else {}

    def maintenance(self,now:datetime, configurations):
        today=now.date().isoformat()
        with closing(self.connect()) as db, db:
            if db.execute("SELECT completed FROM maintenance WHERE name='daily' AND completed=?",(today,)).fetchone():
                return
            yesterday=(now.date()-timedelta(days=1)).isoformat()
            since=int((now-timedelta(days=7)).timestamp())
            cutoff=int((now-timedelta(days=35)).timestamp())
            for room,defaults in configurations.items():
                rows=db.execute("SELECT * FROM samples WHERE room=? AND tick>=? ORDER BY tick",(room,since)).fetchall()
                commands=db.execute("SELECT * FROM commands WHERE room=? AND sent>=? ORDER BY sent",(room,since)).fetchall()
                daily=[r for r in rows if r["localday"]==yesterday]
                start=min((r["tick"] for r in daily),default=0)
                end=max((r["tick"] for r in daily),default=-1)+60
                metrics=self.evaluate(daily,[c for c in commands if start<=c["sent"]<end])
                db.execute("INSERT OR REPLACE INTO daily VALUES(?,?,?)",(room,yesterday,json.dumps(metrics)))
                if now.weekday()!=0: continue
                for mode,default in defaults.items():
                    model=self.quality_model(rows,commands,mode)
                    if model is None: continue
                    previous=db.execute("SELECT * FROM models WHERE room=? AND mode=? ORDER BY generation DESC LIMIT 1",(room,mode)).fetchone()
                    base=previous["offset"] if previous else default
                    parent=previous["generation"] if previous else None
                    generation=(parent or 0)+1
                    candidate=round(base+max(-0.5,min(0.5,model["requested_offset"]-base)),2)
                    candidate=max(-4,min(4,candidate))
                    model["previous_offset"]=base
                    if previous:
                        old=json.loads(previous["metrics"])
                        if (old.get("mae") is not None and model["mae"]>old["mae"]+0.25
                            and not old.get("rolled_back")):
                            candidate=old.get("previous_offset",default)
                            model["rolled_back"]=True
                    db.execute("INSERT INTO models VALUES(?,?,?,?,?,?)",(room,mode,generation,parent,candidate,json.dumps(model)))
            db.execute("DELETE FROM samples WHERE tick<?",(cutoff,))
            db.execute("DELETE FROM commands WHERE sent<?",(cutoff,))
            db.execute("INSERT OR REPLACE INTO maintenance VALUES('daily',?)",(today,))
