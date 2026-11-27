"""Capture journal and SQLite evidence with a public export schema."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
from math import isfinite
from pathlib import Path
import re
import sqlite3
from time import time
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from zoneinfo import ZoneInfo


UTC = timezone.utc
BEIJING = ZoneInfo("Asia/Shanghai")
MAX_POWER_GAP = 120
AUDIT_FIELDS = "at event commit config_sha256 code_sha256 outcome assessment verification proposed final rooms command_at status error stack http observation_version observation_fingerprint"
DECISION_FIELDS = "need normal_need lead gentle topology definition owner demand_order feasible zones environment output_evidence risks equipment action_scopes observations_after candidates rejections invalidated_initial unmet energy_scores recent_effects uncertain_zones"
ZONE_FIELDS = "primary primary_reading inputs constraints measured target at quality trend paused intent reason direction duration report_span reports qualified error evidence_from source_version source key"
ROOM_FIELDS = "room_c target_c mode_reported setpoint_reported power_w ready temperature_age_s power_age_s temperature_report_at power_report_at temperature_source power_context power_bands power_evidence_at status zone zone_intent zone_reason zone_target_c zone_temperature_c house_need prediction_10m_c trend_valid cold_seconds warm_seconds cold_confirmed warm_confirmed"
COMMAND_FIELDS = "room kind value before sent_at status purpose transaction target_at_send expected before_room_c before_watts before_mode_at followup physical_verified proxy_at relay_at http control"
SHADOW_FIELDS = "power_evidence_status observed_at power_report_at power_report_w watts low_power_cutoff reversal_low_cutoff low_power_seconds above_low_seconds above_high_seconds power_evidence measured target temperature_report_at temperature_source reported_mode from_setpoint source_fresh outdoor model model_at release_commit policy_revision learning_error"
SAMPLE_FIELDS = "room tick measured desired reported adapter_mode watts outdoor temp_report quality"
NESTED = set((ZONE_FIELDS + " " + ROOM_FIELDS + " " + DECISION_FIELDS + " " + COMMAND_FIELDS +
    " mode started_at off_at mode_started_at phase members ready_at id target zone source last_report bands threshold floor since seconds context low high above_low learning count delta uncertainty from to value report_span reports fresh interval_seconds watt_hours useful improvement thermal_delta start_c end_c ended_at covered_seconds weather_c served_zone last_increase own_command zone_report power_at deviation projected_adverse point influence temperature_report_at power_report_at room_age_s power_age_s reported watts room_c temperature_c definition before_c unit episode_count at code_sha256 config_sha256 step mode provenance local zone_reason time start finish enabled until reason virtual_mode context sensors limits centers_w edges_w counts samples span_minutes fit file function line status expires expected started last_power context_changed old new phases targets 5 15").split())
WORDS = re.compile(r"^[a-z][a-z0-9_]{0,55}$")
HEXDIGEST = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
SAFE_CHINESE = re.compile(r"^[\u3400-\u9fff\s、，。；：﹕（）±°℃0-9.≤≥+−-]{1,100}$")
STATIC_LABELS = {"heat", "cool", "auto", "off", "on", "fan_only", "dry", "proxy_echo", "unavailable", "unknown",
                 "TypeError", "AttributeError", "OperationalError", "IntegrityError", "HTTPError", "OSError", "TimeoutError",
                 "ValueError", "KeyError", "DiskIOError", "True", "False"}


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def iso(t, zone=UTC):
    return datetime.fromtimestamp(t, zone).isoformat() if t is not None else None


def num(value):
    try:
        value = float(value)
        return value if isfinite(value) else None
    except (TypeError, ValueError):
        return None


def stamp(value):
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return num(value)


def select(obj, fields):
    return {k: obj[k] for k in fields.split() if k in obj}


class PublicSchema:
    def __init__(self, key, rooms, topology):
        self.key = bytes.fromhex(key)
        self.aliases = {}
        for room, config in rooms.items():
            self.aliases[room] = self.id("unit", room)
            for key in ("temperature", "adapter", "power", "power_fallback", "witness_trial_id"):
                if value := config.get(key):
                    self.aliases[value] = self.id(key, value)
            entity = f"climate.{room}_room_thermostat"
            self.aliases[entity] = self.id("virtual", entity)
        for kind in ("zones", "sensors"):
            for name, item in topology.get(kind, {}).items():
                self.aliases[name] = self.id(kind, name)
                for entity in (item.get("entity"), item.get("pause_entity"), item.get("target", {}).get("entity")):
                    if entity:
                        self.aliases.setdefault(entity, self.id("input", entity))
        for item in topology.get("actuators", {}).values():
            self.aliases.setdefault(item["lock_group"], self.id("group", item["lock_group"]))
        self.codes = {"reason", "status", "error", "event", "intent", "direction", "mode", "purpose", "outcome", "quality",
                      "zone_reason", "zone_intent", "house_need", "need", "normal_need", "power_evidence_status", "kind",
                      "topology", "observation_version", "phase", "file", "function", "state", "mode_reported", "reported_mode"}

    def id(self, kind, value):
        return kind + "_" + hmac.new(self.key, str(value).encode(), sha256).hexdigest()[:16]

    def convert(self, value, field=None):
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value if isfinite(value) else None
        if isinstance(value, str):
            if value in self.aliases:
                return self.aliases[value]
            if field in {"commit", "release_commit", "config_sha256", "code_sha256", "definition",
                         "observation_fingerprint", "source_version"} and HEXDIGEST.fullmatch(value):
                return value
            if stamp(value) is not None and re.match(r"^\d{4}-\d{2}-\d{2}T", value):
                return value
            if value in STATIC_LABELS or (field in self.codes and WORDS.fullmatch(value)) or (
                    field in {"status", "reason", "zone_reason"} and SAFE_CHINESE.fullmatch(value)):
                return value
            if field in {"file", "function"} and re.fullmatch(r"[a-z_][a-z0-9_]*(?:\.py)?", value):
                return value
            return self.id("text", value)
        if isinstance(value, (list, tuple)):
            return [self.convert(v, field) for v in value]
        if isinstance(value, dict):
            return {(self.aliases[k] if k in self.aliases else k): self.convert(v, k)
                    for k, v in value.items() if k in NESTED or k in self.aliases}
        return None

    def record(self, value, fields):
        return {k: self.convert(v, k) for k, v in select(value, fields).items()}

    def audit(self, data):
        result = self.record(data, AUDIT_FIELDS)
        for key in ("assessment", "verification"):
            if data.get(key):
                result[key] = self.record(data[key], DECISION_FIELDS)
        if data.get("rooms"):
            result["rooms"] = {self.aliases[k]: self.record(v, ROOM_FIELDS) for k,v in data["rooms"].items() if k in self.aliases}
        if data.get("stack"):
            result["stack"] = [self.record(f, "file function line") for f in data["stack"]]
        room = data.get("room")
        if room in self.aliases:
            result["room"] = self.aliases[room]
        result["id"] = self.id("event", (data["at"], data.get("event"), data.get("room"), data.get("command_at")))
        if data.get("command_at") is not None:
            selected_room = room or (data.get("final") or [""])[0]
            result["command_id"] = self.id("command", (selected_room, data["command_at"]))
        return result

    def configuration(self, rooms, topology, policy):
        sources = {}
        for name, sensor in topology.get("sensors", {}).items():
            sources[self.aliases[name]] = {"role": sensor["role"], "enabled": sensor.get("enabled", True),
                  "source": self.aliases.get(sensor.get("entity", rooms.get(sensor.get("from_actuator"), {}).get("temperature"))),
                  "max_age_seconds": sensor.get("max_age_seconds", 1200), "weight": sensor.get("weight", 1)}
        zones = {self.aliases[k]: {"primary": self.aliases[v["primary"]], "sensors": [self.aliases[x] for x in v.get("sensors", ())],
                "fusion": v.get("fusion", "primary"), "target": {name: (value if name=="celsius" else self.aliases.get(value))
                    for name,value in v["target"].items() if name in {"actuator", "entity", "celsius"}},
                "pause_input": self.aliases.get(v.get("pause_entity"))} for k,v in topology.get("zones", {}).items()}
        actuators = {self.aliases[k]: {"zone": self.aliases[v["zone"]], "influences": {self.aliases[z]: w for z,w in v["influences"].items()},
                     "priority": v.get("priority",0), "lock_group": self.aliases[v["lock_group"]]} for k,v in topology.get("actuators", {}).items()}
        return {"observation_version": self.convert(topology.get("observation_version"), "observation_version"), "schema": topology.get("schema"),
                "policy": {k:v for k,v in policy.items() if isinstance(v, (int,float))}, "sensors": sources, "zones": zones,
                "actuators": actuators, "connections": [{"from": self.aliases[l["from"]], "to": self.aliases[l["to"]], "coupling": l["coupling"]}
                 for l in topology.get("connections", ())], "opposite_operation": select(topology.get("opposite_operation", {}), "enabled max_cross_influence"),
                "bindings": {self.aliases[k]: {"power": self.aliases[v["power"]], "adapter": self.aliases[v["adapter"]],
                   "fallback": self.aliases.get(v.get("power_fallback")), "soft_limits": select(v.get("soft_limits", {}), "heat cool"),
                   "off_reference": select(v.get("verified_off_power", {}), "ceiling_w status"),
                   **select(v, "transition_off_seconds overshoot_high_power_seconds")} for k,v in rooms.items()}}


@contextmanager
def readonly(path):
    """Pin a read transaction. SQLite itself reads committed pages in the WAL."""
    db = sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True, timeout=4)
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA temp_store=MEMORY")
        db.execute("BEGIN")
        db.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
        yield db
    finally:
        db.close()


def load_json(path):
    return json.loads(path.read_bytes())


def audit_rows(folder, start, end):
    rows = {}
    for path in folder.glob("live-audit.jsonl*"):
        try:
            with path.open() as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue   # A rotation or an in-flight final line can be observed.
                    at = row.get("at", 0)
                    if at < end and (at >= start or row.get("event") == "startup"):
                        rows[(at, row.get("event"), row.get("room"), row.get("command_at"))] = row
        except FileNotFoundError:
            continue
    for path in folder.glob("startup-*.json"):
        if not re.fullmatch(r"startup-\d+",path.stem):
            continue
        raw=load_json(path)
        if raw["at"]<end:
            rows[(raw["at"],"startup",None,None)]=raw
    # The startup file and log are the same event with different emission times.
    for key, row in list(rows.items()):
        if row.get("startup_at") is not None and (row["startup_at"],"startup",None,None) in rows:
            del rows[key]
    return sorted(rows.values(), key=lambda row: row["at"])


def read_journal(folder, start, end, captured, schema):
    commands = {}
    states = []
    for path in folder.glob("continuous-*.json"):
        if not re.fullmatch(r"continuous-\d+(?:-actions-\d{4}-\d{2}-\d{2})?", path.stem):
            continue
        raw = load_json(path)
        if "-actions-" not in path.stem:
            states.append(raw)
        for action in raw.values() if "-actions-" in path.stem else raw["commands"]:
            sent = action["sent_at"]
            follow = action.get("followup", {})
            effects = [x for x in (action.get("proxy_at"), action.get("relay_at"), *(p["at"] for p in follow.values())) if x is not None]
            if sent >= end or (sent < start and not any(start <= x < end for x in effects)
                               and action["status"] not in {"prepared", "accepted"}):
                continue
            command_id = schema.id("command", (action["room"], sent))
            row = schema.record(action, COMMAND_FIELDS)
            row.update(id=command_id, context_only=sent < start,
                       state_at_collection=row.get("status"), collected_before=cutoff_marker(captured))
            row["followup"] = {k: schema.record(v, "at room_c room_age_s watts power_age_s reported temperature_report_at power_report_at")
                               for k,v in follow.items() if k in {"5", "15"} and v["at"] <= captured}
            commands[command_id] = row
    permit = {}
    for filename in ("live-permit.json", "continuous-permit.json"):
        path = folder / filename
        if path.is_file():
            permit = load_json(path)
            break
    current = next((s for s in states if s.get("start") == permit.get("start")), {})
    status = {"permit": {k: schema.convert(v, k) for k,v in select(permit, "enabled start end isolated_rooms paused_zones").items()},
              "manual_holds": {schema.convert(k): schema.convert(v, "status") for k,v in current.get("manual_holds", {}).items()},
              "failure_holds": {schema.convert(k): schema.convert(v, "status") for k,v in current.get("holds", {}).items()},
              "parked": schema.convert(current.get("parked", {})),
              "transitions": schema.convert(current.get("transitions", {})),
              "pending": [schema.record(a, COMMAND_FIELDS) for a in current.get("commands", ()) if a["status"] in {"prepared","accepted"}]}
    return sorted(commands.values(), key=lambda a: (a["sent_at"],a["id"])), status


def cutoff_marker(captured):
    return iso(captured)


def live_receipts(folder, start, end, schema):
    rows = []
    for path in folder.glob("receipt-*.json"):
        try:
            row = load_json(path)
        except FileNotFoundError:
            continue
        at = row.get("received_at", row.get("at", row.get("sent_at", 0)+row.get("relay_delay_s",0)))
        if start <= at < end:
            selected = schema.record(row, "room sent_at received_at at status purpose relay relay_delay_s")
            selected["received_at"] = at
            selected["command_id"] = schema.id("command", (row.get("room"), row.get("sent_at")))
            selected["id"] = schema.id("receipt", (selected["command_id"], at))
            rows.append(selected)
    return sorted(rows,key=lambda row: (row.get("received_at",row.get("at",0)), row["id"]))


def source_watermark(rows, field, acquired, expected_start, end):
    points = [r[field] for r in rows if r.get(field) is not None]
    return {"acquired_at": iso(acquired), "first_at": min(points, default=None), "last_at": max(points, default=None),
            "rows": len(rows), "window_from": expected_start, "window_to": end}


def read_databases(root, start, end, rooms, topology, schema):
    chunks, marks = {}, {}
    configure = root/"config"
    configs = [("samples", configure/".storage/room_climate_control.sqlite3"),
               ("shadow", root/"shadow-data/shadow.sqlite3"), ("recorder", configure/"home-assistant_v2.db")]
    for name, path in configs:
        acquired = time()
        try:
            with readonly(path) as db:
                if name == "samples":
                    rows = db.execute("SELECT * FROM samples WHERE tick>=? AND tick<? ORDER BY tick,room", (start-60,end)).fetchall()
                    output = [schema.record(dict(r), SAMPLE_FIELDS) for r in rows if r["room"] in rooms]
                    chunks["room-samples"] = output
                    marks[name] = source_watermark(output,"tick",acquired,start,end)
                elif name == "shadow":
                    rows = db.execute("SELECT room,tick,value FROM decisions WHERE tick>=? AND tick<? ORDER BY tick,room", (start-120,end))
                    output = [dict(schema.record(json.loads(v), SHADOW_FIELDS), room=schema.aliases[r], tick=t) for r,t,v in rows if r in rooms]
                    chunks["shadow-decisions"] = output
                    marks[name] = source_watermark(output,"tick",acquired,start,end)
                    meters = {v["power"] for v in rooms.values()}
                    chunks["power-reports"] = [{"source":schema.aliases[s], "observer_epoch":e, "report_at":t, "watts":w,
                        "id":schema.id("power_report",(s,e,t))} for s,e,t,w in db.execute(
                        "SELECT source,observer_epoch,report_at,watts FROM power_reports WHERE report_at BETWEEN ? AND ? ORDER BY report_at,source,observer_epoch",
                        (start-120,end+120)) if s in meters]
                    chunks["power-models"] = [dict(room=schema.aliases[r], trained=t, status=s, model=schema.convert(json.loads(m)) if m else None)
                      for r,t,s,m in db.execute("SELECT room,trained,status,model FROM power_models WHERE trained BETWEEN ? AND ? ORDER BY trained,room",(start-3600,end)) if r in rooms]
                    marks["power_ledger"] = source_watermark(chunks["power-reports"], "report_at",acquired,start,end)
                else:
                    entities = {f"climate.{r}_room_thermostat" for r in rooms}
                    entities.update(v[k] for v in rooms.values() for k in ("temperature","power","adapter") if k in v)
                    entities.update(s["entity"] for s in topology.get("sensors", {}).values() if s.get("enabled",True) and "entity" in s)
                    output = []
                    for entity in sorted(entities):
                        rows = db.execute("""SELECT s.state_id,s.state,s.last_changed_ts,s.last_updated_ts,s.last_reported_ts,a.shared_attrs
                                  FROM states_meta m JOIN states s ON m.metadata_id=s.metadata_id
                                  LEFT JOIN state_attributes a ON a.attributes_id=s.attributes_id
                                  WHERE m.entity_id=? AND s.last_updated_ts>=? AND s.last_updated_ts<?
                                  ORDER BY s.last_updated_ts,s.state_id""", (entity,start-120,end+120))
                        for ident, state, changed, updated, reported, raw in rows:
                            attrs = json.loads(raw or "{}")
                            fields = select(attrs, "temperature temperature_reported_at power_reported_at power_observer_started_at configured_for_actuation automatic_control_enabled control_hold_latched")
                            # Exact state report identity and allowlisted attributes are sufficient to reproduce joins.
                            output.append(dict(id=schema.id("recorder", ident), source=schema.aliases[entity],
                                               value=num(state) if num(state) is not None else schema.convert(state,"state"),
                                               updated_at=updated, changed_at=changed, reported_at=reported,
                                               fields={k: schema.convert(v,k) for k,v in fields.items()}))
                    chunks["recorder-states"] = sorted(output, key=lambda x: (x["updated_at"], x["id"]))
                    marks[name] = source_watermark(output,"updated_at",acquired,start,end)
        except (sqlite3.Error, FileNotFoundError) as error:
            marks[name] = {"acquired_at":iso(acquired),"error":type(error).__name__,"missing":[start,end]}
    return chunks, marks


def gaps(points, start, end, cadence):
    bins = sorted({int(t)//cadence*cadence for t in points if start <= t < end})
    previous = start
    missing = []
    for tick in bins:
        if tick > previous:
            missing.append([previous,tick])
        previous = max(previous, tick+cadence)
    if previous < end:
        missing.append([previous,end])
    return missing


def independent_minutes(audit, topology, schema, versions=None):
    frames = {}
    for row in audit:
        if row.get("event") != "cycle" or not row.get("rooms"):
            continue
        values = {}
        definition = (versions or {}).get(row.get("config_sha256"),topology)
        for zone, spec in definition.get("zones", {}).items():
            alias = schema.aliases[zone]
            info = row.get("assessment", {}).get("zones", {}).get(alias)
            if info:
                raw = info["primary_reading"]
                if raw["quality"] == "fresh" and info["target"] is not None:
                    values[alias] = {"temperature": raw["value"], "target": info["target"], "report_at": raw["at"],
                                     "source": raw.get("source"), "paused": info["paused"]}
            else:
                # Legacy audit has equipment samples. Map only each configured
                # independent primary to an actuator sample, never its auxiliaries.
                unit = definition["sensors"][spec["primary"]].get("from_actuator")
                legacy = row["rooms"].get(schema.aliases.get(unit), {})
                if legacy.get("room_c") is not None and legacy.get("target_c") is not None and legacy.get("temperature_age_s", 1e9) <= 1200:
                    values[alias] = {"temperature":legacy["room_c"],"target":legacy["target_c"],
                                     "report_at":legacy.get("temperature_report_at"),"source":schema.aliases[spec["primary"]]}
        minute = int(row["at"])//60*60
        frames[minute] = {"tick": minute, "at": row["at"], "zones": values, "commit": row.get("commit"),
                          "config_sha256": row.get("config_sha256"),
                          "controllable": {k: bool(v.get("ready")) for k,v in row["rooms"].items()},
                          "uncontrollable_reasons": {k: v.get("status") for k,v in row["rooms"].items() if not v.get("ready")}}
    return [frames[k] for k in sorted(frames)]


def reports(frames, schema, audit=()):
    unique = {}
    for frame in frames:
        for zone, obs in frame["zones"].items():
            if obs["report_at"] is not None:
                ident = schema.id("temperature_report", (zone, obs["source"], obs["report_at"]))
                unique[ident] = {"id":ident,"zone":zone,"report_at":obs["report_at"],"source":obs["source"],
                                 "temperature":obs["temperature"]}
    for row in audit:
        for zone, view in row.get("assessment", {}).get("zones", {}).items():
            primary=view.get("primary_reading") or {}
            for sensor, obs in {view.get("primary",zone):primary, **view.get("inputs",{})}.items():
                if obs.get("at") is not None:
                    ident=schema.id("temperature_report",(sensor,obs.get("source"),obs["at"]))
                    unique[ident]={"id":ident,"zone":zone,"sensor":sensor,"source":obs.get("source"),"report_at":obs["at"],
                                   "temperature":obs.get("value"),"quality":obs.get("quality")}
        for sensor, obs in row.get("assessment", {}).get("environment",{}).items():
            if obs.get("at") is not None:
                ident=schema.id("temperature_report",(sensor,obs.get("source"),obs["at"]))
                unique[ident]={"id":ident,"sensor":sensor,"role":"environment","source":obs.get("source"),"report_at":obs["at"],
                               "temperature":obs.get("value"),"quality":obs.get("quality")}
    return sorted(unique.values(), key=lambda r: (r["report_at"],r["id"]))


def temp_stats(frames, zone, band):
    points = [(f["tick"],f["zones"][zone]) for f in frames if zone in f["zones"]]
    errors = [p["temperature"]-p["target"] for _,p in points]
    spells = []
    active = None
    previous = None
    for (tick, p), error in zip(points, errors):
        sign = -1 if error < -band-1e-6 else 1 if error > band+1e-6 else 0
        if active and ((sign,p["target"]) != (active["side"], active["target"]) or tick-previous != 60):
            if sign == 0 and tick-previous == 60 and p["target"] == active["target"]:
                active["recovered_at"] = tick
                active["recovery_seconds"] = tick-active["since"]
            active = None
        if sign:
            if active is None:
                active = {"since":tick,"side":sign,"target":p["target"],"observed_minutes":0,
                          "recovered_at":None,"recovery_seconds":None}
                spells.append(active)
            active["observed_minutes"] += 1
        previous = tick
    n = len(errors)
    return {"valid_minutes":n, "mean_signed_c": sum(errors)/n if n else None,
            "inside_tracking_pct": 100*sum(abs(x)<=band+1e-6 for x in errors)/n if n else None,
            "inside_0_5_pct":100*sum(abs(x)<=0.5+1e-6 for x in errors)/n if n else None,
            "inside_0_7_pct":100*sum(abs(x)<=0.7+1e-6 for x in errors)/n if n else None,
            "cold_minutes":sum(x < -band-1e-6 for x in errors), "warm_minutes":sum(x > band+1e-6 for x in errors),
            "longest_cold_minutes":max((s["observed_minutes"] for s in spells if s["side"] == -1), default=0),
            "longest_warm_minutes":max((s["observed_minutes"] for s in spells if s["side"] == 1), default=0), "excursions":spells}


def power_ranges(reports, start, end):
    epochs = defaultdict(dict)
    for row in reports:
        epochs[row["observer_epoch"]][row["report_at"]] = row["watts"]
    ranges = []
    for series in epochs.values():
        values = sorted(series.items())
        for (a,w), (b,v) in zip(values, values[1:]):
            left,right = max(a,start),min(b,end)
            if (w is not None and v is not None and w>=0 and v>=0 and 0<b-a<=MAX_POWER_GAP and right>left):
                ranges.append((left,right,w))
    ordered = sorted(ranges)
    result = []
    # Epochs may overlap around an observer restart. Conflicting overlaps
    # contribute no energy instead of double-counting either source.
    for item in ordered:
        if result and item[0] < result[-1][1]:
            left, _, watts = result[-1]
            result[-1] = (left,item[0],watts)
        result.append(item)
    return result


def summarize(frames, power, meter_by_unit, zones, commands, audit, start, end, band):
    expected = end-start
    real_frames = [f for f in frames if start <= f["tick"] < end]
    complete = [f for f in real_frames if all(z in f["zones"] for z in zones)]
    simultaneous = {str(limit): (100*sum(all(abs(f["zones"][z]["temperature"]-f["zones"][z]["target"])<=limit+1e-6 for z in zones)
                                  for f in complete)/len(complete) if complete else None) for limit in (band, 0.5, 0.7)}
    ranges = {unit: power_ranges([row for row in power if row["source"]==source], start,end) for unit,source in meter_by_unit.items()}
    meters = {}
    edge_events = defaultdict(list)
    for unit, intervals in ranges.items():
        covered = sum(b-a for a,b,_ in intervals)
        meters[unit] = {"kwh":sum((b-a)*v for a,b,v in intervals)/3600000,
                        "covered_seconds":covered,"coverage_pct":100*covered/expected}
        for a,b,v in intervals:
            if a < b:
                edge_events[a].append((unit,v))
                edge_events[b].append((unit,None))
    active = {}
    total = coverage = 0
    previous = start
    for t, changes in sorted(edge_events.items()):
        if active and len(active)==len(ranges):
            total += sum(active.values())*(t-previous)
            coverage += t-previous
        for unit,watts in sorted(changes,key=lambda row: row[1] is not None):
            if watts is None:
                active.pop(unit,None)
            else:
                active[unit] = watts
        previous = t
    acts = Counter()
    previous_modes = {}
    for a in commands:
        mode = a.get("value")
        if a.get("proxy_at") and a["kind"]=="mode" and a.get("before"):
            unit, before = a["room"],a["before"][0]
            if mode=="off":
                previous_modes[unit] = before
            if mode in {"heat","cool"}:
                last = previous_modes.pop(unit, before)
                if start <= a["proxy_at"] < end and last in {"heat","cool"} and last != mode:
                    acts["reversal_proxy"] += 1
        if start <= a["sent_at"] < end:
            acts["attempted"] += a["status"] not in {"prepared","canceled"}
            acts["unconfirmed"] += a["status"] == "unconfirmed"
            acts["rejected"] += a["status"] == "rejected"
        if a.get("proxy_at") and start <= a["proxy_at"] < end:
            acts["proxy_echo"] += 1
            if a["kind"] == "temperature":
                acts["setpoint_proxy"] += 1
            else:
                acts["stop_proxy" if mode=="off" else "start_proxy"] += 1
    return {"duration_seconds":expected, "zones":{z:temp_stats(real_frames,z,band) for z in zones},
            "joint_temperature_minutes":len(complete), "joint_temperature_pct":simultaneous,
            "minute_coverage": {"observed":len(real_frames),"expected_seconds":expected,
                               "missing":gaps((f["at"] for f in real_frames),start,end,60)},
            "metered":meters, "joint_kwh":total/3600000,"joint_coverage_pct":100*coverage/expected,
            "joint_covered_seconds":coverage,"actuation_counts":dict(acts),
            "uncontrollable_observed_minutes":{k:sum(not f["controllable"].get(k,False) for f in real_frames) for k in meter_by_unit},
            "uncontrollable_reasons":{k:dict(Counter(f["uncontrollable_reasons"][k] for f in real_frames if k in f["uncontrollable_reasons"])) for k in meter_by_unit},
            "controller_errors":dict(Counter(r["event"] for r in audit if start<=r["at"]<end and r["event"].endswith("error")))}


def release_groups(audit, frames, power, meter_by_unit, zones, commands, start, end, band, specs=None):
    version = None
    points = [(start, None)]
    for row in audit:
        if row.get("event") != "startup":
            continue
        if row["at"] >= end:
            break
        new = {key:row.get(key) for key in ("commit","config_sha256","code_sha256","observation_version","observation_fingerprint","at")}
        if row["at"] <= start:
            points[0]=(start,new)
        elif version is None or any(new[k]!=version[k] for k in ("commit","config_sha256","code_sha256")):
            points.append((row["at"],new))
        version = new
    results=[]
    for (a,v),(b,_) in zip(points,points[1:]+[(end,None)]):
        if a>=b:
            continue
        sites=(specs or {}).get((v or {}).get("config_sha256"),{})
        refs=list(sites["zones"]) if sites else zones
        limits=sites["policy"].get("tracking_band",band) if sites else band
        results.append({"start":a,"end":b,"version":v,
                        "statistics":summarize(frames,power,meter_by_unit,refs,commands,audit,a,b,limits)})
    return results


def collect(root, start, end, runtime, key):
    captured = time()
    room_path = root/"shadow/rooms.json"
    rooms = load_json(room_path)
    topology = load_json(root/"shadow/topology.json")
    policy = load_json(root/"shadow/house_policy.json")
    schema = PublicSchema(key,rooms,topology)
    audit_floor = end-86400
    folder = root/"pilot-data"
    full_raw = audit_rows(folder,audit_floor,end)
    versions={}
    definitions={}
    for ref in {r.get("config_sha256") for r in full_raw}:
        if ref and HEXDIGEST.fullmatch(ref) and (path:=folder/f"configuration-{ref}.json").is_file():
            old=load_json(path)
            old_schema=PublicSchema(key,old["rooms"],old["topology"])
            schema.aliases.update({k:v for k,v in old_schema.aliases.items() if k not in schema.aliases})
            definitions[ref]=old["topology"]
            versions[ref]=schema.configuration(old["rooms"],old["topology"],old["policy"])
    raw_audit = [schema.audit(row) for row in full_raw]
    current_audit = audit_rows(folder,captured-300,captured+1)
    latest = next((schema.audit(r) for r in reversed(current_audit) if r.get("event")=="cycle"),None)
    commands, status = read_journal(folder,audit_floor,end,captured,schema)
    receipts = live_receipts(folder,start,end,schema)
    chunks, marks = read_databases(root,audit_floor,end,rooms,topology,schema)
    frames = independent_minutes(raw_audit,topology,schema,definitions)
    zones = [schema.aliases[k] for k in topology["zones"]]
    meters = {schema.aliases[k]:schema.aliases[c["power"]] for k,c in rooms.items()}
    power = chunks.get("power-reports",[])
    # Controller polls preserve reports even when the shadow database lags.
    for row in raw_audit:
        for unit, v in row.get("rooms", {}).items():
            context, report = v.get("power_context"),v.get("power_report_at")
            if unit in meters and context and context[0]==meters[unit] and report is not None and report<=end+120:
                power.append({"source":context[0],"observer_epoch":context[1],"report_at":report,"watts":v.get("power_w"),
                              "id":schema.id("power_report",(context[0],context[1],report)),"seen_in":"live"})
    merged = {}
    for r in power:
        ident = (r["source"],r["observer_epoch"],r["report_at"])
        if ident not in merged or r["watts"] is None:
            merged[ident] = r
    power = sorted(merged.values(),key=lambda r:(r["report_at"],r["source"],r["observer_epoch"]))
    window = [{k:v for k,v in a.items() if k not in {"collected_before"}} for a in commands if a["sent_at"]>=start or
              any(start <= x < end for x in (a.get("proxy_at"),a.get("relay_at"),*(p["at"] for p in a["followup"].values())) if x is not None)
              or a["status"] in {"prepared","accepted"}]
    raw = {"live-audit":[r for r in raw_audit if start<=r["at"]<end or r.get("event")=="startup"],
           "commands":window,"receipts":receipts,"temperature-reports":reports([f for f in frames if start-60<=f["tick"]<end and not definitions.get(f["config_sha256"])],schema,
                                      (r for r in raw_audit if start<=r["at"]<end)),
           "independent-minutes":[f for f in frames if start<=f["at"]<end],
           "power-reports":[r for r in power if start-120<=r["report_at"]<=end+120]}
    for name,rows in chunks.items():
        if name not in raw:
            lower = start-3600 if name=="power-models" else start-120
            time_field = {"recorder-states":"updated_at","power-models":"trained"}.get(name,"tick")
            raw[name]=[r for r in rows if lower<=r[time_field]]
    marks["controller"]=source_watermark(raw["live-audit"],"at",captured,start,end)
    marks["controller"]["missing_intervals"] = gaps((r["at"] for r in raw["live-audit"] if r.get("event")=="cycle"),start,end,60)
    for name,key in (("samples","room-samples"),("shadow","shadow-decisions")):
        marks[name]["missing_by_unit"]={u:gaps((r["tick"] for r in raw.get(key,[]) if r["room"]==u),start,end,60) for u in meters}
    marks["power_report_stream"]={u:gaps((r["report_at"] for r in raw["power-reports"] if r["source"]==s and r["watts"] is not None),start,end,120)
                                  for u,s in meters.items()}
    band = policy["tracking_band"]
    available = max(audit_floor,min((r["at"] for r in raw_audit if r.get("event")=="cycle"),default=start))
    reusable = [r for r in raw_audit if r.get("event")=="startup"]
    summary = {"window":summarize(frames,power,meters,zones,commands,raw_audit,start,end,band),
               "by_version":release_groups(reusable,frames,power,meters,zones,commands,start,end,band,versions),
               "rolling_24h":{"actual_from":available,"end":end,"elapsed_seconds":end-available,
                 "by_version":release_groups(reusable,frames,power,meters,zones,commands,available,end,band)}}
    return {"captured_at":captured,"runtime":runtime,"raw":raw,"status":{"last_completed_cycle":latest,**status},
            "effective":schema.configuration(rooms,topology,policy),"configurations":versions,"summary":summary,"watermarks":marks,
            "private_values":[k for k in schema.aliases if "." in k]}


def observed_versions(stats):
    return [{"from_utc":iso(row["start"]),"to_utc":iso(row["end"]),
             **{k:(row.get("version") or {}).get(k) for k in ("commit","config_sha256","code_sha256")}}
            for row in stats.get("by_version", ())]


def bundle(window, kind, snapshot, prior, previous_revision):
    observed = snapshot["raw"]
    logical = sha256(json_bytes(observed)).hexdigest()
    files = {"current-state.json":json_bytes(snapshot["status"]), "effective-config.json":json_bytes(snapshot["effective"]),
             "summary.json":json_bytes(snapshot["summary"]),
             "configuration-versions.json":json_bytes(snapshot.get("configurations",{}))}
    files.update({f"raw/{name}.jsonl":b"".join(json_bytes(r) for r in rows) for name,rows in observed.items()})
    start,end=window
    manifest = {"schema":1,"kind":kind,"from_utc":iso(start),"to_utc":iso(end),"from_beijing":iso(start,BEIJING),
                "to_beijing":iso(end,BEIJING),"capture_utc":iso(snapshot["captured_at"]),"original_source_timezone":"Asia/Tokyo",
                "version":snapshot["runtime"],"observation_version":snapshot["effective"]["observation_version"],
                "sources":snapshot["watermarks"],"coverage":{k:snapshot["summary"]["window"][k] for k in ("joint_coverage_pct","joint_temperature_minutes","minute_coverage")},
                "preceding_package":prior,"supersedes":previous_revision,"logical_sha256":logical,
                "observed_versions":observed_versions(snapshot["summary"]),
                "files":{k:sha256(v).hexdigest() for k,v in files.items()},
                "evidence_note":"Temperature and meter reports are observations. Device echoes are proxy confirmation. SQLite sources use separate read transactions."}
    files["manifest.json"]=json_bytes(manifest)
    secrets = re.compile(rb'(?i)(?:\bgh[pousr]_|\bgithub_pat_)[A-Za-z0-9_]{12,}|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}|(?:\b192\.168\.|\b10\.\d{1,3}\.|\b172\.(?:1[6-9]|2\d|3[01])\.)\d{1,3}\.\d{1,3}')
    for content in files.values():
        if secrets.search(content) or any(value.encode() in content for value in snapshot["private_values"]):
            raise ValueError("Protected identifier in public export")
    return manifest,files


def write_zip(path, files):
    with ZipFile(path,"w",ZIP_DEFLATED,compresslevel=6) as package:
        for name,body in sorted(files.items()):
            package.writestr(ZipInfo(name),body,compress_type=ZIP_DEFLATED,compresslevel=6)


def verify_zip(path, expected):
    if sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError("Downloaded archive digest mismatch")
    with ZipFile(path) as package:
        assert package.testzip() is None
        manifest = json.loads(package.read("manifest.json"))
        if set(package.namelist()) != {*manifest["files"],"manifest.json"}:
            raise ValueError("Downloaded archive inventory mismatch")
        if not all(sha256(package.read(k)).hexdigest()==v for k,v in manifest["files"].items()):
            raise ValueError("Downloaded archive file digest mismatch")
    return manifest
