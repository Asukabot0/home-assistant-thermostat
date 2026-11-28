"""Offline app that reads only HA history and never dispatches commands."""
from contextlib import closing, contextmanager
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from math import ceil
import signal
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event
from traceback import extract_tb
import time
from zoneinfo import ZoneInfo

from cadence import JOURNALS, SHARED_ROOMS, TRIALS_SCHEMA, latest_beeps
from model import Activity, advise, calibrate, identify, numeric
from pilot_review import trial_review
from steady_learning import (
    EPISODE_SECONDS,
    SCHEMA as LEARNING_SCHEMA,
    initial_setpoint,
    learned_offset,
    learning_active,
    learning_session,
    record_episode,
)


LOG = logging.getLogger("climate-shadow")
SOURCE = Path("/ha-readonly")
DATA = Path("/data")
OUTPUT = Path("/output")
LOCAL = ZoneInfo("Asia/Tokyo")
CONFIG = Path(os.getenv("ROOM_CLIMATE_ROOMS_FILE", str(Path(__file__).with_name("rooms.json"))))
DAY = 86400
HISTORY = 7 * DAY
POWER_REPORT_GAP_SECONDS = 120
RECORDER_INPUT_GRACE_SECONDS = 90
LATE_RECORDER = {"recorder_snapshot_missing", "recorder_power_unmatched"}
POLICY_REVISION = 9
POLICY_START_KEY = f"policy_started_{POLICY_REVISION}"
SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS power_reports (
    source TEXT NOT NULL, observer_epoch TEXT NOT NULL, report_at REAL NOT NULL, watts REAL,
    PRIMARY KEY (source, observer_epoch, report_at)
);
CREATE TABLE IF NOT EXISTS power_models (
    room TEXT NOT NULL, trained INTEGER NOT NULL, status TEXT NOT NULL, model TEXT,
    PRIMARY KEY (room, trained)
);
CREATE TABLE IF NOT EXISTS decisions (
    room TEXT NOT NULL, tick INTEGER NOT NULL, localday TEXT NOT NULL,
    hypothetical INTEGER NOT NULL, value TEXT NOT NULL, PRIMARY KEY (room, tick)
);
CREATE INDEX IF NOT EXISTS decisions_daily ON decisions (localday, room, tick);
CREATE INDEX IF NOT EXISTS decisions_sent ON decisions (room, tick) WHERE hypothetical=1;
CREATE TABLE IF NOT EXISTS daily (
    localday TEXT NOT NULL, room TEXT NOT NULL, metrics TEXT NOT NULL, PRIMARY KEY (localday, room)
);
CREATE TABLE IF NOT EXISTS stable_setpoints (
    room TEXT NOT NULL, mode TEXT NOT NULL, tick INTEGER NOT NULL, value TEXT NOT NULL,
    PRIMARY KEY (room,mode,tick)
);
"""


def open_readonly(path):
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=3)
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA temp_store=MEMORY")
    db.row_factory = sqlite3.Row
    return db


@contextmanager
def room_source(path):
    # HA closes its observation WAL connections between samples. SQLite then needs
    # to recreate WAL sidecars even for a reader. Copy a quiescent checkpoint to
    # an owned temporary directory rather than write to the HA configuration.
    try:
        connection = open_readonly(path)
        try:
            connection.execute("SELECT tick FROM samples LIMIT 1").fetchone()
        except BaseException:
            connection.close()
            raise
    except sqlite3.OperationalError:
        companions = [path.with_name(path.name + suffix) for suffix in ("-wal", "-shm")]
        before = path.stat()
        if any(item.exists() for item in companions):
            raise
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_mtime_ns, before.st_ino, before.st_size) != (after.st_mtime_ns, after.st_ino, after.st_size) or any(
            item.exists() for item in companions
        ):
            raise
        with TemporaryDirectory(dir=DATA) as folder:
            copy = Path(folder) / path.name
            copy.write_bytes(raw)
            with closing(sqlite3.connect(copy)) as snapshot:
                snapshot.row_factory = sqlite3.Row
                snapshot.execute("PRAGMA query_only=ON")
                yield snapshot
    else:
        with closing(connection):
            yield connection


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def local(stamp):
    return datetime.fromtimestamp(stamp, LOCAL)


def state_history(recorder, entity, start, *, end=None, tail=False):
    sql = """SELECT s.state, a.shared_attrs, s.last_updated_ts AS updated,
                    coalesce(s.last_reported_ts,s.last_updated_ts) AS reported
             FROM states s JOIN states_meta m ON m.metadata_id=s.metadata_id
             LEFT JOIN state_attributes a ON a.attributes_id=s.attributes_id
             WHERE m.entity_id=? AND s.last_updated_ts>=? """
    params = (entity, start)
    if end is not None:
        sql += "AND s.last_updated_ts < ? "
        params += (end,)
    if tail:
        return recorder.execute(sql + "ORDER BY s.last_updated_ts DESC, s.state_id DESC LIMIT 1", params).fetchone()
    return recorder.execute(sql + "ORDER BY s.last_updated_ts,s.state_id", params)


def powers(recorder, entity, start, *, end=None, fallback=None):
    points = [(float(row["reported"]), numeric(row["state"]))
              for row in state_history(recorder, entity, start, end=end) if row["reported"] is not None]
    if fallback is None:
        return points
    # Check the retained history before the window as well. A quiet primary
    # sensor may already have been active at the start of the window.
    first = next((float(row["reported"]) for row in state_history(recorder, entity, 0, end=end)
                  if row["reported"] is not None and (watts := numeric(row["state"])) is not None and watts >= 0), None)
    if first is None:
        return powers(recorder, fallback, start, end=end)
    if first <= start:
        return points
    older = powers(recorder, fallback, start, end=first)
    return [point for point in older if point[0] < first] + [point for point in points if point[0] >= first]


def power_context(attrs, entity):
    started = attrs.get("power_observer_started_at")
    if attrs.get("power_entity_id") == entity and started:
        return [entity, started]
    return None


def recent_power_reports(recorder, entity, thermostat, start, end, *, as_of=None, context=None, ledger=None):
    """Merge timestamped Recorder/observer reports with pairs already verified in this epoch."""
    upper = min(end, as_of + 1e-6) if as_of is not None else end
    if context is not None:
        if context[0] != entity:
            return []
        start = max(start, datetime.fromisoformat(context[1]).timestamp())
    if upper < start:
        return []
    def in_window(stamp):
        return start <= stamp < end and (as_of is None or stamp <= upper)

    reports = {stamp: watts for stamp, watts in powers(recorder, entity, start, end=end) if in_window(stamp)}
    if context is not None and ledger is not None:
        # Recorder's last_reported can be advanced for the same underlying
        # state row. Keep both the exact timestamps and the values we saw.
        for stamp, watts in ledger.execute(
            """SELECT report_at,watts FROM power_reports WHERE source=? AND observer_epoch=?
               AND report_at BETWEEN ? AND ?""", (*context, start, upper)
        ):
            if in_window(stamp) and (stamp not in reports or (reports[stamp] is not None and reports[stamp] >= 0)):
                reports[stamp] = watts
    for row in state_history(recorder, thermostat, start, end=end):
        attrs = json.loads(row["shared_attrs"] or "{}")
        observed_context = power_context(attrs, entity)
        if observed_context is None or (context is not None and observed_context != context):
            continue
        observer = attrs.get("power_reported_at")
        if not observer:
            continue
        stamp = datetime.fromisoformat(observer).timestamp()
        if not in_window(stamp) or stamp in reports or stamp > row["updated"]+1e-6:
            continue
        raw = state_history(recorder, entity, 0, end=stamp + 1e-6, tail=True)
        if raw and raw["reported"] is not None and raw["reported"] <= stamp + 1e-6:
            reports[stamp] = numeric(raw["state"])
    return sorted(reports.items())


def metrics(records, *, fixed_band=False):
    values = [json.loads(r["value"]) for r in records]
    valid = [x for x in values if x["measured"] is not None and x["source_fresh"]]
    errors = [abs(x["measured"] - x["target"]) for x in valid]
    minutes = len(values)
    count = sum(x["hypothetical_send"] for x in values)
    metered_seconds = sum(x.get("metered_seconds", 0) for x in values)
    points = {row["tick"]: value for row, value in zip(records, values)}
    forecast_errors = []
    for at, value in points.items():
        after = points.get(at + 900)
        if (after and value["source_fresh"] and after["source_fresh"]
            and value.get("temperature_report_at") and after.get("temperature_report_at")
            and after["temperature_report_at"] > value["temperature_report_at"]
            and value["prediction_15m_delta"] is not None
            and value["reported_mode"] == after["reported_mode"] and value["from_setpoint"] == after["from_setpoint"]):
            actual = after["measured"] - value["measured"]
            forecast_errors.append(abs(actual - value["prediction_15m_delta"]))
    episodes = []
    excursion = []
    for row, value in zip(records, values):
        tick = row["tick"]
        fresh = value["measured"] is not None and value["source_fresh"]
        error = value["measured"]-value["target"] if fresh and value["target"] is not None else None
        side = (1 if error > .7+1e-6 else -1 if error < -.7-1e-6 else 0) if error is not None else None
        if excursion and (side != excursion[-1][1] or tick != excursion[-1][0]+60
                          or value["target"] != excursion[-1][3]):
            recovered = side == 0 and tick == excursion[-1][0]+60 and value["target"] == excursion[-1][3]
            episodes.append((excursion, recovered))
            excursion = []
        if side:
            excursion.append((tick, side, abs(error), value["target"]))
    if excursion:
        episodes.append((excursion, False))
    overshoots = []
    for run, complete in episodes:
        if not complete:
            continue
        recovered_at, side, _, target = run[-1]
        displacement = 0.0
        for tick in range(recovered_at+60, recovered_at+1801, 60):
            point = points.get(tick)
            if (not point or not point["source_fresh"] or point["target"] != target
                    or point["measured"] is None):
                break
            displacement = max(displacement, -side*(point["measured"]-target))
        overshoots.append(round(displacement, 3))
    return {"minutes": minutes,
            "post_recovery_overshoot_30m_c": overshoots,
            "outside_comfort_minutes": sum(len(run) for run, _ in episodes),
            "longest_outside_comfort_minutes": max((len(run) for run, _ in episodes), default=0),
            "recovery_minutes_completed": [len(run) for run, complete in episodes if complete],
            "peak_above_target_c": round(max((x["measured"]-x["target"] for x in valid), default=0), 3) if valid else None,
            "peak_below_target_c": round(max((x["target"]-x["measured"] for x in valid), default=0), 3) if valid else None, "valid_temperature_minutes": len(valid),
            "nonunknown_power_minutes": sum(x["confidence"] != "未知" for x in values),
            "hypothetical_beeps": count,
            "projected_beeps_24h": round(count * 1440 / minutes, 1) if minutes >= 60 else None,
            "within_comfort_pct": round(100 * sum(error <= (.7 if fixed_band else x["comfort_half_band"]) + 1e-6
                                               for error, x in zip(errors, valid)) / len(errors), 1) if errors else None,
            "within_ideal_pct": round(100 * sum(error <= .5 + 1e-6 for error in errors) / len(errors), 1) if errors else None,
            "metered_kwh": round(sum(x.get("metered_kwh") or 0 for x in values), 6) if metered_seconds else None,
            "metered_seconds": round(metered_seconds, 1),
            "mae_c": round(sum(errors) / len(errors), 2) if errors else None,
            "forecasts_with_later_measurement": len(forecast_errors),
            "forecast_15m_mae_c": round(sum(forecast_errors) / len(forecast_errors), 2) if forecast_errors else None}


def init(db, tick):
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    db.executescript(LEARNING_SCHEMA)
    with db:
        db.execute(TRIALS_SCHEMA)
        db.execute("INSERT OR IGNORE INTO settings VALUES ('started',?)", (str(tick),))


def latest_model(db, room, tick, recorder, entity, fresh, *, fallback=None):
    row = db.execute("SELECT * FROM power_models WHERE room=? ORDER BY trained DESC LIMIT 1", (room,)).fetchone()
    source_key = f"power_model_source:{room}"
    source = encode([entity, fallback])
    previous_source = db.execute("SELECT value FROM settings WHERE key=?", (source_key,)).fetchone()
    source_changed = previous_source["value"] != source if previous_source else fallback is not None
    if fresh and (not row or source_changed or tick - row["trained"] >= 1800):
        candidate = calibrate(powers(recorder, entity, tick - HISTORY, end=tick + 60, fallback=fallback))
        status = "calibrated" if candidate else "insufficient"
        db.execute("INSERT OR REPLACE INTO power_models VALUES (?,?,?,?)", (room, tick, status, encode(candidate)))
        db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (source_key, source))
        return candidate, tick
    return (json.loads(row["model"]), row["trained"]) if row else (None, None)


def stable_setpoints(db, room, tick, history, session, active_since, above_low_seconds):
    record_episode(db, room, tick, history, session, active_since, above_low_seconds)
    return {mode: learned_offset(db, room, mode, tick) for mode in ("heat", "cool")}


@contextmanager
def learning_transaction(db):
    db.execute("SAVEPOINT stable_learning")
    try:
        yield
    except Exception:
        db.execute("ROLLBACK TO stable_learning")
        raise
    finally:
        db.execute("RELEASE stable_learning")


def profile_for(registry_state, sample):
    attrs = json.loads(registry_state["shared_attrs"] or "{}") if registry_state else {}
    target = sample["desired"] if sample else numeric(attrs.get("temperature"))
    return {"target": target if target is not None else 24.0,
            "quiet": attrs.get("preset_mode") == "睡眠静音",
            "offsets": {"heat": numeric(attrs.get("heating_offset")) or 0,
                        "cool": numeric(attrs.get("cooling_offset")) or 0}}


def power_span(tick, states, accepts):
    since = tick
    for at, watts in reversed(states):
        if watts is None or not accepts(watts):
            break
        since = min(at, tick)
    return tick-since


def low_power_span(tick, states, model):
    if not model or not model["edges_w"]:
        return None, 0
    cutoff = min(model["edges_w"][0], 1.25 * model["centers_w"][0])
    return cutoff, power_span(tick, states, lambda watts: 0 <= watts < cutoff)


def reversal_low_cutoff(config, model_cutoff):
    # The lowest statistical band can include variable-speed operation.
    # Only a source-specific, independently commissioned off reference caps it.
    reference = config.get("verified_off_power")
    if model_cutoff is None or not isinstance(reference, dict) or reference.get("entity") != config["power"]:
        return None
    ceiling = numeric(reference.get("ceiling_w"))
    return min(ceiling, model_cutoff) if ceiling is not None and ceiling > 0 else None


def power_evidence(at, reports, context, thresholds, previous):
    """Count elapsed time between distinct, contiguous reports in one observation context."""
    previous = previous or {}
    same_context = previous.get("context") == context
    previous_bands = previous.get("bands", {}) if same_context else {}
    started = datetime.fromisoformat(context[1]).timestamp()
    reports = sorted(dict((stamp, watts) for stamp, watts in reports if started <= stamp <= at).items())
    latest = reports[-1][0] if reports and at - reports[-1][0] <= POWER_REPORT_GAP_SECONDS else None
    result = {"context": context, "last_report": latest, "bands": {}}
    for name, threshold in thresholds.items():
        old = previous_bands.get(name, {})
        floor = old.get("floor", at)
        low = name == "low"
        old_threshold = old.get("threshold")
        if threshold is not None and old_threshold is not None:
            relaxed = threshold > old_threshold if low else threshold < old_threshold
            if relaxed:
                # Keep the suffix already proved by the previous band, without
                # admitting older readings that only the relaxed band accepts.
                floor = max(floor, old["since"] if old["since"] is not None else at)
        elif old_threshold is None:
            floor = at
        since = None
        if latest is not None and threshold is not None:
            for stamp, watts in reversed(reports):
                if (stamp < floor or watts is None or watts < 0
                        or (watts >= threshold if low else watts <= threshold)
                        or (since is not None and since - stamp > POWER_REPORT_GAP_SECONDS)):
                    break
                since = stamp
        result["bands"][name] = {"threshold": threshold, "floor": floor, "since": since,
                                 "seconds": latest - since if since is not None else 0}
    return result


def power_state_window(recorder, entity, since, end):
    # The sensor publishes changes. A valid state spans the interval until the next state or offline transition.
    prior = state_history(recorder, entity, 0, end=since, tail=True)
    states = [(since, numeric(prior["state"]))] if prior else []
    changes = state_history(recorder, entity, since, end=end).fetchall()
    states.extend((row["updated"], numeric(row["state"])) for row in changes)
    return states, changes[-1] if changes else prior


def metered_interval(reports, start, end):
    """Integrate covered report intervals, leaving outages out of coverage."""
    watt_seconds = covered = 0
    for (left, watts), (right, following) in zip(reports, reports[1:]):
        if (watts is None or following is None or watts < 0 or following < 0
                or not 0 < right-left <= POWER_REPORT_GAP_SECONDS):
            continue
        seconds = max(0, min(end, right)-max(start, left))
        if seconds:
            covered += seconds
            watt_seconds += watts * seconds
    return watt_seconds / 3_600_000, covered


def continuous_entity(recorder, entity, since, end, qualifies):
    initial = state_history(recorder, entity, 0, end=since+1e-6, tail=True)
    return bool(initial and qualifies(initial) and all(
        qualifies(row) for row in state_history(recorder, entity, since, end=end)
    ))


def load_room(db, rooms, recorder, code, config, tick, actual_beep, shared_simulated_beep, session=None, *, lock_group=None, active_lock_groups=None):
    # Read ordinary observational snapshots using each room's configured sensor.
    history = rooms.execute("SELECT * FROM samples WHERE room=? AND tick BETWEEN ? AND ? ORDER BY tick",
                            (code, tick - HISTORY, tick)).fetchall()
    latest = history[-1] if history and tick-history[-1]["tick"] <= 120 else None
    thermostat = f"climate.{code}_room_thermostat"
    virtual = state_history(recorder, thermostat, 0, end=tick + 60, tail=True)
    attrs = json.loads(virtual["shared_attrs"] or "{}") if virtual else {}
    context = power_context(attrs, config["power"])
    observed = None
    if context is not None and virtual["updated"] >= tick:
        # The local poll may run after the observer within the same minute.
        observed = attrs.get("power_reported_at")
    as_of = datetime.fromisoformat(observed).timestamp() if observed else None
    recent = recent_power_reports(recorder, config["power"], thermostat, tick - 1860 - POWER_REPORT_GAP_SECONDS,
                                  tick + 60, as_of=as_of, context=context, ledger=db)
    if (as_of is not None and 0 <= virtual["updated"]-as_of <= POWER_REPORT_GAP_SECONDS
            and latest and latest["tick"] == tick and latest["watts"] is not None
            and not any(abs(stamp-as_of) <= 1e-6 for stamp, _ in recent)):
        # Match the observer's actual report timestamp to the room sample
        # and Recorder state in this minute. The Recorder row can already
        # carry a later last_reported for the same value by the time it commits.
        raw = state_history(recorder, config["power"], 0, end=as_of+1e-6, tail=True)
        if raw and numeric(raw["state"]) == latest["watts"]:
            recent.append((as_of, latest["watts"]))
            recent.sort()
    source_fresh = bool(latest and latest["tick"] == tick and latest["temp_report"] is not None
                        and latest["measured"] is not None and -59 <= tick-latest["temp_report"] <= 1200)
    fresh_power = bool(latest and latest["tick"] == tick and latest["watts"] is not None and recent
                       and recent[-1][1] == latest["watts"] and -59 <= tick-recent[-1][0] <= 600)
    observed_at = virtual["updated"] if virtual and tick <= virtual["updated"] < tick + 60 else None
    power_without_room_sample = False
    if ((not latest or latest["tick"] != tick) and observed_at is not None and as_of is not None
            and 0 <= observed_at-as_of <= POWER_REPORT_GAP_SECONDS):
        # The meter and observer can be valid when the minute room sample did
        # not commit. Derive the exact pair from Recorder only, preserving
        # unavailable points and refusing a mutable report dated after as_of.
        raw = state_history(recorder, config["power"], 0, end=as_of+1e-6, tail=True)
        watts = numeric(raw["state"]) if raw else None
        native_at = raw["reported"] if raw else None
        if (watts is not None and 0 <= watts <= 10000 and native_at is not None
                and native_at <= as_of+1e-6 and as_of-native_at <= POWER_REPORT_GAP_SECONDS):
            exact = dict(recent)
            exact.setdefault(as_of, watts)
            recent = sorted(exact.items())
            power_without_room_sample = recent[-1] == (as_of, watts)
            fresh_power = power_without_room_sample
    evidence_status = ("recorder_snapshot_missing" if observed_at is None else
                       "observer_context_missing" if context is None else
                       "no_fresh_report")
    if (observed_at is not None and context is not None and as_of is not None
            and 0 <= observed_at-as_of <= POWER_REPORT_GAP_SECONDS and latest and latest["tick"] == tick
            and latest["watts"] is not None
            and (not fresh_power or abs(recent[-1][0]-as_of) > 1e-6)):
        evidence_status = "recorder_power_unmatched"
    model, trained = latest_model(db, code, tick, recorder, config["power"], fresh_power,
                                  fallback=config.get("power_fallback"))
    observed_since = tick
    if latest and latest["tick"] == tick:
        for row in reversed(history[:-1]):
            if observed_since-row["tick"] != 60 or tick-row["tick"] > 1860:
                break
            observed_since = row["tick"]
    _, power_state = power_state_window(recorder, config["power"], observed_since, tick+60)
    low_cutoff, _ = low_power_span(tick, (), model)
    verified_low = reversal_low_cutoff(config, low_cutoff)
    high_cutoff = model["edges_w"][-1] if model and model["edges_w"] else None
    evidence = None
    if (context is not None and observed_at is not None and as_of is not None
            and evidence_status != "recorder_power_unmatched"):
        # Durable, epoch-scoped pairs. Keep unavailable points as separators.
        # Retrying the same report cannot change a saved value or extend time.
        db.executemany("INSERT OR IGNORE INTO power_reports VALUES (?,?,?,?)",
                       [(*context, stamp, watts) for stamp, watts in recent])
    if (fresh_power and context is not None and observed_at is not None and as_of is not None
            and evidence_status != "recorder_power_unmatched"):
        prior = db.execute("""SELECT value FROM decisions WHERE room=? AND tick BETWEEN ? AND ?
                              ORDER BY tick DESC""", (code, tick-240, tick-60))
        previous = None
        for row in prior:
            saved = json.loads(row["value"])
            if saved.get("power_evidence_status") not in LATE_RECORDER:
                previous = saved.get("power_evidence")
                break
        evidence = power_evidence(observed_at, recent, context,
                                  {"low": verified_low, "above_low": verified_low, "high": high_cutoff, "learning": low_cutoff}, previous)
        evidence_status = ("power_only" if power_without_room_sample else "ready") if evidence["last_report"] is not None else "no_fresh_report"
    bands = evidence["bands"] if evidence else {}
    low_seconds = bands.get("low", {}).get("seconds", 0)
    above_low_seconds = bands.get("above_low", {}).get("seconds", 0)
    above_high_seconds = bands.get("high", {}).get("seconds", 0)
    profile = profile_for(virtual, latest)
    if latest and latest["tick"] == tick:
        activity = identify(tick, latest, [point for point in recent if point[0] >= tick - 900], model, history)
    else:
        activity = Activity("未知", "未知", "等待该分钟的 HA 观测")
    active_since = (learning_active(session, code, virtual, tick, lock_group=lock_group, active_lock_groups=active_lock_groups)
                    if source_fresh and fresh_power and not power_without_room_sample else None)
    learning_power_span = bands.get("learning", {}).get("seconds", 0)
    if active_since is not None and learning_power_span >= EPISODE_SECONDS:
        since = tick - EPISODE_SECONDS
        auto = continuous_entity(recorder, thermostat, since, tick+60,
                                 lambda row: learning_active(session, code, row, tick, lock_group=lock_group, active_lock_groups=active_lock_groups) is not None
                                 and (fields := json.loads(row["shared_attrs"] or "{}")).get("temperature_entity_id") == attrs.get("temperature_entity_id")
                                 and numeric(fields.get("temperature")) == latest["desired"]
                                 and power_context(fields, config["power"]) == context)
        steady = auto and continuous_entity(recorder, config["adapter"], since, tick+60,
            lambda row: row["state"] == latest["adapter_mode"]
            and numeric(json.loads(row["shared_attrs"] or "{}").get("temperature")) == latest["reported"])
        if not steady:
            learning_power_span = 0
    try:
        with learning_transaction(db):
            learned = stable_setpoints(db, code, tick, history, session, active_since, learning_power_span)
        learning_error = None
    except Exception as error:
        learning_error = type(error).__name__
        learned = {}
        LOG.error("Learning skipped %s %s", learning_error,
                  [(f.name, f.lineno) for f in extract_tb(error.__traceback__)])
    last_row = db.execute("SELECT tick,value FROM decisions WHERE room=? AND hypothetical=1 ORDER BY tick DESC LIMIT 1",
                          (code,)).fetchone()
    last = dict(json.loads(last_row["value"]), tick=last_row["tick"]) if last_row else None
    actual_at = actual_beep.at if actual_beep is not None else None
    anchors = (actual_at, shared_simulated_beep)
    current = advise(tick, latest, profile, history, activity, last, {}, ignore_beep=True,
                     last_beep_at=max((at for at in anchors if at is not None), default=None),
                     reversal_from_low_power=low_seconds >= 300, limits=config.get("soft_limits"))
    energy, coverage = metered_interval(recent, tick-60, tick) if context and evidence_status != "recorder_power_unmatched" else (0, 0)
    current.update({"activity": activity.state, "power_reason": activity.reason,
                    "metered_kwh": energy if coverage else None, "metered_seconds": coverage,
                    "source_fresh": source_fresh, "physical_state_verified": False,
                    "last_recorded_beep_at": actual_at,
                    "last_recorded_beep_source": actual_beep.source if actual_beep is not None else None,
                    "temperature_report_at": latest["temp_report"] if latest else None,
                    "temperature_source": [attrs.get("temperature_entity_id"), context[1]] if context else None,
                    "observed_at": observed_at,
                    "power_report_at": recent[-1][0] if fresh_power else None,
                    "power_report_w": recent[-1][1] if fresh_power else None,
                    "power_state_updated_at": power_state["updated"] if power_state else None,
                    "watts": latest["watts"] if latest else None,
                    "low_power_cutoff": low_cutoff,
                    "reversal_low_cutoff": verified_low,
                    "low_power_seconds": low_seconds,
                    "above_low_seconds": above_low_seconds,
                    "above_high_seconds": above_high_seconds,
                    "power_evidence": evidence, "power_evidence_status": evidence_status,
                    "reported_mode": latest["adapter_mode"] if latest else None,
                    "from_setpoint": latest["reported"] if latest else None,
                    "outdoor": latest["outdoor"] if latest else None,
                    "quiet": profile["quiet"], "comfort_half_band": 0.95 if profile["quiet"] else 0.7,
                    "model": model, "model_at": trained, "passive_setpoints": learned,
                    "learning_session": active_since, "learning_error": learning_error,
                    "reference": config["reference"]})
    return current


def shared_sensor_groups(config):
    by_sensor = {}
    for room, settings in config.items():
        if sensor := settings.get("temperature"):
            by_sensor.setdefault(sensor, []).append(room)
    return tuple(tuple(rooms) for rooms in by_sensor.values() if len(rooms) >= 2)


def independent_temperature_rooms(config):
    if not all(settings.get("temperature") for settings in config.values()):
        return [room for room in ("bedroom", "living") if room in config]
    seen, rooms = set(), []
    for room, settings in config.items():
        sensor = settings["temperature"]
        if sensor not in seen:
            seen.add(sensor)
            rooms.append(room)
    return rooms


def room_lock_groups(config):
    path = CONFIG.with_name("topology.json")
    if path.is_file():
        try:
            actuators = json.loads(path.read_text(encoding="utf-8")).get("actuators", {})
            if isinstance(actuators, dict) and set(actuators) == set(config):
                return {room: unit.get("lock_group") or room for room, unit in actuators.items() if isinstance(unit, dict)}
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return {room: ("living_study" if room in SHARED_ROOMS else room) for room in config}


def coordinate(current, config=None):
    groups = shared_sensor_groups(config) if config is not None else (SHARED_ROOMS,)
    for group in groups:
        if not all(room in current for room in group):
            continue
        members = [current[room] for room in group]
        proposals = [room for room in members if room["send_after"] is not None]
        if not proposals:
            continue
        if ({"heat", "cool"} <= {room["mode"] for room in members}
                or not all(room["source_fresh"] for room in members)):
            for room in proposals:
                room.update(hypothetical_send=False, send_after=None, reason="共享参考温度存在方向冲突或数据过期")
            continue
        _, *waiting = sorted(proposals, key=lambda item: item["send_after"])
        label = ("客厅与书房" if set(group) == set(SHARED_ROOMS)
                 else "与".join((config or {}).get(room, {}).get("name", room) for room in group))
        for room in waiting:
            room.update(hypothetical_send=False, send_after=None, reason=f"{label}共享参考温度，本轮只计一条建议")


def daily(db, today):
    candidates = db.execute("""SELECT DISTINCT localday FROM decisions AS d WHERE localday < ? AND NOT EXISTS (
                               SELECT 1 FROM daily AS a WHERE a.localday=d.localday AND a.room=d.room)
                               ORDER BY localday""", (today,)).fetchall()
    for row in candidates:
        for room, in db.execute("SELECT DISTINCT room FROM decisions WHERE localday=?", (row["localday"],)):
            entries = db.execute("SELECT tick,value FROM decisions WHERE localday=? AND room=? ORDER BY tick",
                                 (row["localday"], room)).fetchall()
            db.execute("INSERT OR REPLACE INTO daily VALUES (?,?,?)", (row["localday"], room, encode(metrics(entries))))


def atomic(path, body):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(body, encoding="utf-8")
    temp.replace(path)


def deployed_release():
    marker = Path(__file__).with_name(".release")
    try:
        with marker.open() as release:
            marker_time = marker.stat().st_mtime
            return release.read().strip(), marker_time
    except FileNotFoundError:
        return None, None


def version_rows(db, floor, tick, commit, room=None):
    return db.execute("""SELECT room,tick,value FROM decisions WHERE tick BETWEEN ? AND ?
                        AND json_extract(value,'$.policy_revision')=?
                        AND json_extract(value,'$.release_commit') IS ?
                        AND (? IS NULL OR room=?) ORDER BY tick,room""",
                      (floor, tick, POLICY_REVISION, commit, room, room))


def live_actuations(start, end, journals=JOURNALS):
    counted = {}
    for path in journals.glob("continuous-*.json"):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        for action in state.get("commands", state.values() if "-actions-" in path.stem else ()):
            if action["sent_at"] < end and action["status"] not in {"prepared", "canceled"}:
                counted[action["room"], action["sent_at"]] = action
    counts = {name: 0 for name in ("attempts", "proxy_echoes", "starts", "stops", "reversals")}
    last_off = {}
    for action in sorted(counted.values(), key=lambda item: item["sent_at"]):
        in_window = start <= action["sent_at"]
        counts["attempts"] += in_window
        if action["status"] != "proxy_echo":
            continue
        counts["proxy_echoes"] += in_window
        if action["kind"] != "mode":
            continue
        mode, before, room = action["value"], action["before"][0], action["room"]
        if mode == "off":
            counts["stops"] += in_window
            last_off[room] = before
        elif mode in {"cool", "heat"}:
            previous = last_off.pop(room, None) if before == "off" else before
            counts["starts"] += in_window and before == "off"
            counts["reversals"] += in_window and previous in {"cool", "heat"} and previous != mode
    return counts


def display(db, config, tick, error=None, *, release=(None, None)):
    arrivals = dict(db.execute("SELECT key,value FROM settings WHERE key IN ('started','valid_started',?)", (POLICY_START_KEY,)))
    started = int(arrivals.get("valid_started", arrivals["started"]))
    commit, deployed_at = release
    policy_start = int(arrivals[POLICY_START_KEY]) if POLICY_START_KEY in arrivals else None
    assessment = deployed_at if deployed_at is not None else policy_start
    today = local(tick).date().isoformat()
    release_floor = ceil(assessment/60)*60 if assessment is not None else tick+60
    floor = max(release_floor, tick-DAY+60)
    expected = max(1, (tick-floor)//60+1)
    current = {}
    for room in config:
        row = db.execute("SELECT tick,value FROM decisions WHERE room=? ORDER BY tick DESC LIMIT 1", (room,)).fetchone()
        latest = json.loads(row["value"]) if row else {}
        recent = version_rows(db, floor, tick, commit, room).fetchall()
        today_rows = db.execute("SELECT sum(hypothetical) FROM decisions WHERE room=? AND localday=?", (room, today)).fetchone()
        totals = metrics(recent, fixed_band=True)
        current[room] = {"latest": latest, "tick": row["tick"] if row else None,
                         "totals": totals, "today_hypothetical_beeps": today_rows[0] or 0,
                         "coverage_pct": round(100 * totals["valid_temperature_minutes"] / expected, 1)}
    live_summary = None
    try:
        live_summary = json.loads((JOURNALS/"status.json").read_bytes()).get("control")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    complete_days = db.execute("SELECT * FROM daily ORDER BY localday DESC,room LIMIT 9").fetchall()
    independent_rooms = independent_temperature_rooms(config)
    enough_data = (assessment is not None and tick+60-assessment >= DAY and
                   all(current[room]["coverage_pct"] >= 80 for room in independent_rooms if room in current))
    rules = db.execute("SELECT value FROM settings WHERE key='mijia_rules_confirmation'").fetchone()
    rules = json.loads(rules[0]) if rules else None
    shared_energy = {}
    for row in version_rows(db, floor, tick, commit):
        value = json.loads(row["value"])
        if (row["room"] in config and row["tick"]-60 >= release_floor
                and value.get("metered_seconds", 0) >= 60-1e-6):
            shared_energy.setdefault(row["tick"], []).append(value["metered_kwh"])
    complete_energy = [sum(readings) for readings in shared_energy.values() if len(readings) == len(config)]
    latest_tick = max((item["tick"] for item in current.values() if item["tick"] is not None), default=None)
    generated = time.time()
    summary = {"started_at":local(started).isoformat(), "checked_at":local(tick).isoformat(),
               "published_at": local(generated).isoformat(),
               "latest_observation_at": local(latest_tick).isoformat() if latest_tick is not None else None,
               "stale_after": local(latest_tick+240).isoformat() if latest_tick is not None else None,
               "stale": error is not None or latest_tick is None or generated-latest_tick > 240,
               "policy_revision": POLICY_REVISION, "release_commit": commit,
               "deployment_at": local(deployed_at).isoformat() if deployed_at is not None else None,
               "assessment_window_from": local(floor).isoformat(), "assessment_window_to": local(tick+60).isoformat(),
               "assessment_completed_minutes": max(0, (tick-floor)//60+1),
               "assessment_started_at": local(assessment).isoformat() if assessment is not None else None,
               "earliest_24h_review":local(assessment+DAY).isoformat() if assessment else None,
               "72h_review":local(assessment+3*DAY).isoformat() if assessment else None,
               "data_ready_for_24h_review":enough_data and not error,
               "mijia_rules_confirmation": rules, "live_control": live_summary,
               "source_error":error,"rooms":current,
               "observed_home_kwh": round(sum(complete_energy), 6) if complete_energy else None,
               "observed_home_minutes": len(complete_energy),
               "observed_home_coverage_pct": round(100 * len(complete_energy) / expected, 1),
               "independent_temperature_rooms": independent_rooms,
               "live_actuations": live_actuations(max(floor-60, assessment or floor), tick+60),
               "metered_seconds_by_room": {room: r["totals"]["metered_seconds"] for room, r in current.items()},
               "completed_days_scope": "cumulative_history",
               "completed_days":[dict(r, metrics=json.loads(r["metrics"])) for r in complete_days]}
    atomic(OUTPUT/"shadow-status.json", json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    assessment_line = (f"本部署窗口的 24 小时评估点为 {local(assessment+DAY):%Y-%m-%d %H:%M %Z}。72 小时点 {local(assessment+3*DAY):%Y-%m-%d %H:%M %Z}。"
                       if assessment else "等待首组有效房间数据，24 小时评估尚未开始。")
    lines = ["# 三房温控影子记录", "", f"开始 {local(started):%Y-%m-%d %H:%M %Z}　最新观测时段 {local(tick):%Y-%m-%d %H:%M %Z}",
             assessment_line,
             f"当前策略版本 {POLICY_REVISION}。观测与历史建议继续累计。预计蜂鸣按每条影子指令一次计算。",
             "", "| 房间 | 数据覆盖率 | 功率识别 | 拟用模式 | 建议设温 | 拟发动作及时间 | 15 分钟温度变化 | 当日拟发令 |", "|---|---:|---|---|---:|---|---:|---:|"]
    if rules:
        reported = local(rules["recorded_at"])
        lines[5:5] = ["", f"{reported:%Y-%m-%d %H:%M %Z} 已记录用户确认关闭三台相关米家自动化。评估时以该记录区分前后时段。"]
    if error:
        lines[3:3] = ["HA 历史暂不可读，暂停生成新建议。以下保留最后一次成功记录。"]
    outdoor = next((item["latest"]["outdoor"] for item in current.values()
                    if item["latest"].get("outdoor") is not None), None)
    if outdoor is not None:
        lines[5:5] = ["", f"HA 家庭位置室外气温 {outdoor:g}℃。"]
    names = {"heat":"制热", "cool":"制冷", "hold":"维持"}
    actions = {"mode":"切换模式", "temperature":"调整设温"}
    for room, item in current.items():
        record = item["latest"]
        if not record:
            continue
        after = local(record["send_after"]).strftime("%m-%d %H:%M") if record["send_after"] else "暂缓"
        if record["send_after"] and record.get("command_kind") in actions:
            after = f"{actions[record['command_kind']]} {after}"
        setpoint = f"{record['setpoint']:.1f}℃" if record["setpoint"] is not None else "—"
        predicted = record["prediction_15m_delta"]
        trend = f"{predicted:+.2f}℃" if predicted is not None else "样本不足"
        lines.append(f"| {config[room]['name']} | {item['coverage_pct']}% | {record['activity']} / {record['confidence']} | {names[record['mode']]} | {setpoint} | {after} | {trend} | {item['today_hypothetical_beeps']} |")
    lines.append("")
    for room, item in current.items():
        record = item["latest"]
        if not record:
            continue
        model = record["model"]
        groups = "、".join(f"{x:g} W" for x in model["centers_w"]) if model else "样本不足"
        projected = item["totals"]["projected_beeps_24h"]
        beep = "不足60分钟" if projected is None else f"{projected:g} 次每24小时（按当前记录外推）"
        actual_errors = item["totals"]["forecasts_with_later_measurement"]
        forecast = f"预测平均误差 {item['totals']['forecast_15m_mae_c']}℃（{actual_errors} 次对照）" if actual_errors else "暂缺后续实测用于核对预测"
        outdoor_note = (f"室外气温支持换向，室温滞回 {record['reversal_half_band']:g}℃。"
                        if record.get("weather_assisted_reversal") else "")
        lines.append(f"- **{config[room]['name']}**　{record['reason']}。{outdoor_note}功率分群中心 {groups}。预计蜂鸣 {beep}。{forecast}。")
    if complete_days:
        lines.extend(["", "## 已归档的日评估", "", "| 日期 | 房间 | 有效温度分钟 | 舒适区覆盖 | 平均绝对偏差 | 模拟发令 |", "|---|---|---:|---:|---:|---:|"])
        for row in complete_days:
            values = json.loads(row["metrics"])
            percent = f"{values['within_comfort_pct']}%" if values["within_comfort_pct"] is not None else "—"
            error = f"{values['mae_c']}℃" if values["mae_c"] is not None else "—"
            lines.append(f"| {row['localday']} | {config[row['room']]['name']} | {values['valid_temperature_minutes']} | {percent} | {error} | {values['hypothetical_beeps']} |")
    study_shared = ("living" in config and "study" in config
                    and config["living"].get("temperature") == config["study"].get("temperature"))
    shared_note = "书房使用客厅参考温度。" if study_shared else ""
    lines.extend(["", f"温度预测沿用现有米家运行趋势。{shared_note}功率分类属于推测，闭环精度与指令执行需真机验证。", ""])
    atomic(OUTPUT/"shadow-status.md", "\n".join(lines))


def decision_complete(db, config, tick):
    return db.execute("SELECT count(*) FROM decisions WHERE tick=?", (tick,)).fetchone()[0] == len(config)


def cycle(db, config, tick, *, release_commit=None, now=None):
    if decision_complete(db, config, tick):
        return False
    now = time.time() if now is None else now
    awaiting_inputs = now < tick+60+RECORDER_INPUT_GRACE_SECONDS
    with closing(open_readonly(SOURCE / "home-assistant_v2.db")) as recorder, room_source(
        SOURCE / ".storage/room_climate_control.sqlite3"
    ) as rooms, db:
        if awaiting_inputs and rooms.execute("SELECT count(*) FROM samples WHERE tick=?", (tick,)).fetchone()[0] < len(config):
            return False
        groups = shared_sensor_groups(config)
        shared_ticks = {}
        for group in groups:
            placeholders = ",".join("?" for _ in group)
            latest_shared = db.execute(
                f"SELECT max(tick) FROM decisions WHERE hypothetical=1 AND room IN ({placeholders})", group
            ).fetchone()[0]
            for room in group:
                shared_ticks[room] = latest_shared
        recorded = latest_beeps(db, now, shared_groups=groups)
        lock_groups = room_lock_groups(config)
        active_groups = set(lock_groups.values())
        session = learning_session(JOURNALS, tick)
        db.execute("SAVEPOINT input_snapshot")
        data = {room: load_room(db, rooms, recorder, room, settings, tick, recorded.get(room),
                                shared_ticks.get(room), session, lock_group=lock_groups.get(room),
                                active_lock_groups=active_groups)
                for room, settings in config.items()}
        if awaiting_inputs and any(d["power_evidence_status"] in LATE_RECORDER
                                   for d in data.values()):
            # The observer SQLite commit can lead the Recorder commit. A
            # partial snapshot is not a power interruption or a decision.
            db.execute("ROLLBACK TO input_snapshot")
            db.execute("RELEASE input_snapshot")
            return False
        coordinate(data, config)
        if all(value["source_fresh"] for value in data.values()):
            db.execute("INSERT OR IGNORE INTO settings VALUES ('valid_started',?)", (str(int(now)),))
            db.execute("INSERT OR IGNORE INTO settings VALUES (?,?)", (POLICY_START_KEY, str(tick)))
        for value in data.values():
            value["policy_revision"] = POLICY_REVISION
            value["release_commit"] = release_commit
        db.executemany("INSERT OR IGNORE INTO decisions VALUES (?,?,?,?,?)",
                       [(room,tick,local(tick).date().isoformat(),int(value["hypothetical_send"]),encode(value)) for room,value in data.items()])
        daily(db,local(tick).date().isoformat())
        db.execute("RELEASE input_snapshot")
    return True


def daily_snapshot(db, tick):
    folder = DATA / "backups"
    folder.mkdir(exist_ok=True)
    name = local(tick).date().isoformat()
    path = folder / f"shadow-{name}.sqlite3"
    if path.exists():
        return
    with db:
        db.execute("DELETE FROM power_reports WHERE report_at<?", (tick-HISTORY,))
    temporary = path.with_suffix(".tmp")
    with closing(sqlite3.connect(temporary)) as dest:
        db.backup(dest)
    temporary.replace(path)
    for obsolete in sorted(folder.glob("shadow-????-??-??.sqlite3"))[:-7]:
        obsolete.unlink()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(CONFIG.read_text())
    release = deployed_release()
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with closing(sqlite3.connect(DATA/"shadow.sqlite3")) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA temp_store=MEMORY")
        init(db, int(time.time()))
        LOG.info("Offline shadow service running for %d rooms", len(config))
        last_report = None
        last_error = None
        cursor = None
        while not stop.is_set():
            # HA writes its interval sample at any point within the minute.
            now = time.time()
            tick = int(now)//60*60 - 60
            if cursor is None:
                latest = db.execute("SELECT max(tick) FROM decisions").fetchone()[0]
                cursor = min(tick, latest+60) if latest is not None else tick
            cursor = max(cursor, tick-120)
            try:
                changed = False
                while cursor <= tick:
                    written = cycle(db, config, cursor, release_commit=release[0], now=now)
                    if not written and not decision_complete(db, config, cursor):
                        break
                    cursor += 60
                    changed |= written
                if last_report is None or changed or last_error:
                    display(db, config, tick, release=release)
                    try:
                        review = trial_review(db, config, JOURNALS, time.time(), tick, LOCAL)
                        path = OUTPUT / "live-pilot-review.md"
                        if review and (not path.exists() or path.read_text(encoding="utf-8") != review):
                            atomic(path, review)
                    except (sqlite3.Error, OSError, KeyError, ValueError) as exc:
                        LOG.warning("Live trial review unavailable: %s", type(exc).__name__)
                    last_report = tick
                    if changed:
                        daily_snapshot(db, tick)
                last_error = None
                if last_report is not None:
                    atomic(OUTPUT/"shadow-health.json", encode({"at": now, "last_success_tick": last_report,
                               "stale_after": last_report+240, "state": "stale" if now-last_report > 240 else "healthy"}))
            except (sqlite3.Error, OSError, KeyError, ValueError, AttributeError, TypeError) as exc:
                name = type(exc).__name__
                if last_error != name:
                    LOG.warning("Shadow input or storage unavailable: %s", name)
                last_error = name
                try:
                    atomic(OUTPUT/"shadow-health.json", encode({"at": now, "last_success_tick": last_report,
                               "stale_after": (last_report+240) if last_report is not None else None,
                               "state": "stale", "error": name}))
                except OSError:
                    LOG.warning("Shadow health output unavailable")
                try:
                    display(db, config, tick, name, release=release)
                except (sqlite3.Error, OSError):
                    LOG.warning("Shadow status output unavailable")
            stop.wait(10)


if __name__ == "__main__":
    main()
