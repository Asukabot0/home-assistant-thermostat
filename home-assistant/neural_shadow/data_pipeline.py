"""Read-only data ingestion, deduplication, power integration, and causal dataset building.

Key invariants:
1. All inputs are opened strictly read-only (never writes to HA Recorder, shadow, or pilot dirs).
2. Deduplicates by (source, actual report_at); repeated minute-level cache copies do NOT count
   as new measurements, while genuinely new reports with the same numeric value DO count.
3. Only independent zone primary sensors appear in temperature targets (study sharing living
   room temperature is never double-counted as a third temperature label).
4. Power integration reports exact coverage and gaps; missing measurements are NEVER treated
   as zero energy.
5. Historical actions are conditioning inputs; proxy_echo is explicitly flagged as unverified
   (not independent indoor-unit verification).
6. Causal alignment guarantees zero future temperature, power, weather, or post-hoc leakage
   into features at cutoff time t0.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZipFile

import numpy as np

from . import (
    DEFAULT_HISTORY_MINUTES,
    DataOrigin,
    FEATURE_VERSION,
    HORIZONS_MINUTES,
    SCHEMA_VERSION,
    STEP_MINUTES,
)
from .config import (
    SiteConfig,
    canonical_hash,
    load_site_config_from_acceptance,
    load_site_config_from_effective_telemetry,
    load_site_config_from_files,
    shadow_model,
    telemetry_evidence,
)

MAX_POWER_GAP_SECONDS = 120.0
DEFAULT_SETTLE_WINDOW_SECONDS = 300.0
DEFAULT_MIN_WH_COVERAGE = 1.0
MODE_LIST = ("off", "heat", "cool")


def _num(val: Any) -> float | None:
    return shadow_model.numeric(val)


def _ts(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if math.isfinite(float(val)) else None
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val).timestamp()
        except ValueError:
            return _num(val)
    return None


def _iso_utc(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


@contextmanager
def open_sqlite_readonly(path: Path):
    """Open SQLite strictly read-only, copying a quiescent file to a temp dir if WAL sidecars are absent."""
    try:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=4)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("BEGIN")
        conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        companions = [path.with_name(path.name + s) for s in ("-wal", "-shm")]
        if any(c.exists() for c in companions):
            raise
        raw = path.read_bytes()
        with TemporaryDirectory() as tmp:
            copy = Path(tmp) / path.name
            copy.write_bytes(raw)
            with closing(sqlite3.connect(copy)) as snap:
                snap.row_factory = sqlite3.Row
                snap.execute("PRAGMA query_only=ON")
                yield snap
    else:
        try:
            yield conn
        finally:
            conn.close()


@dataclass(frozen=True)
class TempReport:
    """Single deduplicated temperature report at its actual sensor report timestamp."""

    zone: str  # independent zone name OR environment sensor name
    role: str  # "zone_primary" or "environment"
    source_entity: str
    report_at: float
    value_c: float
    observed_at: float
    quality: str = "fresh"
    source_version: str | None = None
    is_exact_precision: bool = True


@dataclass(frozen=True)
class PowerReport:
    """Single deduplicated power reading within an observer epoch. watts=None marks an explicit gap."""

    unit: str
    source_entity: str
    observer_epoch: str
    report_at: float
    watts: float | None


@dataclass(frozen=True)
class ProxyStateEvent:
    """Device proxy state snapshot (adapter mode/setpoint, target, holds, quiet preset)."""

    unit: str
    at: float
    mode: str | None
    setpoint: float | None
    target_c: float | None
    quiet: bool = False
    manual_hold: bool = False
    failure_hold: bool = False
    isolated: bool = False
    zone_paused: bool = False
    source_fresh: bool = True
    uncertain: bool = False


@dataclass(frozen=True)
class ActionEvent:
    """Historical control command or detected external/manual change."""

    at: float  # effective time (proxy_at if echoed, else sent_at / observed change time)
    sent_at: float
    unit: str
    kind: str  # "mode" | "temperature" | "manual_change"
    before_mode: str | None
    before_setpoint: float | None
    after_mode: str | None
    after_setpoint: float | None
    source: str  # "controller" | "manual_external"
    receipt_status: str  # proxy_echo, accepted, prepared, unconfirmed, rejected, canceled, manual_observed
    physical_verified: bool = False  # proxy_echo MUST remain False
    purpose: str | None = None
    target_at_send: float | None = None
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PowerIntegration:
    """Covered electrical energy integration over [start, end] with explicit gap reporting."""

    wh: float | None  # Covered-only integral; None when covered_seconds == 0 (never falsely 0.0!)
    covered_seconds: float
    expected_seconds: float
    coverage_ratio: float
    gap_seconds: float
    has_gap: bool
    n_intervals: int

    @property
    def full_interval_wh(self) -> float | None:
        """Full-interval Wh ONLY when metering coverage is complete (no gaps); None on any gap."""
        if self.wh is None or self.has_gap:
            return None
        return self.wh


@dataclass
class CanonicalHistory:
    """Deduplicated, source-attributed canonical event streams ready for causal alignment."""

    site: SiteConfig
    data_origin: str
    source_description: str
    zone_temp_reports: dict[str, list[TempReport]]
    env_temp_reports: dict[str, list[TempReport]]
    power_reports: dict[str, list[PowerReport]]
    proxy_states: dict[str, list[ProxyStateEvent]]
    actions: list[ActionEvent]
    config_versions: list[dict[str, Any]] = field(default_factory=list)
    quality_counts: dict[str, int] = field(default_factory=dict)
    raw_duplicate_temp_skipped: int = 0
    raw_duplicate_power_skipped: int = 0


def deduplicate_temp_reports(raw_reports: list[TempReport]) -> tuple[list[TempReport], int]:
    """Deduplicate by (zone, source_entity, actual report_at).

    - Repeated cache copies with the same report_at collapse to ONE report (not counted as new).
    - Reconciles integer-truncated room_samples timestamps (e.g. 1795765496) with exact-precision
      Recorder/audit timestamps (e.g. 1795765496.312908) when value matches.
    - Preserves genuinely new reports at a newer report_at even when value_c is identical!
    """
    # Sort exact-precision reports first, then by report_at, then observed_at.
    ordered = sorted(
        raw_reports,
        key=lambda r: (r.zone, r.source_entity, not r.is_exact_precision, r.report_at, r.observed_at),
    )
    by_exact: dict[tuple[str, str, int], TempReport] = {}
    # Map (zone, source_entity, int(report_at)) -> exact report_at key for reconciling int-truncated samples.
    int_index: dict[tuple[str, str, int], tuple[str, str, int]] = {}
    skipped = 0

    for rep in ordered:
        if not math.isfinite(rep.report_at) or not math.isfinite(rep.value_c):
            skipped += 1
            continue
        micro = int(round(rep.report_at * 1_000_000))
        key = (rep.zone, rep.source_entity, micro)
        if key in by_exact:
            # Keep the earliest observed_at (strictly causal: when it first became known).
            prev = by_exact[key]
            if rep.observed_at < prev.observed_at:
                by_exact[key] = TempReport(
                    zone=prev.zone,
                    role=prev.role,
                    source_entity=prev.source_entity,
                    report_at=prev.report_at,
                    value_c=prev.value_c,
                    observed_at=rep.observed_at,
                    quality=prev.quality,
                    source_version=prev.source_version or rep.source_version,
                    is_exact_precision=prev.is_exact_precision or rep.is_exact_precision,
                )
            skipped += 1
            continue

        sec_key = (rep.zone, rep.source_entity, int(math.floor(rep.report_at)))
        if not rep.is_exact_precision and sec_key in int_index:
            existing = by_exact[int_index[sec_key]]
            if abs(existing.value_c - rep.value_c) <= 1e-4:
                # Same physical report, merely integer-truncated in minute samples.
                if rep.observed_at < existing.observed_at:
                    by_exact[int_index[sec_key]] = TempReport(
                        zone=existing.zone,
                        role=existing.role,
                        source_entity=existing.source_entity,
                        report_at=existing.report_at,
                        value_c=existing.value_c,
                        observed_at=rep.observed_at,
                        quality=existing.quality,
                        source_version=existing.source_version or rep.source_version,
                        is_exact_precision=True,
                    )
                skipped += 1
                continue

        by_exact[key] = rep
        if rep.is_exact_precision:
            int_index.setdefault(sec_key, key)

    deduped = sorted(by_exact.values(), key=lambda r: (r.report_at, r.observed_at, r.source_entity))
    return deduped, skipped


def deduplicate_power_reports(raw_reports: list[PowerReport]) -> tuple[list[PowerReport], int]:
    """Deduplicate by (unit, source_entity, observer_epoch, report_at).

    - Preserves watts=None separators (unavailable readings always win over cached values at same stamp).
    - Same wattage at a NEW report_at is preserved as a distinct report.
    """
    merged: dict[tuple[str, str, str, int], PowerReport] = {}
    skipped = 0
    for rep in raw_reports:
        if not math.isfinite(rep.report_at):
            skipped += 1
            continue
        watts = rep.watts
        if watts is not None and (not math.isfinite(watts) or watts < 0.0 or watts > 10000.0):
            watts = None
        key = (rep.unit, rep.source_entity, rep.observer_epoch, int(round(rep.report_at * 1_000_000)))
        if key in merged:
            skipped += 1
            if watts is None and merged[key].watts is not None:
                merged[key] = PowerReport(rep.unit, rep.source_entity, rep.observer_epoch, rep.report_at, None)
        else:
            merged[key] = PowerReport(rep.unit, rep.source_entity, rep.observer_epoch, rep.report_at, watts)
    deduped = sorted(merged.values(), key=lambda r: (r.report_at, r.observer_epoch, r.source_entity))
    return deduped, skipped


def integrate_power_interval(
    reports: list[PowerReport],
    start: float,
    end: float,
    *,
    max_gap_s: float = MAX_POWER_GAP_SECONDS,
) -> PowerIntegration:
    """Integrate electrical Wh over [start, end] reusing evidence.power_ranges semantics.

    Missing or gapped intervals are excluded from coverage and NEVER counted as 0 Wh.
    """
    expected = max(0.0, float(end - start))
    if expected <= 0.0:
        return PowerIntegration(None, 0.0, 0.0, 0.0, 0.0, True, 0)

    # Filter to reports that could bracket [start, end].
    bracketed = [
        {"observer_epoch": f"{r.source_entity}|{r.observer_epoch}", "report_at": r.report_at, "watts": r.watts}
        for r in reports
        if start - max_gap_s <= r.report_at <= end + max_gap_s
    ]
    if not bracketed:
        return PowerIntegration(None, 0.0, expected, 0.0, expected, True, 0)

    if abs(max_gap_s - telemetry_evidence.MAX_POWER_GAP) < 1e-9:
        intervals = telemetry_evidence.power_ranges(bracketed, start, end)
    else:
        epochs: dict[str, dict[float, float | None]] = defaultdict(dict)
        for row in bracketed:
            epochs[row["observer_epoch"]][row["report_at"]] = row["watts"]
        raw_ranges = []
        for series in epochs.values():
            values = sorted(series.items())
            for (a, w), (b, v) in zip(values, values[1:]):
                left, right = max(a, start), min(b, end)
                if w is not None and v is not None and w >= 0 and v >= 0 and 0 < b - a <= max_gap_s and right > left:
                    raw_ranges.append((left, right, w))
        raw_ranges.sort()
        intervals = []
        for item in raw_ranges:
            if intervals and item[0] < intervals[-1][1]:
                left, _, watts = intervals[-1]
                intervals[-1] = (left, item[0], watts)
            intervals.append(item)

    covered = sum(max(0.0, b - a) for a, b, _ in intervals)
    covered = min(covered, expected)
    watt_seconds = sum(max(0.0, b - a) * float(w) for a, b, w in intervals)
    coverage_ratio = covered / expected if expected > 0 else 0.0
    gap_seconds = max(0.0, expected - covered)
    has_gap = coverage_ratio < 1.0 - 1e-6
    wh = (watt_seconds / 3600.0) if covered > 0.0 else None
    return PowerIntegration(
        wh=wh,
        covered_seconds=covered,
        expected_seconds=expected,
        coverage_ratio=coverage_ratio,
        gap_seconds=gap_seconds,
        has_gap=has_gap,
        n_intervals=len(intervals),
    )


def _action_quality_flags(status: str, source: str, physical_verified: bool) -> tuple[str, ...]:
    flags: list[str] = []
    if status in {"rejected", "unconfirmed", "prepared", "transport_failed"}:
        flags.append("failed_call")
    if source == "manual_external":
        flags.append("manual_operation")
    if status == "proxy_echo" and not physical_verified:
        flags.append("proxy_only_unverified")
    return tuple(flags)


def _detect_manual_actions(
    proxy_by_unit: dict[str, list[ProxyStateEvent]],
    controller_actions: list[ActionEvent],
) -> list[ActionEvent]:
    """Infer external/manual mode or setpoint changes not explained by controller commands."""
    manual: list[ActionEvent] = []
    cmds_by_unit: dict[str, list[ActionEvent]] = defaultdict(list)
    for act in controller_actions:
        cmds_by_unit[act.unit].append(act)

    for unit, states in proxy_by_unit.items():
        if len(states) < 2:
            continue
        unit_cmds = cmds_by_unit.get(unit, [])
        prev = states[0]
        for cur in states[1:]:
            if cur.mode in {None, "unavailable", "unknown"} or prev.mode in {None, "unavailable", "unknown"}:
                prev = cur
                continue
            mode_changed = cur.mode != prev.mode
            sp_changed = (
                cur.mode in {"heat", "cool"}
                and prev.mode == cur.mode
                and cur.setpoint is not None
                and prev.setpoint is not None
                and abs(cur.setpoint - prev.setpoint) > 1e-4
            )
            if mode_changed or sp_changed:
                matched = any(
                    (c.sent_at - 10.0 <= cur.at <= c.at + 240.0)
                    and (
                        (mode_changed and c.after_mode == cur.mode)
                        or (sp_changed and c.after_setpoint is not None and abs(c.after_setpoint - cur.setpoint) <= 1e-4)
                    )
                    for c in unit_cmds
                )
                if not matched:
                    manual.append(
                        ActionEvent(
                            at=cur.at,
                            sent_at=cur.at,
                            unit=unit,
                            kind="manual_change",
                            before_mode=prev.mode,
                            before_setpoint=prev.setpoint,
                            after_mode=cur.mode,
                            after_setpoint=cur.setpoint,
                            source="manual_external",
                            receipt_status="manual_observed",
                            physical_verified=False,
                            purpose="external_or_manual_change",
                            target_at_send=cur.target_c,
                            quality_flags=("manual_operation",),
                        )
                    )
            prev = cur
    return manual


def _build_zone_lookup(site: SiteConfig) -> tuple[dict[str, str], dict[str, str]]:
    """Map primary temperature entity -> zone name, and unit -> home zone."""
    entity_to_zone = {z.primary_entity: z.name for z in site.zones}
    unit_to_zone = {a.name: a.home_zone for a in site.actuators}
    return entity_to_zone, unit_to_zone


# ---------------------------------------------------------------------------
# Adapter 1: Production read-only SQLite + pilot-data journals
# ---------------------------------------------------------------------------

def ingest_production_readonly(
    production_root: Path,
    *,
    rooms_file: Path | None = None,
    topology_file: Path | None = None,
    policy_file: Path | None = None,
    start_ts: float | None = None,
    end_ts: float | None = None,
    data_origin: str = DataOrigin.REAL.value,
) -> CanonicalHistory:
    """Ingest read-only history from a production Home Assistant thermostat installation."""
    root = production_root.resolve()
    shadow_dir = root / "shadow"
    rooms_path = rooms_file or (shadow_dir / "rooms.json")
    topo_path = topology_file or (shadow_dir / "topology.json")
    pol_path = policy_file or (shadow_dir / "house_policy.json")
    site = load_site_config_from_files(rooms_path, topo_path, pol_path)

    entity_to_zone, unit_to_zone = _build_zone_lookup(site)
    power_entity_to_unit: dict[str, str] = {}
    adapter_entity_to_unit: dict[str, str] = {}
    for act in site.actuators:
        power_entity_to_unit[act.power_entity] = act.name
        if act.power_fallback:
            power_entity_to_unit[act.power_fallback] = act.name
        adapter_entity_to_unit[act.adapter_entity] = act.name

    raw_temp: list[TempReport] = []
    raw_env: list[TempReport] = []
    raw_power: list[PowerReport] = []
    proxy_by_unit: dict[str, list[ProxyStateEvent]] = defaultdict(list)
    actions: list[ActionEvent] = []
    config_versions: list[dict[str, Any]] = []
    quality_counts: Counter[str] = Counter()

    def in_range(t: float | None) -> bool:
        if t is None:
            return False
        if start_ts is not None and t < start_ts - 7200:
            return False
        if end_ts is not None and t > end_ts + 3600:
            return False
        return True

    # 1. Read shadow-data/shadow.sqlite3 (durable power_reports + decisions)
    shadow_db_path = root / "shadow-data" / "shadow.sqlite3"
    if shadow_db_path.is_file():
        with open_sqlite_readonly(shadow_db_path) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "power_reports" in tables:
                sql = "SELECT source, observer_epoch, report_at, watts FROM power_reports"
                params: list[Any] = []
                clauses = []
                if start_ts is not None:
                    clauses.append("report_at >= ?")
                    params.append(start_ts - 7200)
                if end_ts is not None:
                    clauses.append("report_at <= ?")
                    params.append(end_ts + 3600)
                if clauses:
                    sql += " WHERE " + " AND ".join(clauses)
                for row in db.execute(sql, params):
                    src = row["source"]
                    unit = power_entity_to_unit.get(src)
                    if unit is None:
                        continue
                    raw_power.append(
                        PowerReport(
                            unit=unit,
                            source_entity=src,
                            observer_epoch=str(row["observer_epoch"]),
                            report_at=float(row["report_at"]),
                            watts=_num(row["watts"]),
                        )
                    )
            if "decisions" in tables:
                sql = "SELECT room, tick, value FROM decisions"
                params = []
                clauses = []
                if start_ts is not None:
                    clauses.append("tick >= ?")
                    params.append(int(start_ts - 7200))
                if end_ts is not None:
                    clauses.append("tick <= ?")
                    params.append(int(end_ts + 3600))
                if clauses:
                    sql += " WHERE " + " AND ".join(clauses)
                for row in db.execute(sql, params):
                    unit = row["room"]
                    if unit not in unit_to_zone:
                        continue
                    tick = float(row["tick"])
                    val = json.loads(row["value"])
                    obs_at = _ts(val.get("observed_at")) or tick
                    t_rep = _ts(val.get("temperature_report_at"))
                    measured = _num(val.get("measured"))
                    tsrc = val.get("temperature_source")
                    src_entity = tsrc[0] if isinstance(tsrc, list) and tsrc else site.actuator_by_name(unit).temperature_entity
                    zone = entity_to_zone.get(src_entity)
                    if zone is not None and t_rep is not None and measured is not None and 0 < measured <= 50:
                        raw_temp.append(
                            TempReport(
                                zone=zone,
                                role="zone_primary",
                                source_entity=src_entity,
                                report_at=t_rep,
                                value_c=measured,
                                observed_at=obs_at,
                                quality="fresh" if val.get("source_fresh") else "stale",
                                is_exact_precision=abs(t_rep - round(t_rep)) > 1e-6,
                            )
                        )
                    outdoor = _num(val.get("outdoor"))
                    if outdoor is not None and -50 <= outdoor <= 65:
                        raw_env.append(
                            TempReport(
                                zone="outdoor_reference",
                                role="environment",
                                source_entity="outdoor_reference",
                                report_at=tick,
                                value_c=outdoor,
                                observed_at=obs_at,
                                quality="fresh",
                                is_exact_precision=False,
                            )
                        )
                    p_rep = _ts(val.get("power_report_at"))
                    p_w = _num(val.get("power_report_w"))
                    pe = val.get("power_evidence") or {}
                    pctx = pe.get("context")
                    if p_rep is not None and isinstance(pctx, list) and len(pctx) == 2:
                        raw_power.append(
                            PowerReport(
                                unit=unit,
                                source_entity=str(pctx[0]),
                                observer_epoch=str(pctx[1]),
                                report_at=p_rep,
                                watts=p_w,
                            )
                        )
                    proxy_by_unit[unit].append(
                        ProxyStateEvent(
                            unit=unit,
                            at=obs_at,
                            mode=val.get("reported_mode"),
                            setpoint=_num(val.get("from_setpoint")),
                            target_c=_num(val.get("target")),
                            quiet=bool(val.get("quiet", False)),
                            source_fresh=bool(val.get("source_fresh", True)),
                            uncertain=val.get("power_evidence_status") not in {"ready", "power_only", None},
                        )
                    )

    # 2. Read config/.storage/room_climate_control.sqlite3 (samples)
    samples_db_path = root / "config" / ".storage" / "room_climate_control.sqlite3"
    if samples_db_path.is_file():
        with open_sqlite_readonly(samples_db_path) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "samples" in tables:
                sql = "SELECT room, tick, measured, desired, reported, adapter_mode, watts, outdoor, temp_report, quality FROM samples"
                params = []
                clauses = []
                if start_ts is not None:
                    clauses.append("tick >= ?")
                    params.append(int(start_ts - 7200))
                if end_ts is not None:
                    clauses.append("tick <= ?")
                    params.append(int(end_ts + 3600))
                if clauses:
                    sql += " WHERE " + " AND ".join(clauses)
                for row in db.execute(sql, params):
                    unit = row["room"]
                    if unit not in unit_to_zone:
                        continue
                    tick = float(row["tick"])
                    act = site.actuator_by_name(unit)
                    zone = entity_to_zone.get(act.temperature_entity)
                    t_rep = _ts(row["temp_report"])
                    measured = _num(row["measured"])
                    if zone is not None and t_rep is not None and measured is not None and 0 < measured <= 50:
                        raw_temp.append(
                            TempReport(
                                zone=zone,
                                role="zone_primary",
                                source_entity=act.temperature_entity,
                                report_at=t_rep,
                                value_c=measured,
                                observed_at=tick,
                                quality=str(row["quality"] or "observing"),
                                is_exact_precision=False,
                            )
                        )
                    outdoor = _num(row["outdoor"])
                    if outdoor is not None and -50 <= outdoor <= 65:
                        raw_env.append(
                            TempReport(
                                zone="outdoor_reference",
                                role="environment",
                                source_entity="outdoor_reference",
                                report_at=tick,
                                value_c=outdoor,
                                observed_at=tick,
                                quality="fresh",
                                is_exact_precision=False,
                            )
                        )
                    proxy_by_unit[unit].append(
                        ProxyStateEvent(
                            unit=unit,
                            at=tick,
                            mode=row["adapter_mode"],
                            setpoint=_num(row["reported"]),
                            target_c=_num(row["desired"]),
                            source_fresh=measured is not None and t_rep is not None and (tick - t_rep) <= 1200,
                        )
                    )

    # 3. Read config/home-assistant_v2.db (Recorder states for sensors, power, adapters, future balcony sensor)
    recorder_db_path = root / "config" / "home-assistant_v2.db"
    if recorder_db_path.is_file():
        with open_sqlite_readonly(recorder_db_path) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if {"states", "states_meta"}.issubset(tables):
                entities = set(entity_to_zone.keys()) | set(power_entity_to_unit.keys()) | set(adapter_entity_to_unit.keys())
                env_entity_to_name: dict[str, tuple[str, str | None]] = {}
                for es in site.active_environment_sensors:
                    if es.entity:
                        entities.add(es.entity)
                        env_entity_to_name[es.entity] = (es.name, es.attribute)
                for entity in sorted(entities):
                    sql = """
                        SELECT s.state, a.shared_attrs, s.last_updated_ts AS updated,
                               coalesce(s.last_reported_ts, s.last_updated_ts) AS reported
                        FROM states s
                        JOIN states_meta m ON m.metadata_id = s.metadata_id
                        LEFT JOIN state_attributes a ON a.attributes_id = s.attributes_id
                        WHERE m.entity_id = ?
                    """
                    params = [entity]
                    if start_ts is not None:
                        sql += " AND s.last_updated_ts >= ?"
                        params.append(start_ts - 7200)
                    if end_ts is not None:
                        sql += " AND s.last_updated_ts <= ?"
                        params.append(end_ts + 3600)
                    sql += " ORDER BY s.last_updated_ts, s.state_id"
                    for row in db.execute(sql, params):
                        rep_ts = _ts(row["reported"])
                        upd_ts = _ts(row["updated"])
                        if rep_ts is None or upd_ts is None:
                            continue
                        if entity in entity_to_zone:
                            val = _num(row["state"])
                            if val is not None and 0 < val <= 50:
                                raw_temp.append(
                                    TempReport(
                                        zone=entity_to_zone[entity],
                                        role="zone_primary",
                                        source_entity=entity,
                                        report_at=rep_ts,
                                        value_c=val,
                                        observed_at=upd_ts,
                                        quality="fresh",
                                        is_exact_precision=True,
                                    )
                                )
                        elif entity in env_entity_to_name:
                            ename, attr = env_entity_to_name[entity]
                            attrs = json.loads(row["shared_attrs"] or "{}") if row["shared_attrs"] else {}
                            val = _num(attrs.get(attr)) if attr else _num(row["state"])
                            if val is not None and -50 <= val <= 65:
                                raw_env.append(
                                    TempReport(
                                        zone=ename,
                                        role="environment",
                                        source_entity=entity,
                                        report_at=rep_ts,
                                        value_c=val,
                                        observed_at=upd_ts,
                                        quality="fresh",
                                        is_exact_precision=True,
                                    )
                                )
                        elif entity in power_entity_to_unit:
                            unit = power_entity_to_unit[entity]
                            raw_power.append(
                                PowerReport(
                                    unit=unit,
                                    source_entity=entity,
                                    observer_epoch="recorder",
                                    report_at=rep_ts,
                                    watts=_num(row["state"]),
                                )
                            )
                        elif entity in adapter_entity_to_unit:
                            unit = adapter_entity_to_unit[entity]
                            attrs = json.loads(row["shared_attrs"] or "{}") if row["shared_attrs"] else {}
                            proxy_by_unit[unit].append(
                                ProxyStateEvent(
                                    unit=unit,
                                    at=upd_ts,
                                    mode=str(row["state"]),
                                    setpoint=_num(attrs.get("temperature")),
                                    target_c=None,
                                )
                            )

    # 4. Read pilot-data (commands, receipts, live-audit, startup config versions)
    pilot_dir = root / "pilot-data"
    seen_cmd_keys: set[tuple[str, int]] = set()
    if pilot_dir.is_dir():
        for path in sorted(pilot_dir.glob("continuous-*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            cmd_list = list(raw.values()) if "-actions-" in path.stem else raw.get("commands", [])
            for cmd in cmd_list:
                unit = cmd.get("room")
                if unit not in unit_to_zone:
                    continue
                sent_at = _ts(cmd.get("sent_at"))
                if not in_range(sent_at):
                    continue
                assert sent_at is not None
                key = (unit, int(round(sent_at * 1000)))
                if key in seen_cmd_keys:
                    continue
                seen_cmd_keys.add(key)
                status = str(cmd.get("status", "unknown"))
                if status == "canceled":
                    continue
                before = cmd.get("before") or [None, None, None]
                b_mode = before[0] if len(before) >= 1 else None
                b_sp = _num(before[1]) if len(before) >= 2 else None
                kind = str(cmd.get("kind", "unknown"))
                a_mode = str(cmd.get("value")) if kind == "mode" else b_mode
                a_sp = _num(cmd.get("value")) if kind == "temperature" else b_sp
                proxy_at = _ts(cmd.get("proxy_at"))
                eff_at = proxy_at if proxy_at is not None else sent_at
                # proxy_echo MUST NOT be treated as independent physical verification.
                phys_ver = bool(cmd.get("physical_verified", False)) and status != "proxy_echo"
                flags = _action_quality_flags(status, "controller", phys_ver)
                for f in flags:
                    quality_counts[f] += 1
                actions.append(
                    ActionEvent(
                        at=eff_at,
                        sent_at=sent_at,
                        unit=unit,
                        kind=kind,
                        before_mode=b_mode,
                        before_setpoint=b_sp,
                        after_mode=a_mode,
                        after_setpoint=a_sp,
                        source="controller",
                        receipt_status=status,
                        physical_verified=phys_ver,
                        purpose=cmd.get("purpose"),
                        target_at_send=_num(cmd.get("target_at_send")),
                        quality_flags=flags,
                    )
                )

        # Read live-audit for cycle-level holds, exact temp/power reports, and startup versions
        for audit_path in sorted(pilot_dir.glob("live-audit.jsonl*")):
            try:
                with audit_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        at = _ts(row.get("at"))
                        if not in_range(at):
                            continue
                        assert at is not None
                        ev = row.get("event")
                        if ev == "startup":
                            config_versions.append(
                                {
                                    "at": at,
                                    "commit": row.get("commit"),
                                    "config_sha256": row.get("config_sha256"),
                                    "observation_version": row.get("observation_version"),
                                }
                            )
                        elif ev == "cycle":
                            assessment = row.get("assessment") or {}
                            for zname, zinfo in (assessment.get("zones") or {}).items():
                                if zname not in site.zone_names:
                                    continue
                                pr = zinfo.get("primary_reading") or {}
                                pt = _ts(pr.get("at"))
                                pv = _num(pr.get("value"))
                                if pt is not None and pv is not None and 0 < pv <= 50:
                                    raw_temp.append(
                                        TempReport(
                                            zone=zname,
                                            role="zone_primary",
                                            source_entity=site.zone_by_name(zname).primary_entity,
                                            report_at=pt,
                                            value_c=pv,
                                            observed_at=at,
                                            quality=str(pr.get("quality", "fresh")),
                                            source_version=zinfo.get("source_version"),
                                            is_exact_precision=True,
                                        )
                                    )
                            for ename, einfo in (assessment.get("environment") or {}).items():
                                et = _ts(einfo.get("at"))
                                ev_val = _num(einfo.get("value"))
                                if et is not None and ev_val is not None and -50 <= ev_val <= 65:
                                    raw_env.append(
                                        TempReport(
                                            zone=ename,
                                            role="environment",
                                            source_entity=ename,
                                            report_at=et,
                                            value_c=ev_val,
                                            observed_at=at,
                                            quality=str(einfo.get("quality", "fresh")),
                                            is_exact_precision=True,
                                        )
                                    )
                            for uname, uinfo in (row.get("rooms") or {}).items():
                                if uname not in unit_to_zone:
                                    continue
                                pctx = uinfo.get("power_context")
                                prep = _ts(uinfo.get("power_report_at"))
                                if isinstance(pctx, list) and len(pctx) == 2 and prep is not None:
                                    raw_power.append(
                                        PowerReport(
                                            unit=uname,
                                            source_entity=str(pctx[0]),
                                            observer_epoch=str(pctx[1]),
                                            report_at=prep,
                                            watts=_num(uinfo.get("power_w")),
                                        )
                                    )
                                status_str = str(uinfo.get("status", ""))
                                proxy_by_unit[uname].append(
                                    ProxyStateEvent(
                                        unit=uname,
                                        at=at,
                                        mode=uinfo.get("mode_reported"),
                                        setpoint=_num(uinfo.get("setpoint_reported")),
                                        target_c=_num(uinfo.get("zone_target_c") or uinfo.get("target_c")),
                                        manual_hold="外部修改" in status_str,
                                        failure_hold="暂停复发" in status_str,
                                        isolated="手动隔离" in status_str,
                                        zone_paused=uinfo.get("zone_reason") == "zone_paused",
                                        source_fresh=(_num(uinfo.get("temperature_age_s")) or 9999) <= 1200,
                                    )
                                )
            except OSError:
                continue

    return _finalize_canonical(
        site=site,
        data_origin=data_origin,
        source_description=f"production:{canonical_hash(str(root))[:12]}",
        raw_temp=raw_temp,
        raw_env=raw_env,
        raw_power=raw_power,
        proxy_by_unit=proxy_by_unit,
        actions=actions,
        config_versions=config_versions,
        quality_counts=quality_counts,
    )


# ---------------------------------------------------------------------------
# Adapter 2: Telemetry & Acceptance ZIP Archives (read-only)
# ---------------------------------------------------------------------------

def ingest_evidence_archives(
    archive_paths: list[Path],
    *,
    data_origin: str = DataOrigin.REAL_SHORT_PACK.value,
) -> CanonicalHistory:
    """Ingest one or more exported telemetry or acceptance ZIP archives read-only."""
    if not archive_paths:
        raise ValueError("Provide at least one archive path")

    site: SiteConfig | None = None
    archive_kind: str | None = None
    for path in archive_paths:
        with ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if "effective-config.json" in names:
                cfg = json.loads(zf.read("effective-config.json"))
                site = load_site_config_from_effective_telemetry(cfg)
                archive_kind = "telemetry"
                break
            if "effective-config.redacted.json" in names:
                cfg = json.loads(zf.read("effective-config.redacted.json"))
                site = load_site_config_from_acceptance(cfg)
                archive_kind = "acceptance"
                break
    if site is None:
        raise ValueError("Archive does not contain effective-config.json or effective-config.redacted.json")

    entity_to_zone, unit_to_zone = _build_zone_lookup(site)
    power_entity_to_unit: dict[str, str] = {}
    for act in site.actuators:
        power_entity_to_unit[act.power_entity] = act.name
        if act.power_fallback:
            power_entity_to_unit[act.power_fallback] = act.name

    raw_temp: list[TempReport] = []
    raw_env: list[TempReport] = []
    raw_power: list[PowerReport] = []
    proxy_by_unit: dict[str, list[ProxyStateEvent]] = defaultdict(list)
    actions: list[ActionEvent] = []
    seen_cmd_keys: set[tuple[str, int]] = set()
    config_versions: list[dict[str, Any]] = []
    quality_counts: Counter[str] = Counter()

    def _read_jsonl(zf: ZipFile, member: str) -> list[dict[str, Any]]:
        if member not in zf.namelist():
            return []
        return [json.loads(line) for line in zf.read(member).splitlines() if line.strip()]

    primary_env_name = site.active_environment_sensors[0].name if site.active_environment_sensors else "outdoor_reference"
    for path in sorted(archive_paths):
        with ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            # Power reports
            for member in ("raw/power-reports.jsonl", "raw/durable-power-reports.jsonl", "context/durable-power-reports.jsonl"):
                for row in _read_jsonl(zf, member):
                    unit = row.get("room") if row.get("room") in unit_to_zone else power_entity_to_unit.get(str(row.get("source", "")))
                    if unit is None:
                        continue
                    rep_at = _ts(row.get("report_at"))
                    if rep_at is None:
                        continue
                    raw_power.append(
                        PowerReport(
                            unit=unit,
                            source_entity=str(row.get("source", site.actuator_by_name(unit).power_entity)),
                            observer_epoch=str(row.get("observer_epoch", "default")),
                            report_at=rep_at,
                            watts=_num(row.get("watts")),
                        )
                    )

            # Temperature reports
            for row in _read_jsonl(zf, "raw/temperature-reports.jsonl"):
                rep_at = _ts(row.get("report_at"))
                val = _num(row.get("temperature") if "temperature" in row else row.get("c"))
                if rep_at is None or val is None:
                    continue
                role = row.get("role")
                if role == "environment":
                    raw_env.append(
                        TempReport(
                            zone=str(row.get("sensor", "outdoor_reference")),
                            role="environment",
                            source_entity=str(row.get("sensor", "outdoor_reference")),
                            report_at=rep_at,
                            value_c=val,
                            observed_at=rep_at,
                            quality=str(row.get("quality", "fresh")),
                            is_exact_precision=True,
                        )
                    )
                else:
                    zone = row.get("zone")
                    if zone not in site.zone_names:
                        src = row.get("source")
                        if isinstance(src, str) and "bedroom" in src:
                            zone = "sleep" if "sleep" in site.zone_names else site.zone_names[0]
                        elif isinstance(src, str) and "living" in src:
                            zone = "lounge" if "lounge" in site.zone_names else site.zone_names[-1]
                        else:
                            zone = entity_to_zone.get(src[0] if isinstance(src, list) and src else str(src))
                    if zone in site.zone_names and 0 < val <= 50:
                        raw_temp.append(
                            TempReport(
                                zone=zone,
                                role="zone_primary",
                                source_entity=site.zone_by_name(zone).primary_entity,
                                report_at=rep_at,
                                value_c=val,
                                observed_at=rep_at,
                                quality=str(row.get("quality", "fresh")),
                                is_exact_precision=True,
                            )
                        )

            # Room samples (minute copies -> deduplicated against exact temp reports!)
            for member in ("context/room-samples.jsonl", "raw/room-samples.jsonl"):
                for row in _read_jsonl(zf, member):
                    unit = row.get("room")
                    if unit not in unit_to_zone:
                        continue
                    tick = _ts(row.get("tick"))
                    if tick is None:
                        continue
                    act = site.actuator_by_name(unit)
                    zone = entity_to_zone.get(act.temperature_entity, act.home_zone)
                    t_rep = _ts(row.get("temp_report"))
                    measured = _num(row.get("measured"))
                    if zone in site.zone_names and t_rep is not None and measured is not None and 0 < measured <= 50:
                        raw_temp.append(
                            TempReport(
                                zone=zone,
                                role="zone_primary",
                                source_entity=site.zone_by_name(zone).primary_entity,
                                report_at=t_rep,
                                value_c=measured,
                                observed_at=tick,
                                quality=str(row.get("quality", "observing")),
                                is_exact_precision=False,
                            )
                        )
                    outdoor = _num(row.get("outdoor"))
                    if outdoor is not None and -50 <= outdoor <= 65:
                        raw_env.append(
                            TempReport(
                                zone=primary_env_name,
                                role="environment",
                                source_entity=primary_env_name,
                                report_at=tick,
                                value_c=outdoor,
                                observed_at=tick,
                                quality="fresh",
                                is_exact_precision=False,
                            )
                        )
                    proxy_by_unit[unit].append(
                        ProxyStateEvent(
                            unit=unit,
                            at=tick,
                            mode=row.get("adapter_mode"),
                            setpoint=_num(row.get("reported")),
                            target_c=_num(row.get("desired")),
                            source_fresh=measured is not None and t_rep is not None and (tick - t_rep) <= 1200,
                        )
                    )

            # Shadow decisions
            for member in ("context/shadow-decisions.jsonl", "raw/shadow-decisions.jsonl"):
                for row in _read_jsonl(zf, member):
                    unit = row.get("room")
                    if unit not in unit_to_zone:
                        continue
                    tick = _ts(row.get("tick"))
                    if tick is None:
                        continue
                    val = row.get("decision") if isinstance(row.get("decision"), dict) else row
                    obs_at = _ts(val.get("observed_at")) or tick
                    t_rep = _ts(val.get("temperature_report_at"))
                    measured = _num(val.get("measured"))
                    zone = site.actuator_by_name(unit).home_zone
                    if zone in site.zone_names and t_rep is not None and measured is not None and 0 < measured <= 50:
                        raw_temp.append(
                            TempReport(
                                zone=zone,
                                role="zone_primary",
                                source_entity=site.zone_by_name(zone).primary_entity,
                                report_at=t_rep,
                                value_c=measured,
                                observed_at=obs_at,
                                quality="fresh" if val.get("source_fresh") else "stale",
                                is_exact_precision=abs(t_rep - round(t_rep)) > 1e-6,
                            )
                        )
                    proxy_by_unit[unit].append(
                        ProxyStateEvent(
                            unit=unit,
                            at=obs_at,
                            mode=val.get("reported_mode"),
                            setpoint=_num(val.get("from_setpoint")),
                            target_c=_num(val.get("target")),
                            source_fresh=bool(val.get("source_fresh", True)),
                            uncertain=val.get("power_evidence_status") not in {"ready", "power_only", None},
                        )
                    )

            # Commands
            for member in ("context/controller-commands.jsonl", "raw/controller-commands.jsonl", "raw/commands.jsonl"):
                for cmd in _read_jsonl(zf, member):
                    unit = cmd.get("room")
                    if unit not in unit_to_zone:
                        continue
                    sent_at = _ts(cmd.get("sent_at"))
                    if sent_at is None:
                        continue
                    key = (unit, int(round(sent_at * 1000)))
                    if key in seen_cmd_keys:
                        continue
                    seen_cmd_keys.add(key)
                    status = str(cmd.get("status", "unknown"))
                    if status == "canceled":
                        continue
                    before = cmd.get("before") or [None, None, None]
                    b_mode = before[0] if len(before) >= 1 else None
                    b_sp = _num(before[1]) if len(before) >= 2 else None
                    kind = str(cmd.get("kind", "unknown"))
                    a_mode = str(cmd.get("value")) if kind == "mode" else b_mode
                    a_sp = _num(cmd.get("value")) if kind == "temperature" else b_sp
                    proxy_at = _ts(cmd.get("proxy_at"))
                    eff_at = proxy_at if proxy_at is not None else sent_at
                    phys_ver = bool(cmd.get("physical_verified", False)) and status != "proxy_echo"
                    flags = _action_quality_flags(status, "controller", phys_ver)
                    for f in flags:
                        quality_counts[f] += 1
                    actions.append(
                        ActionEvent(
                            at=eff_at,
                            sent_at=sent_at,
                            unit=unit,
                            kind=kind,
                            before_mode=b_mode,
                            before_setpoint=b_sp,
                            after_mode=a_mode,
                            after_setpoint=a_sp,
                            source="controller",
                            receipt_status=status,
                            physical_verified=phys_ver,
                            purpose=cmd.get("purpose"),
                            target_at_send=_num(cmd.get("target_at_send")),
                            quality_flags=flags,
                        )
                    )

            # Live audit (startup versions + fresh zone readings)
            for member in ("context/live-audit.jsonl", "raw/live-audit.jsonl"):
                for row in _read_jsonl(zf, member):
                    at = _ts(row.get("at"))
                    if at is None:
                        continue
                    if row.get("event") == "startup":
                        config_versions.append(
                            {
                                "at": at,
                                "commit": row.get("commit"),
                                "config_sha256": row.get("config_sha256"),
                                "observation_version": row.get("observation_version"),
                            }
                        )

    return _finalize_canonical(
        site=site,
        data_origin=data_origin,
        source_description=f"{archive_kind}_archives:{len(archive_paths)}",
        raw_temp=raw_temp,
        raw_env=raw_env,
        raw_power=raw_power,
        proxy_by_unit=proxy_by_unit,
        actions=actions,
        config_versions=config_versions,
        quality_counts=quality_counts,
    )


# ---------------------------------------------------------------------------
# Adapter 3: Canonical JSON Snapshot (also used by export-history & offline tests)
# ---------------------------------------------------------------------------

def save_canonical_snapshot(history: CanonicalHistory, out_dir: Path) -> Path:
    """Export a CanonicalHistory as clean, auditable JSON/JSONL files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "data_origin": history.data_origin,
        "source_description": history.source_description,
        "site": history.site.summary_dict(),
        "rooms": history.site.raw_rooms,
        "topology": history.site.raw_topology,
        "policy": history.site.policy,
        "config_versions": history.config_versions,
        "quality_counts": history.quality_counts,
        "raw_duplicate_temp_skipped": history.raw_duplicate_temp_skipped,
        "raw_duplicate_power_skipped": history.raw_duplicate_power_skipped,
    }
    (out_dir / "snapshot_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def write_jsonl(name: str, rows: list[dict[str, Any]]) -> None:
        body = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
        (out_dir / name).write_text(body, encoding="utf-8")

    write_jsonl(
        "temp_reports.jsonl",
        [
            {
                "zone": r.zone,
                "role": r.role,
                "source_entity": r.source_entity,
                "report_at": r.report_at,
                "value_c": r.value_c,
                "observed_at": r.observed_at,
                "quality": r.quality,
                "source_version": r.source_version,
                "is_exact_precision": r.is_exact_precision,
            }
            for reps in (*history.zone_temp_reports.values(), *history.env_temp_reports.values())
            for r in reps
        ],
    )
    write_jsonl(
        "power_reports.jsonl",
        [
            {
                "unit": r.unit,
                "source_entity": r.source_entity,
                "observer_epoch": r.observer_epoch,
                "report_at": r.report_at,
                "watts": r.watts,
            }
            for reps in history.power_reports.values()
            for r in reps
        ],
    )
    write_jsonl(
        "proxy_states.jsonl",
        [
            {
                "unit": s.unit,
                "at": s.at,
                "mode": s.mode,
                "setpoint": s.setpoint,
                "target_c": s.target_c,
                "quiet": s.quiet,
                "manual_hold": s.manual_hold,
                "failure_hold": s.failure_hold,
                "isolated": s.isolated,
                "zone_paused": s.zone_paused,
                "source_fresh": s.source_fresh,
                "uncertain": s.uncertain,
            }
            for states in history.proxy_states.values()
            for s in states
        ],
    )
    write_jsonl(
        "actions.jsonl",
        [
            {
                "at": a.at,
                "sent_at": a.sent_at,
                "unit": a.unit,
                "kind": a.kind,
                "before_mode": a.before_mode,
                "before_setpoint": a.before_setpoint,
                "after_mode": a.after_mode,
                "after_setpoint": a.after_setpoint,
                "source": a.source,
                "receipt_status": a.receipt_status,
                "physical_verified": a.physical_verified,
                "purpose": a.purpose,
                "target_at_send": a.target_at_send,
                "quality_flags": list(a.quality_flags),
            }
            for a in history.actions
        ],
    )
    return out_dir


def ingest_canonical_snapshot(snapshot_dir: Path) -> CanonicalHistory:
    """Load a previously exported canonical snapshot directory read-only."""
    from .config import build_site_config

    meta = json.loads((snapshot_dir / "snapshot_meta.json").read_text(encoding="utf-8"))
    site = build_site_config(meta["rooms"], meta["topology"], meta.get("policy"))

    def read_jsonl(name: str) -> list[dict[str, Any]]:
        path = snapshot_dir / name
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    raw_temp: list[TempReport] = []
    raw_env: list[TempReport] = []
    for r in read_jsonl("temp_reports.jsonl"):
        rep = TempReport(
            zone=r["zone"],
            role=r["role"],
            source_entity=r["source_entity"],
            report_at=float(r["report_at"]),
            value_c=float(r["value_c"]),
            observed_at=float(r["observed_at"]),
            quality=str(r.get("quality", "fresh")),
            source_version=r.get("source_version"),
            is_exact_precision=bool(r.get("is_exact_precision", True)),
        )
        if rep.role == "environment":
            raw_env.append(rep)
        else:
            raw_temp.append(rep)

    raw_power = [
        PowerReport(
            unit=r["unit"],
            source_entity=r["source_entity"],
            observer_epoch=str(r["observer_epoch"]),
            report_at=float(r["report_at"]),
            watts=_num(r.get("watts")),
        )
        for r in read_jsonl("power_reports.jsonl")
    ]
    proxy_by_unit: dict[str, list[ProxyStateEvent]] = defaultdict(list)
    for r in read_jsonl("proxy_states.jsonl"):
        proxy_by_unit[r["unit"]].append(
            ProxyStateEvent(
                unit=r["unit"],
                at=float(r["at"]),
                mode=r.get("mode"),
                setpoint=_num(r.get("setpoint")),
                target_c=_num(r.get("target_c")),
                quiet=bool(r.get("quiet", False)),
                manual_hold=bool(r.get("manual_hold", False)),
                failure_hold=bool(r.get("failure_hold", False)),
                isolated=bool(r.get("isolated", False)),
                zone_paused=bool(r.get("zone_paused", False)),
                source_fresh=bool(r.get("source_fresh", True)),
                uncertain=bool(r.get("uncertain", False)),
            )
        )
    actions = [
        ActionEvent(
            at=float(r["at"]),
            sent_at=float(r["sent_at"]),
            unit=r["unit"],
            kind=r["kind"],
            before_mode=r.get("before_mode"),
            before_setpoint=_num(r.get("before_setpoint")),
            after_mode=r.get("after_mode"),
            after_setpoint=_num(r.get("after_setpoint")),
            source=str(r.get("source", "controller")),
            receipt_status=str(r.get("receipt_status", "proxy_echo")),
            physical_verified=bool(r.get("physical_verified", False)) and r.get("receipt_status") != "proxy_echo",
            purpose=r.get("purpose"),
            target_at_send=_num(r.get("target_at_send")),
            quality_flags=tuple(r.get("quality_flags", ())),
        )
        for r in read_jsonl("actions.jsonl")
    ]
    return _finalize_canonical(
        site=site,
        data_origin=str(meta.get("data_origin", DataOrigin.REAL.value)),
        source_description=str(meta.get("source_description", "snapshot")),
        raw_temp=raw_temp,
        raw_env=raw_env,
        raw_power=raw_power,
        proxy_by_unit=proxy_by_unit,
        actions=actions,
        config_versions=list(meta.get("config_versions", ())),
        quality_counts=Counter(meta.get("quality_counts", {})),
    )


def _finalize_canonical(
    *,
    site: SiteConfig,
    data_origin: str,
    source_description: str,
    raw_temp: list[TempReport],
    raw_env: list[TempReport],
    raw_power: list[PowerReport],
    proxy_by_unit: dict[str, list[ProxyStateEvent]],
    actions: list[ActionEvent],
    config_versions: list[dict[str, Any]],
    quality_counts: Counter[str],
) -> CanonicalHistory:
    dedup_temp, skip_t = deduplicate_temp_reports(raw_temp)
    dedup_env, skip_e = deduplicate_temp_reports(raw_env)
    dedup_power, skip_p = deduplicate_power_reports(raw_power)

    zone_temp: dict[str, list[TempReport]] = {z.name: [] for z in site.zones}
    for rep in dedup_temp:
        if rep.zone in zone_temp:
            zone_temp[rep.zone].append(rep)

    env_temp: dict[str, list[TempReport]] = defaultdict(list)
    for rep in dedup_env:
        env_temp[rep.zone].append(rep)

    power_by_unit: dict[str, list[PowerReport]] = {u: [] for u in site.unit_names}
    for rep in dedup_power:
        if rep.unit in power_by_unit:
            power_by_unit[rep.unit].append(rep)

    clean_proxy: dict[str, list[ProxyStateEvent]] = {}
    for u in site.unit_names:
        by_ts: dict[int, ProxyStateEvent] = {}
        for ev in sorted(proxy_by_unit.get(u, ()), key=lambda x: x.at):
            key = int(round(ev.at * 1000))
            if key in by_ts:
                prev = by_ts[key]
                by_ts[key] = ProxyStateEvent(
                    unit=u,
                    at=ev.at,
                    mode=ev.mode if ev.mode is not None else prev.mode,
                    setpoint=ev.setpoint if ev.setpoint is not None else prev.setpoint,
                    target_c=ev.target_c if ev.target_c is not None else prev.target_c,
                    quiet=ev.quiet or prev.quiet,
                    manual_hold=ev.manual_hold or prev.manual_hold,
                    failure_hold=ev.failure_hold or prev.failure_hold,
                    isolated=ev.isolated or prev.isolated,
                    zone_paused=ev.zone_paused or prev.zone_paused,
                    source_fresh=ev.source_fresh and prev.source_fresh,
                    uncertain=ev.uncertain or prev.uncertain,
                )
            else:
                by_ts[key] = ev
        clean_proxy[u] = sorted(by_ts.values(), key=lambda x: x.at)

    # Detect external/manual actions not already present in controller actions.
    existing_manual = [a for a in actions if a.source == "manual_external"]
    ctrl_only = [a for a in actions if a.source != "manual_external"]
    detected_manual = [] if existing_manual else _detect_manual_actions(clean_proxy, ctrl_only)
    all_actions = sorted([*actions, *detected_manual], key=lambda a: (a.at, a.sent_at, a.unit))
    for a in detected_manual:
        for fl in a.quality_flags:
            quality_counts[fl] += 1

    dedup_cfg: dict[tuple[Any, ...], dict[str, Any]] = {}
    for cv in config_versions:
        ckey = (round(float(cv.get("at", 0.0)), 3), cv.get("commit"), cv.get("config_sha256"))
        dedup_cfg[ckey] = cv

    return CanonicalHistory(
        site=site,
        data_origin=data_origin,
        source_description=source_description,
        zone_temp_reports=zone_temp,
        env_temp_reports=dict(env_temp),
        power_reports=power_by_unit,
        proxy_states=clean_proxy,
        actions=all_actions,
        config_versions=sorted(dedup_cfg.values(), key=lambda c: c.get("at", 0)),
        quality_counts=dict(quality_counts),
        raw_duplicate_temp_skipped=skip_t + skip_e,
        raw_duplicate_power_skipped=skip_p,
    )


# ---------------------------------------------------------------------------
# Causal Feature Extraction & Aligned Dataset Builder
# ---------------------------------------------------------------------------

def _mode_onehot(mode: str | None) -> tuple[float, float, float, float]:
    if mode == "off":
        return (1.0, 0.0, 0.0, 0.0)
    if mode == "heat":
        return (0.0, 1.0, 0.0, 0.0)
    if mode == "cool":
        return (0.0, 0.0, 1.0, 0.0)
    return (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class FeatureSpec:
    """Explicit schema for history and future-action tensors (prevents version drift)."""

    feature_version: str
    zone_names: tuple[str, ...]
    unit_names: tuple[str, ...]
    env_sensor_names: tuple[str, ...]
    history_minutes: int
    step_minutes: int
    horizons_minutes: tuple[int, ...]
    hist_feature_names: tuple[str, ...]
    act_feature_names: tuple[str, ...]

    @property
    def n_zones(self) -> int:
        return len(self.zone_names)

    @property
    def n_units(self) -> int:
        return len(self.unit_names)

    @property
    def n_hist_steps(self) -> int:
        return self.history_minutes // self.step_minutes

    @property
    def n_future_steps(self) -> int:
        return max(self.horizons_minutes) // self.step_minutes

    @property
    def horizon_step_indices(self) -> tuple[int, ...]:
        return tuple((h // self.step_minutes) - 1 for h in self.horizons_minutes)

    @property
    def hist_dim(self) -> int:
        return len(self.hist_feature_names)

    @property
    def act_dim(self) -> int:
        return len(self.act_feature_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_version": self.feature_version,
            "zone_names": list(self.zone_names),
            "unit_names": list(self.unit_names),
            "env_sensor_names": list(self.env_sensor_names),
            "history_minutes": self.history_minutes,
            "step_minutes": self.step_minutes,
            "horizons_minutes": list(self.horizons_minutes),
            "hist_dim": self.hist_dim,
            "act_dim": self.act_dim,
            "hist_feature_names": list(self.hist_feature_names),
            "act_feature_names": list(self.act_feature_names),
        }


def build_feature_spec(
    site: SiteConfig,
    *,
    history_minutes: int = DEFAULT_HISTORY_MINUTES,
    step_minutes: int = STEP_MINUTES,
    horizons_minutes: tuple[int, ...] = HORIZONS_MINUTES,
) -> FeatureSpec:
    env_names = tuple(s.name for s in site.active_environment_sensors) or ("outdoor_reference",)
    hist_names: list[str] = []
    for z in site.zone_names:
        hist_names.extend(
            [
                f"zone:{z}:temp_c",
                f"zone:{z}:temp_minus_target_c",
                f"zone:{z}:temp_age_min",
                f"zone:{z}:temp_missing",
                f"zone:{z}:new_temp_report",
                f"zone:{z}:temp_step_delta_c",
                f"zone:{z}:target_c",
                f"zone:{z}:paused",
            ]
        )
    for e in env_names:
        hist_names.extend(
            [
                f"env:{e}:temp_c",
                f"env:{e}:age_min",
                f"env:{e}:missing",
                f"env:{e}:minus_indoor_c",
            ]
        )
    for u in site.unit_names:
        hist_names.extend(
            [
                f"unit:{u}:mode_off",
                f"unit:{u}:mode_heat",
                f"unit:{u}:mode_cool",
                f"unit:{u}:mode_other",
                f"unit:{u}:setpoint_c",
                f"unit:{u}:setpoint_missing",
                f"unit:{u}:sp_minus_home_temp_c",
                f"unit:{u}:mode_duration_min",
                f"unit:{u}:last_power_w",
                f"unit:{u}:bin_wh",
                f"unit:{u}:bin_power_cov",
                f"unit:{u}:power_age_min",
                f"unit:{u}:power_missing",
                f"unit:{u}:power_above_off",
                f"unit:{u}:action_sp_delta",
                f"unit:{u}:action_mode_change",
                f"unit:{u}:action_failed_or_manual",
                f"unit:{u}:hold_or_isolated",
            ]
        )
    hist_names.extend(
        [
            "time:sin_hour",
            "time:cos_hour",
            "quality:any_failed_call",
            "quality:any_manual_op",
            "quality:any_uncertain",
            "quality:any_proxy_unverified",
        ]
    )

    act_names: list[str] = []
    for u in site.unit_names:
        act_names.extend(
            [
                f"fut:{u}:mode_off",
                f"fut:{u}:mode_heat",
                f"fut:{u}:mode_cool",
                f"fut:{u}:mode_other",
                f"fut:{u}:setpoint_c",
                f"fut:{u}:sp_delta_from_t0",
                f"fut:{u}:sp_minus_t0_home_temp",
                f"fut:{u}:is_mode_change",
                f"fut:{u}:is_sp_change",
            ]
        )
    act_names.extend(["fut:step_duration_min", "fut:elapsed_min"])

    return FeatureSpec(
        feature_version=FEATURE_VERSION,
        zone_names=site.zone_names,
        unit_names=site.unit_names,
        env_sensor_names=env_names,
        history_minutes=history_minutes,
        step_minutes=step_minutes,
        horizons_minutes=horizons_minutes,
        hist_feature_names=tuple(hist_names),
        act_feature_names=tuple(act_names),
    )


class _HistoryIndex:
    """Fast strictly-causal indexed view of CanonicalHistory."""

    def __init__(self, history: CanonicalHistory):
        self.history = history
        self.site = history.site
        # Index zone temp reports strictly by causal availability time = max(report_at, observed_at).
        # Wait: report_at <= observed_at in real HA; to guarantee ZERO future leakage, a report is
        # only visible at cutoff t if BOTH report_at <= t AND observed_at <= t!
        self.zone_reports: dict[str, list[TempReport]] = {}
        self.zone_avail_ts: dict[str, list[float]] = {}
        self.zone_report_ts: dict[str, list[float]] = {}
        for z in self.site.zone_names:
            reps = sorted(history.zone_temp_reports.get(z, ()), key=lambda r: (max(r.report_at, r.observed_at), r.report_at))
            self.zone_reports[z] = reps
            self.zone_avail_ts[z] = [max(r.report_at, r.observed_at) for r in reps]
            self.zone_report_ts[z] = [r.report_at for r in reps]

        self.env_reports: dict[str, list[TempReport]] = {}
        self.env_avail_ts: dict[str, list[float]] = {}
        for ename, reps in history.env_temp_reports.items():
            ordered = sorted(reps, key=lambda r: (max(r.report_at, r.observed_at), r.report_at))
            self.env_reports[ename] = ordered
            self.env_avail_ts[ename] = [max(r.report_at, r.observed_at) for r in ordered]

        self.power_reports: dict[str, list[PowerReport]] = {}
        self.power_ts: dict[str, list[float]] = {}
        for u in self.site.unit_names:
            reps = sorted(history.power_reports.get(u, ()), key=lambda r: r.report_at)
            self.power_reports[u] = reps
            self.power_ts[u] = [r.report_at for r in reps]

        self.proxy_states: dict[str, list[ProxyStateEvent]] = {}
        self.proxy_ts: dict[str, list[float]] = {}
        for u in self.site.unit_names:
            evs = sorted(history.proxy_states.get(u, ()), key=lambda r: r.at)
            self.proxy_states[u] = evs
            self.proxy_ts[u] = [r.at for r in evs]

        self.actions = sorted(history.actions, key=lambda a: a.at)
        self.action_ts = [a.at for a in self.actions]

    def latest_zone_temp(self, zone: str, cutoff: float) -> TempReport | None:
        idx = bisect_right(self.zone_avail_ts[zone], cutoff) - 1
        if idx < 0:
            return None
        # Among all causally available reports <= cutoff, return the one with the largest report_at.
        # Scan backward slightly in case out-of-order arrival occurred.
        best = self.zone_reports[zone][idx]
        j = idx - 1
        while j >= 0 and cutoff - self.zone_avail_ts[zone][j] <= 3600:
            cand = self.zone_reports[zone][j]
            if cand.report_at > best.report_at:
                best = cand
            j -= 1
        return best

    def zone_reports_in_window(self, zone: str, start_excl: float, end_incl: float) -> list[TempReport]:
        """Reports with report_at in (start_excl, end_incl] AND causally available by end_incl."""
        hi = bisect_right(self.zone_avail_ts[zone], end_incl)
        out = []
        for rep in self.zone_reports[zone][:hi]:
            if start_excl < rep.report_at <= end_incl:
                out.append(rep)
        return sorted(out, key=lambda r: r.report_at)

    def latest_env_temp(self, ename: str, cutoff: float) -> TempReport | None:
        if ename not in self.env_reports:
            if "outdoor_reference" in self.env_reports:
                ename = "outdoor_reference"
            elif self.env_reports:
                ename = next(iter(self.env_reports))
            else:
                return None
        idx = bisect_right(self.env_avail_ts[ename], cutoff) - 1
        if idx < 0:
            return None
        return self.env_reports[ename][idx]

    def latest_power(self, unit: str, cutoff: float) -> PowerReport | None:
        idx = bisect_right(self.power_ts[unit], cutoff) - 1
        return self.power_reports[unit][idx] if idx >= 0 else None

    def power_slice(self, unit: str, start: float, end: float, *, max_gap_s: float = MAX_POWER_GAP_SECONDS) -> list[PowerReport]:
        lo = bisect_right(self.power_ts[unit], start - max_gap_s - 1e-6)
        # Include one prior point if needed
        lo = max(0, lo - 1)
        hi = bisect_right(self.power_ts[unit], end + max_gap_s)
        return self.power_reports[unit][lo:hi]

    def latest_proxy(self, unit: str, cutoff: float) -> ProxyStateEvent | None:
        idx = bisect_right(self.proxy_ts[unit], cutoff) - 1
        return self.proxy_states[unit][idx] if idx >= 0 else None

    def actions_in_window(self, start_excl: float, end_incl: float, unit: str | None = None) -> list[ActionEvent]:
        lo = bisect_right(self.action_ts, start_excl)
        hi = bisect_right(self.action_ts, end_incl)
        out = self.actions[lo:hi]
        if unit is not None:
            out = [a for a in out if a.unit == unit]
        return out

    def mode_duration_seconds(self, unit: str, cutoff: float) -> float:
        idx = bisect_right(self.proxy_ts[unit], cutoff) - 1
        if idx < 0:
            return 0.0
        cur_mode = self.proxy_states[unit][idx].mode
        since = self.proxy_states[unit][idx].at
        j = idx - 1
        while j >= 0 and self.proxy_states[unit][j].mode == cur_mode and cutoff - self.proxy_states[unit][j].at <= 7200:
            since = self.proxy_states[unit][j].at
            j -= 1
        return max(0.0, cutoff - since)

    def zone_target_at(self, zone: str, cutoff: float) -> float | None:
        zmap = self.site.zone_by_name(zone)
        if zmap.target_celsius is not None:
            return zmap.target_celsius
        if zmap.target_actuator:
            ev = self.latest_proxy(zmap.target_actuator, cutoff)
            if ev is not None and ev.target_c is not None:
                return ev.target_c
        # Fallback: check any unit in this home zone
        for act in self.site.actuators:
            if act.home_zone == zone:
                ev = self.latest_proxy(act.name, cutoff)
                if ev is not None and ev.target_c is not None:
                    return ev.target_c
        return None

    def settle_future_temp(
        self,
        zone: str,
        t0: float,
        t0_report_at: float | None,
        horizon_s: float,
        settle_window_s: float = DEFAULT_SETTLE_WINDOW_SECONDS,
    ) -> TempReport | None:
        """Find a strictly newer real temperature report near t0 + horizon_s.

        NEVER counts a cached reading with report_at <= t0_report_at!
        """
        target_ts = t0 + horizon_s
        min_rep = max(target_ts - settle_window_s, (t0_report_at + 1e-6) if t0_report_at is not None else t0 + 1e-6)
        max_rep = target_ts + settle_window_s
        candidates = [
            rep
            for rep in self.zone_reports[zone]
            if min_rep <= rep.report_at <= max_rep and (t0_report_at is None or rep.report_at > t0_report_at + 1e-6)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda r: abs(r.report_at - target_ts))

    def existing_trend_15m_delta(self, zone: str, cutoff: float) -> float | None:
        """Compute existing repo's 15-minute trend prediction strictly causally from unique reports <= cutoff."""
        hi = bisect_right(self.zone_avail_ts[zone], cutoff)
        unique: dict[float, float] = {}
        for rep in self.zone_reports[zone][:hi]:
            if cutoff - 2700.0 <= rep.report_at <= cutoff:
                unique[rep.report_at] = rep.value_c
        points = sorted(unique.items())
        if len(points) < 3 or points[-1][0] - points[0][0] < 600.0 or cutoff - points[-1][0] > 1200.0:
            return None
        # Reuse exact math from home-assistant/shadow/model.py:predict()
        edge_count = min(3, len(points) // 2)
        initial = float(np.median([v for _, v in points[:edge_count]]))
        final = float(np.median([v for _, v in points[-edge_count:]]))
        if abs(final - initial) < 0.2 - 1e-6:
            return 0.0
        rates = [
            (v2 - v1) * 900.0 / (t2 - t1)
            for i, (t1, v1) in enumerate(points)
            for t2, v2 in points[i + 1 :]
            if t2 - t1 >= 120.0
        ]
        if not rates:
            return None
        return round(shadow_model.clip(float(np.median(rates)), -2.0, 2.0), 3)


def build_action_sequence_tensor(
    spec: FeatureSpec,
    site: SiteConfig,
    t0_unit_modes: dict[str, str | None],
    t0_unit_setpoints: dict[str, float | None],
    t0_zone_temps: dict[str, float | None],
    step_plans: list[dict[str, tuple[str | None, float | None]]],
) -> np.ndarray:
    """Encode a future candidate action sequence [n_future_steps, act_dim] using ONLY t0 state + planned actions."""
    n_steps = spec.n_future_steps
    if len(step_plans) != n_steps:
        raise ValueError(f"Expected {n_steps} future action steps, got {len(step_plans)}")
    arr = np.zeros((n_steps, spec.act_dim), dtype=np.float32)
    prev_modes = dict(t0_unit_modes)
    prev_sps = dict(t0_unit_setpoints)
    for s_idx, plan in enumerate(step_plans):
        col = 0
        for uname in spec.unit_names:
            act_map = site.actuator_by_name(uname)
            mode, sp = plan[uname]
            t0_sp = t0_unit_setpoints.get(uname)
            t0_temp = t0_zone_temps.get(act_map.home_zone)
            oh = _mode_onehot(mode)
            arr[s_idx, col : col + 4] = oh
            col += 4
            sp_val = float(sp) if sp is not None else (float(t0_sp) if t0_sp is not None else 24.0)
            arr[s_idx, col] = sp_val
            col += 1
            sp_delta = (sp_val - float(t0_sp)) if (sp is not None and t0_sp is not None) else 0.0
            arr[s_idx, col] = sp_delta
            col += 1
            arr[s_idx, col] = (sp_val - float(t0_temp)) if (t0_temp is not None and mode in {"heat", "cool"}) else 0.0
            col += 1
            arr[s_idx, col] = 1.0 if mode != prev_modes.get(uname) else 0.0
            col += 1
            prev_sp = prev_sps.get(uname)
            arr[s_idx, col] = 1.0 if (sp is not None and prev_sp is not None and abs(sp - prev_sp) > 1e-4) else 0.0
            col += 1
            prev_modes[uname] = mode
            prev_sps[uname] = sp
        arr[s_idx, col] = float(spec.step_minutes)
        arr[s_idx, col + 1] = float((s_idx + 1) * spec.step_minutes)
    return arr


def build_hold_plan(
    spec: FeatureSpec,
    t0_unit_modes: dict[str, str | None],
    t0_unit_setpoints: dict[str, float | None],
) -> list[dict[str, tuple[str | None, float | None]]]:
    step = {u: (t0_unit_modes.get(u), t0_unit_setpoints.get(u)) for u in spec.unit_names}
    return [dict(step) for _ in range(spec.n_future_steps)]


def build_step_action_plan(
    spec: FeatureSpec,
    t0_unit_modes: dict[str, str | None],
    t0_unit_setpoints: dict[str, float | None],
    *,
    unit: str | None = None,
    new_mode: str | None = None,
    new_setpoint: float | None = None,
) -> list[dict[str, tuple[str | None, float | None]]]:
    """Constant candidate action starting at t0 and held over the 60m horizon."""
    step = {u: (t0_unit_modes.get(u), t0_unit_setpoints.get(u)) for u in spec.unit_names}
    if unit is not None:
        m0, sp0 = step[unit]
        step[unit] = (new_mode if new_mode is not None else m0, new_setpoint if new_setpoint is not None else sp0)
    return [dict(step) for _ in range(spec.n_future_steps)]


def extract_history_window(
    index: _HistoryIndex,
    spec: FeatureSpec,
    cutoff_ts: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract strictly causal [n_hist_steps, hist_dim] features and t0 metadata at cutoff_ts.

    ZERO future values (> cutoff_ts) are accessed.
    """
    site = index.site
    step_s = float(spec.step_minutes * 60)
    n_steps = spec.n_hist_steps
    X = np.full((n_steps, spec.hist_dim), np.nan, dtype=np.float32)

    last_seen_zone_rep: dict[str, TempReport | None] = {z: None for z in spec.zone_names}
    step_any_failed = False
    step_any_manual = False
    step_any_uncertain = False
    step_any_unverified = False

    # Seed previous report before window start so first bin can compute step delta causally.
    window_start = cutoff_ts - n_steps * step_s
    for z in spec.zone_names:
        last_seen_zone_rep[z] = index.latest_zone_temp(z, window_start)

    for k in range(n_steps):
        bin_end = cutoff_ts - (n_steps - 1 - k) * step_s
        bin_start = bin_end - step_s
        col = 0

        zone_temps_now: dict[str, float | None] = {}
        for z in spec.zone_names:
            zmap = site.zone_by_name(z)
            latest = index.latest_zone_temp(z, bin_end)
            prev_rep = last_seen_zone_rep[z]
            new_rep = (
                latest is not None
                and (prev_rep is None or latest.report_at > prev_rep.report_at + 1e-6)
                and bin_start < latest.report_at <= bin_end
            )
            age_s = (bin_end - latest.report_at) if latest is not None else float("inf")
            missing = latest is None or age_s < -60.0 or age_s > zmap.max_age_seconds
            temp_val = latest.value_c if (latest is not None and not missing) else float("nan")
            zone_temps_now[z] = None if math.isnan(temp_val) else temp_val
            target = index.zone_target_at(z, bin_end)
            delta_t = (temp_val - target) if (not math.isnan(temp_val) and target is not None) else float("nan")
            step_delta = (
                (latest.value_c - prev_rep.value_c)
                if (new_rep and latest is not None and prev_rep is not None)
                else 0.0
            )
            paused = any(
                (ev := index.latest_proxy(a.name, bin_end)) is not None and ev.zone_paused
                for a in site.actuators
                if a.home_zone == z
            )
            X[k, col] = temp_val
            X[k, col + 1] = delta_t
            X[k, col + 2] = min(age_s / 60.0, 60.0) if math.isfinite(age_s) else 60.0
            X[k, col + 3] = 1.0 if missing else 0.0
            X[k, col + 4] = 1.0 if new_rep else 0.0
            X[k, col + 5] = step_delta
            X[k, col + 6] = target if target is not None else float("nan")
            X[k, col + 7] = 1.0 if paused else 0.0
            col += 8
            if latest is not None and (prev_rep is None or latest.report_at > prev_rep.report_at):
                last_seen_zone_rep[z] = latest

        valid_indoor = [v for v in zone_temps_now.values() if v is not None]
        mean_indoor = float(np.mean(valid_indoor)) if valid_indoor else None

        for ename in spec.env_sensor_names:
            eref = index.latest_env_temp(ename, bin_end)
            age_s = (bin_end - eref.report_at) if eref is not None else float("inf")
            missing = eref is None or age_s < -60.0 or age_s > 7200.0
            eval_c = eref.value_c if (eref is not None and not missing) else float("nan")
            X[k, col] = eval_c
            X[k, col + 1] = min(age_s / 60.0, 180.0) if math.isfinite(age_s) else 180.0
            X[k, col + 2] = 1.0 if missing else 0.0
            X[k, col + 3] = (eval_c - mean_indoor) if (not math.isnan(eval_c) and mean_indoor is not None) else float("nan")
            col += 4

        bin_failed = False
        bin_manual = False
        bin_uncertain = False
        bin_unverified = False
        bin_actions = index.actions_in_window(bin_start, bin_end)
        for u in spec.unit_names:
            act_map = site.actuator_by_name(u)
            proxy = index.latest_proxy(u, bin_end)
            mode = proxy.mode if proxy is not None else None
            sp = proxy.setpoint if proxy is not None else None
            sp_missing = sp is None or mode not in {"heat", "cool"}
            home_t = zone_temps_now.get(act_map.home_zone)
            oh = _mode_onehot(mode)
            mode_dur_min = min(index.mode_duration_seconds(u, bin_end) / 60.0, 180.0)

            latest_p = index.latest_power(u, bin_end)
            p_age_s = (bin_end - latest_p.report_at) if latest_p is not None else float("inf")
            p_missing = latest_p is None or latest_p.watts is None or p_age_s > MAX_POWER_GAP_SECONDS or p_age_s < -60.0
            p_w = latest_p.watts if (latest_p is not None and not p_missing) else float("nan")
            # Causal power integration strictly inside (bin_start, bin_end] using reports <= bin_end.
            p_slice = [r for r in index.power_slice(u, bin_start, bin_end) if r.report_at <= bin_end]
            p_int = integrate_power_interval(p_slice, bin_start, bin_end)
            off_ceil = act_map.verified_off_ceiling_w if act_map.verified_off_ceiling_w is not None else 15.0
            above_off = 1.0 if (not math.isnan(p_w) and p_w > off_ceil) else 0.0

            u_acts = [a for a in bin_actions if a.unit == u]
            act_sp_delta = 0.0
            act_mode_chg = 0.0
            act_bad = 0.0
            for a in u_acts:
                if a.kind == "mode" or (a.before_mode != a.after_mode and a.after_mode is not None):
                    act_mode_chg = 1.0
                if a.after_setpoint is not None and a.before_setpoint is not None:
                    act_sp_delta += a.after_setpoint - a.before_setpoint
                if "failed_call" in a.quality_flags:
                    bin_failed = True
                    act_bad = 1.0
                if "manual_operation" in a.quality_flags:
                    bin_manual = True
                    act_bad = 1.0
                if "proxy_only_unverified" in a.quality_flags:
                    bin_unverified = True

            hold_or_iso = bool(proxy and (proxy.manual_hold or proxy.failure_hold or proxy.isolated))
            if proxy and proxy.manual_hold:
                bin_manual = True
            if proxy and (proxy.uncertain or (mode == "off" and not math.isnan(p_w) and p_w > max(80.0, off_ceil * 5))):
                bin_uncertain = True

            X[k, col : col + 4] = oh
            X[k, col + 4] = sp if sp is not None else float("nan")
            X[k, col + 5] = 1.0 if sp_missing else 0.0
            X[k, col + 6] = (sp - home_t) if (sp is not None and home_t is not None and mode in {"heat", "cool"}) else 0.0
            X[k, col + 7] = mode_dur_min
            X[k, col + 8] = p_w
            X[k, col + 9] = p_int.wh if p_int.wh is not None else float("nan")
            X[k, col + 10] = p_int.coverage_ratio
            X[k, col + 11] = min(p_age_s / 60.0, 30.0) if math.isfinite(p_age_s) else 30.0
            X[k, col + 12] = 1.0 if p_missing else 0.0
            X[k, col + 13] = above_off
            X[k, col + 14] = act_sp_delta
            X[k, col + 15] = act_mode_chg
            X[k, col + 16] = act_bad
            X[k, col + 17] = 1.0 if hold_or_iso else 0.0
            col += 18

        dt_utc = datetime.fromtimestamp(bin_end, timezone.utc)
        hour_f = dt_utc.hour + dt_utc.minute / 60.0
        X[k, col] = math.sin(2.0 * math.pi * hour_f / 24.0)
        X[k, col + 1] = math.cos(2.0 * math.pi * hour_f / 24.0)
        X[k, col + 2] = 1.0 if bin_failed else 0.0
        X[k, col + 3] = 1.0 if bin_manual else 0.0
        X[k, col + 4] = 1.0 if bin_uncertain else 0.0
        X[k, col + 5] = 1.0 if bin_unverified else 0.0

        step_any_failed |= bin_failed
        step_any_manual |= bin_manual
        step_any_uncertain |= bin_uncertain
        step_any_unverified |= bin_unverified

    # Snapshot exact t0 metadata (strictly <= cutoff_ts)
    t0_zone_temps: dict[str, float | None] = {}
    t0_zone_targets: dict[str, float | None] = {}
    t0_zone_report_at: dict[str, float | None] = {}
    t0_zone_age_s: dict[str, float | None] = {}
    t0_zone_trend_15m: dict[str, float | None] = {}
    t0_zone_paused: dict[str, bool] = {}
    for z in spec.zone_names:
        zmap = site.zone_by_name(z)
        rep = index.latest_zone_temp(z, cutoff_ts)
        age = (cutoff_ts - rep.report_at) if rep is not None else None
        fresh = rep is not None and age is not None and -60.0 <= age <= zmap.max_age_seconds
        t0_zone_temps[z] = rep.value_c if fresh else None
        t0_zone_report_at[z] = rep.report_at if rep is not None else None
        t0_zone_age_s[z] = age
        t0_zone_targets[z] = index.zone_target_at(z, cutoff_ts)
        t0_zone_trend_15m[z] = index.existing_trend_15m_delta(z, cutoff_ts)
        t0_zone_paused[z] = any(
            (ev := index.latest_proxy(a.name, cutoff_ts)) is not None and ev.zone_paused
            for a in site.actuators
            if a.home_zone == z
        )

    t0_env_temps: dict[str, float | None] = {}
    t0_env_report_at: dict[str, float | None] = {}
    for ename in spec.env_sensor_names:
        eref = index.latest_env_temp(ename, cutoff_ts)
        fresh = eref is not None and -60.0 <= (cutoff_ts - eref.report_at) <= 7200.0
        t0_env_temps[ename] = eref.value_c if fresh else None
        t0_env_report_at[ename] = eref.report_at if eref is not None else None

    t0_unit_modes: dict[str, str | None] = {}
    t0_unit_setpoints: dict[str, float | None] = {}
    t0_unit_watts: dict[str, float | None] = {}
    t0_unit_power_report_at: dict[str, float | None] = {}
    t0_unit_holds: dict[str, dict[str, bool]] = {}
    t0_unit_last_action_at: dict[str, float | None] = {}
    recent_actions = index.actions_in_window(cutoff_ts - 900.0, cutoff_ts)
    for u in spec.unit_names:
        ev = index.latest_proxy(u, cutoff_ts)
        t0_unit_modes[u] = ev.mode if ev else None
        t0_unit_setpoints[u] = ev.setpoint if ev else None
        t0_unit_holds[u] = {
            "manual_hold": bool(ev.manual_hold) if ev else False,
            "failure_hold": bool(ev.failure_hold) if ev else False,
            "isolated": bool(ev.isolated) if ev else False,
            "quiet": bool(ev.quiet) if ev else False,
        }
        pr = index.latest_power(u, cutoff_ts)
        pfresh = pr is not None and pr.watts is not None and -60.0 <= (cutoff_ts - pr.report_at) <= MAX_POWER_GAP_SECONDS
        t0_unit_watts[u] = pr.watts if pfresh else None
        t0_unit_power_report_at[u] = pr.report_at if pr is not None else None
        u_hist_acts = [a for a in index.actions if a.unit == u and a.at <= cutoff_ts]
        t0_unit_last_action_at[u] = u_hist_acts[-1].at if u_hist_acts else None

    meta = {
        "cutoff_ts": cutoff_ts,
        "zone_temps": t0_zone_temps,
        "zone_targets": t0_zone_targets,
        "zone_report_at": t0_zone_report_at,
        "zone_age_s": t0_zone_age_s,
        "zone_trend_15m": t0_zone_trend_15m,
        "zone_paused": t0_zone_paused,
        "env_temps": t0_env_temps,
        "env_report_at": t0_env_report_at,
        "unit_modes": t0_unit_modes,
        "unit_setpoints": t0_unit_setpoints,
        "unit_watts": t0_unit_watts,
        "unit_power_report_at": t0_unit_power_report_at,
        "unit_holds": t0_unit_holds,
        "unit_last_action_at": t0_unit_last_action_at,
        "post_action_15m": len(recent_actions) > 0,
        "quality_flags": {
            "any_failed_call": step_any_failed,
            "any_manual_op": step_any_manual,
            "any_uncertain": step_any_uncertain,
            "any_proxy_unverified": step_any_unverified,
        },
    }
    return X, meta


@dataclass
class BuiltDataset:
    """Self-contained aligned dataset with manifest and zero-leakage arrays."""

    site: SiteConfig
    spec: FeatureSpec
    manifest: dict[str, Any]
    cutoff_ts: np.ndarray  # [N] float64
    X_hist: np.ndarray  # [N, H, D_hist] float32 (NaNs preserved for train-only imputer)
    X_act_actual: np.ndarray  # [N, F, D_act] float32 (actual executed future actions for replay/training)
    X_act_hold: np.ndarray  # [N, F, D_act] float32 (known-at-t0 hold actions for online-style eval)
    y_temp_delta: np.ndarray  # [N, Z, K] float32 (NaN where invalid)
    y_temp_valid: np.ndarray  # [N, Z, K] bool
    y_unit_wh: np.ndarray  # [N, U, K] float32 (NaN where invalid)
    y_unit_cov: np.ndarray  # [N, U, K] float32 (0..1 exact metering coverage)
    y_unit_valid: np.ndarray  # [N, U, K] bool
    y_total_wh: np.ndarray  # [N, K] float32 (NaN where any unit uncovered)
    y_total_cov: np.ndarray  # [N, K] float32 (min unit coverage across the 3 units)
    y_total_valid: np.ndarray  # [N, K] bool
    intervened: np.ndarray  # [N, K] bool (actual actions or targets changed mid-horizon)
    metadata: list[dict[str, Any]]  # [N] per-sample t0 metadata & label provenance


def build_dataset(
    history: CanonicalHistory,
    *,
    history_minutes: int = DEFAULT_HISTORY_MINUTES,
    step_minutes: int = STEP_MINUTES,
    horizons_minutes: tuple[int, ...] = HORIZONS_MINUTES,
    settle_window_s: float = DEFAULT_SETTLE_WINDOW_SECONDS,
    min_wh_coverage: float = DEFAULT_MIN_WH_COVERAGE,
) -> BuiltDataset:
    """Build causally aligned training/evaluation dataset and reproducible manifest."""
    site = history.site
    spec = build_feature_spec(
        site,
        history_minutes=history_minutes,
        step_minutes=step_minutes,
        horizons_minutes=horizons_minutes,
    )
    index = _HistoryIndex(history)
    step_s = float(step_minutes * 60)
    max_horizon_s = float(max(horizons_minutes) * 60)

    all_ts: list[float] = []
    for reps in history.zone_temp_reports.values():
        all_ts.extend(r.report_at for r in reps)
    for reps in history.power_reports.values():
        all_ts.extend(r.report_at for r in reps)
    for evs in history.proxy_states.values():
        all_ts.extend(e.at for e in evs)

    if not all_ts:
        raise ValueError("Canonical history contains no timestamped observations")

    t_min = min(all_ts)
    t_max = max(all_ts)
    # Start grid after at least 1 hour (or configured history_minutes if available; allow partial initial prefix >= 60m)
    first_grid = math.ceil((t_min + min(history_minutes, 60) * 60.0) / step_s) * step_s
    last_grid = math.floor((t_max - min(horizons_minutes) * 60.0) / step_s) * step_s

    grid_points: list[float] = []
    t = first_grid
    while t <= last_grid + 1e-6:
        grid_points.append(t)
        t += step_s

    n_z = spec.n_zones
    n_u = spec.n_units
    n_k = len(horizons_minutes)

    cutoffs: list[float] = []
    X_hist_list: list[np.ndarray] = []
    X_act_actual_list: list[np.ndarray] = []
    X_act_hold_list: list[np.ndarray] = []
    y_td_list: list[np.ndarray] = []
    y_tv_list: list[np.ndarray] = []
    y_uwh_list: list[np.ndarray] = []
    y_ucov_list: list[np.ndarray] = []
    y_uval_list: list[np.ndarray] = []
    y_twh_list: list[np.ndarray] = []
    y_tcov_list: list[np.ndarray] = []
    y_tval_list: list[np.ndarray] = []
    interv_list: list[np.ndarray] = []
    meta_list: list[dict[str, Any]] = []

    for t0 in grid_points:
        xh, meta = extract_history_window(index, spec, t0)
        # Skip grid points where NO independent zone has a fresh t0 reading AND no proxy state exists.
        if all(v is None for v in meta["zone_temps"].values()) and all(v is None for v in meta["unit_modes"].values()):
            continue

        # Build known-at-t0 hold plan and actual-executed future action sequence (for replay/training only).
        hold_plan = build_hold_plan(spec, meta["unit_modes"], meta["unit_setpoints"])
        actual_plan: list[dict[str, tuple[str | None, float | None]]] = []
        for s_idx in range(spec.n_future_steps):
            s_end = t0 + (s_idx + 1) * step_s
            step_state: dict[str, tuple[str | None, float | None]] = {}
            for u in spec.unit_names:
                ev = index.latest_proxy(u, s_end)
                step_state[u] = (ev.mode if ev else meta["unit_modes"][u], ev.setpoint if ev else meta["unit_setpoints"][u])
            actual_plan.append(step_state)

        x_act_actual = build_action_sequence_tensor(
            spec, site, meta["unit_modes"], meta["unit_setpoints"], meta["zone_temps"], actual_plan
        )
        x_act_hold = build_action_sequence_tensor(
            spec, site, meta["unit_modes"], meta["unit_setpoints"], meta["zone_temps"], hold_plan
        )

        # Build future labels & intervention flags
        y_td = np.full((n_z, n_k), np.nan, dtype=np.float32)
        y_tv = np.zeros((n_z, n_k), dtype=bool)
        y_uwh = np.full((n_u, n_k), np.nan, dtype=np.float32)
        y_ucov = np.zeros((n_u, n_k), dtype=np.float32)
        y_uval = np.zeros((n_u, n_k), dtype=bool)
        y_twh = np.full((n_k,), np.nan, dtype=np.float32)
        y_tcov = np.zeros((n_k,), dtype=np.float32)
        y_tval = np.zeros((n_k,), dtype=bool)
        interv = np.zeros((n_k,), dtype=bool)
        label_report_times: dict[str, dict[str, float | None]] = {z: {} for z in spec.zone_names}

        for h_idx, h_min in enumerate(horizons_minutes):
            h_s = float(h_min * 60)
            # Check mid-horizon action or target changes in (t0, t0 + h_s]
            mid_acts = index.actions_in_window(t0, t0 + h_s)
            action_changed = len(mid_acts) > 0
            if not action_changed:
                # Also verify proxy mode/setpoint/target did not drift relative to t0
                for u in spec.unit_names:
                    for ev in index.proxy_states[u]:
                        if t0 < ev.at <= t0 + h_s:
                            if ev.mode != meta["unit_modes"][u] and ev.mode not in {None, "unavailable", "unknown"}:
                                action_changed = True
                                break
                            if (
                                ev.mode in {"heat", "cool"}
                                and ev.setpoint is not None
                                and meta["unit_setpoints"][u] is not None
                                and abs(ev.setpoint - meta["unit_setpoints"][u]) > 1e-4
                            ):
                                action_changed = True
                                break
                    if action_changed:
                        break
            target_changed = False
            for z in spec.zone_names:
                tgt0 = meta["zone_targets"][z]
                tgt_h = index.zone_target_at(z, t0 + h_s)
                if tgt0 is not None and tgt_h is not None and abs(tgt_h - tgt0) > 1e-4:
                    target_changed = True
                    break
            interv[h_idx] = action_changed or target_changed

            # Temperature labels (2 independent zones ONLY)
            for z_idx, z in enumerate(spec.zone_names):
                t0_temp = meta["zone_temps"][z]
                t0_rep = meta["zone_report_at"][z]
                fut = index.settle_future_temp(z, t0, t0_rep, h_s, settle_window_s=settle_window_s)
                label_report_times[z][str(h_min)] = fut.report_at if fut is not None else None
                if t0_temp is not None and fut is not None:
                    y_td[z_idx, h_idx] = float(fut.value_c - t0_temp)
                    y_tv[z_idx, h_idx] = True

            # Power labels (3 units + total): require complete coverage (has_gap == False)
            # so missing intervals are NEVER silently treated as 0 Wh.
            effective_min_cov = max(float(min_wh_coverage), 1.0 - 1e-6)
            unit_whs_valid = True
            total_wh_acc = 0.0
            min_cov = 1.0
            for u_idx, u in enumerate(spec.unit_names):
                pslice = index.power_slice(u, t0, t0 + h_s)
                pint = integrate_power_interval(pslice, t0, t0 + h_s)
                y_ucov[u_idx, h_idx] = float(pint.coverage_ratio)
                min_cov = min(min_cov, pint.coverage_ratio)
                full_wh = pint.full_interval_wh
                if full_wh is not None and pint.coverage_ratio >= effective_min_cov:
                    y_uwh[u_idx, h_idx] = float(full_wh)
                    y_uval[u_idx, h_idx] = True
                    total_wh_acc += float(full_wh)
                else:
                    unit_whs_valid = False
            y_tcov[h_idx] = float(min_cov)
            if unit_whs_valid:
                y_twh[h_idx] = float(total_wh_acc)
                y_tval[h_idx] = True

        # Keep sample if at least one horizon has a valid temp or energy label
        if not (y_tv.any() or y_tval.any()):
            continue

        meta["label_report_times"] = label_report_times
        meta["max_label_ts"] = t0 + max_horizon_s
        meta["intervened_by_horizon"] = {str(h): bool(interv[i]) for i, h in enumerate(horizons_minutes)}

        cutoffs.append(t0)
        X_hist_list.append(xh)
        X_act_actual_list.append(x_act_actual)
        X_act_hold_list.append(x_act_hold)
        y_td_list.append(y_td)
        y_tv_list.append(y_tv)
        y_uwh_list.append(y_uwh)
        y_ucov_list.append(y_ucov)
        y_uval_list.append(y_uval)
        y_twh_list.append(y_twh)
        y_tcov_list.append(y_tcov)
        y_tval_list.append(y_tval)
        interv_list.append(interv)
        meta_list.append(meta)

    n = len(cutoffs)
    if n == 0:
        cutoff_arr = np.zeros((0,), dtype=np.float64)
        X_hist = np.zeros((0, spec.n_hist_steps, spec.hist_dim), dtype=np.float32)
        X_act_actual = np.zeros((0, spec.n_future_steps, spec.act_dim), dtype=np.float32)
        X_act_hold = np.zeros((0, spec.n_future_steps, spec.act_dim), dtype=np.float32)
        y_td_arr = np.zeros((0, n_z, n_k), dtype=np.float32)
        y_tv_arr = np.zeros((0, n_z, n_k), dtype=bool)
        y_uwh_arr = np.zeros((0, n_u, n_k), dtype=np.float32)
        y_ucov_arr = np.zeros((0, n_u, n_k), dtype=np.float32)
        y_uval_arr = np.zeros((0, n_u, n_k), dtype=bool)
        y_twh_arr = np.zeros((0, n_k), dtype=np.float32)
        y_tcov_arr = np.zeros((0, n_k), dtype=np.float32)
        y_tval_arr = np.zeros((0, n_k), dtype=bool)
        interv_arr = np.zeros((0, n_k), dtype=bool)
    else:
        cutoff_arr = np.asarray(cutoffs, dtype=np.float64)
        X_hist = np.stack(X_hist_list, axis=0)
        X_act_actual = np.stack(X_act_actual_list, axis=0)
        X_act_hold = np.stack(X_act_hold_list, axis=0)
        y_td_arr = np.stack(y_td_list, axis=0)
        y_tv_arr = np.stack(y_tv_list, axis=0)
        y_uwh_arr = np.stack(y_uwh_list, axis=0)
        y_ucov_arr = np.stack(y_ucov_list, axis=0)
        y_uval_arr = np.stack(y_uval_list, axis=0)
        y_twh_arr = np.stack(y_twh_list, axis=0)
        y_tcov_arr = np.stack(y_tcov_list, axis=0)
        y_tval_arr = np.stack(y_tval_list, axis=0)
        interv_arr = np.stack(interv_list, axis=0)

    # Action coverage summary
    action_type_counts = Counter()
    for a in history.actions:
        if a.kind == "mode":
            action_type_counts[f"mode_{a.after_mode}"] += 1
        elif a.kind == "temperature" and a.before_setpoint is not None and a.after_setpoint is not None:
            direction = "up" if a.after_setpoint > a.before_setpoint else "down"
            action_type_counts[f"setpoint_{direction}_{a.after_mode}"] += 1
        else:
            action_type_counts[a.kind] += 1

    # Power coverage over full observed span
    power_cov_summary = {}
    for u in spec.unit_names:
        pint = integrate_power_interval(history.power_reports.get(u, []), t_min, t_max)
        power_cov_summary[u] = {
            "unique_reports": len(history.power_reports.get(u, [])),
            "coverage_ratio": round(pint.coverage_ratio, 4),
            "covered_seconds": round(pint.covered_seconds, 1),
            "gap_seconds": round(pint.gap_seconds, 1),
            "covered_wh": round(pint.wh, 3) if pint.wh is not None else None,
        }

    # Mode distribution across aligned samples
    mode_counts = Counter()
    for m in meta_list:
        modes = set(m["unit_modes"].values())
        if "heat" in modes:
            mode_counts["any_heat"] += 1
        if "cool" in modes:
            mode_counts["any_cool"] += 1
        if all(v in {"off", None} for v in m["unit_modes"].values()):
            mode_counts["all_off"] += 1

    data_digest = sha256(
        cutoff_arr.tobytes() + np.nan_to_num(X_hist, nan=-9999.0).tobytes() + y_tv_arr.tobytes()
    ).hexdigest()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "data_origin": history.data_origin,
        "source_description": history.source_description,
        "site": site.summary_dict(),
        "feature_spec": spec.to_dict(),
        "time_range_utc": {"from": _iso_utc(t_min), "to": _iso_utc(t_max)},
        "time_range_ts": {"from": t_min, "to": t_max},
        "observed_span_hours": round((t_max - t_min) / 3600.0, 3),
        "n_samples": n,
        "n_zones": n_z,
        "n_units": n_u,
        "unique_temp_reports_by_zone": {z: len(history.zone_temp_reports.get(z, [])) for z in spec.zone_names},
        "unique_env_reports": {e: len(reps) for e, reps in history.env_temp_reports.items()},
        "duplicate_temp_skipped": history.raw_duplicate_temp_skipped,
        "duplicate_power_skipped": history.raw_duplicate_power_skipped,
        "power_coverage_by_unit": power_cov_summary,
        "valid_temp_labels_by_horizon": {
            str(h): int(y_tv_arr[:, :, i].sum()) if n else 0 for i, h in enumerate(horizons_minutes)
        },
        "valid_total_wh_labels_by_horizon": {
            str(h): int(y_tval_arr[:, i].sum()) if n else 0 for i, h in enumerate(horizons_minutes)
        },
        "mean_joint_wh_coverage_by_horizon": {
            str(h): round(float(y_tcov_arr[:, i].mean()), 4) if n else 0.0 for i, h in enumerate(horizons_minutes)
        },
        "intervened_samples_by_horizon": {
            str(h): int(interv_arr[:, i].sum()) if n else 0 for i, h in enumerate(horizons_minutes)
        },
        "action_events_total": len(history.actions),
        "action_type_counts": dict(action_type_counts),
        "mode_sample_counts": dict(mode_counts),
        "quality_counts": history.quality_counts,
        "config_versions": history.config_versions,
        "min_wh_coverage": max(float(min_wh_coverage), 1.0),
        "require_complete_wh_coverage": True,
        "settle_window_seconds": settle_window_s,
        "data_sha256": data_digest,
    }
    return BuiltDataset(
        site=site,
        spec=spec,
        manifest=manifest,
        cutoff_ts=cutoff_arr,
        X_hist=X_hist,
        X_act_actual=X_act_actual,
        X_act_hold=X_act_hold,
        y_temp_delta=y_td_arr,
        y_temp_valid=y_tv_arr,
        y_unit_wh=y_uwh_arr,
        y_unit_cov=y_ucov_arr,
        y_unit_valid=y_uval_arr,
        y_total_wh=y_twh_arr,
        y_total_cov=y_tcov_arr,
        y_total_valid=y_tval_arr,
        intervened=interv_arr,
        metadata=meta_list,
    )


def save_dataset(dataset: BuiltDataset, out_dir: Path) -> Path:
    """Persist BuiltDataset arrays + manifest + site definition to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "dataset.npz",
        cutoff_ts=dataset.cutoff_ts,
        X_hist=dataset.X_hist,
        X_act_actual=dataset.X_act_actual,
        X_act_hold=dataset.X_act_hold,
        y_temp_delta=dataset.y_temp_delta,
        y_temp_valid=dataset.y_temp_valid,
        y_unit_wh=dataset.y_unit_wh,
        y_unit_cov=dataset.y_unit_cov,
        y_unit_valid=dataset.y_unit_valid,
        y_total_wh=dataset.y_total_wh,
        y_total_cov=dataset.y_total_cov,
        y_total_valid=dataset.y_total_valid,
        intervened=dataset.intervened,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(dataset.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "site_definition.json").write_text(
        json.dumps(
            {
                "rooms": dataset.site.raw_rooms,
                "topology": dataset.site.raw_topology,
                "policy": dataset.site.policy,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for row in dataset.metadata:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return out_dir


def load_dataset(dataset_dir: Path) -> BuiltDataset:
    """Load a persisted BuiltDataset from disk."""
    from .config import build_site_config

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    site_raw = json.loads((dataset_dir / "site_definition.json").read_text(encoding="utf-8"))
    site = build_site_config(site_raw["rooms"], site_raw["topology"], site_raw.get("policy"))
    fspec_raw = manifest["feature_spec"]
    spec = build_feature_spec(
        site,
        history_minutes=int(fspec_raw["history_minutes"]),
        step_minutes=int(fspec_raw["step_minutes"]),
        horizons_minutes=tuple(int(h) for h in fspec_raw["horizons_minutes"]),
    )
    meta_lines = [
        json.loads(line)
        for line in (dataset_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with np.load(dataset_dir / "dataset.npz") as z:
        return BuiltDataset(
            site=site,
            spec=spec,
            manifest=manifest,
            cutoff_ts=z["cutoff_ts"],
            X_hist=z["X_hist"],
            X_act_actual=z["X_act_actual"],
            X_act_hold=z["X_act_hold"],
            y_temp_delta=z["y_temp_delta"],
            y_temp_valid=z["y_temp_valid"],
            y_unit_wh=z["y_unit_wh"],
            y_unit_cov=z["y_unit_cov"],
            y_unit_valid=z["y_unit_valid"],
            y_total_wh=z["y_total_wh"],
            y_total_cov=z["y_total_cov"],
            y_total_valid=z["y_total_valid"],
            intervened=z["intervened"],
            metadata=meta_lines,
        )
