"""Read-only state collector, state-machine normalizer, and next-command forecaster.

Reads only persisted files produced by `continuous_pilot.py`, `service.py`,
`neural_shadow`, and static site config (`house_policy.json`, `rooms.json`,
`topology.json`). Never writes to controller directories or calls Home Assistant.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import isfinite
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

CYCLE_INTERVAL_SECONDS = 20
ECHO_TIMEOUT_SECONDS = 180
STATUS_STALE_SECONDS = 90
POWER_REPORT_GAP_SECONDS = 120
TEMP_EPSILON = 1e-6
ACTIVE_MODES = frozenset({"heat", "cool"})
DIRECTION = {"heat": 1, "cool": -1}

DEFAULT_POLICY: dict[str, Any] = {
    "ideal_band": 0.5,
    "comfort_band": 0.7,
    "ordinary_step": 1,
    "large_step": 2,
    "large_deviation": 2.0,
    "tracking_band": 0.3,
    "tracking_dwell_seconds": 1200,
    "tracking_report_span_seconds": 600,
    "recovery_settle_seconds": 600,
    "feedback_wait_seconds": 600,
    "reduction_wait_seconds": 180,
    "min_run_seconds": 900,
    "min_off_seconds": 300,
    "reversal_run_seconds": 3600,
    "lower_output_observe_seconds": 180,
    "peer_fairness_wait_seconds": 1800,
    "maintain_confirmation_seconds": 1200,
    "gentle_dwell_seconds": 1200,
    "gentle_report_span_seconds": 600,
    "gentle_exit_band": 0.3,
    "gentle_step": 1,
    "gentle_switch_wait_seconds": 1200,
}

INTENT_LABELS = {
    "observe": "观测 (Observe)",
    "recover": "恢复 (Recover)",
    "maintain": "维持 (Maintain)",
    "weaken": "减载 (Weaken)",
    "parked": "停机驻留 (Parked)",
    "limited": "受限 (Limited)",
}

ZONE_REASON_LABELS = {
    "data_missing": "测温数据缺失或已过期",
    "zone_paused": "该区域已暂停自动控制",
    "verify_stable_recovery": "已回到目标带内，等待连续稳定确认",
    "near_target": "处于目标跟踪带内，稳定维持",
    "returning": "温度趋势正在可信回归，暂缓加码",
    "confirmed_bias": "持续偏差已由独立测温上报证实",
    "waiting_reports_or_dwell": "正在累计持续偏差时间与跨时段独立上报",
    "unconfirmed_influencing_equipment": "影响本区的设备存在未确认指令锁",
    "conflicting_priority": "双区反向需求冲突，等待仲裁轮换",
    "no_safe_equipment": "当前无满足舒适保护的可控设备",
    "opposing_output_cannot_yield": "对侧反向运行设备暂不能安全让出",
    "no_comfort_safe_assignment": "无可避免邻区舒适风险的设备分配",
    "handoff_pending": "正在执行对侧设备协调退出交接",
}

HANDOFF_PHASE_LABELS = {
    "idle": "空闲（无反向冲突）",
    "elect": "仲裁选主 (Elect)",
    "limited": "交接受限 (Limited)",
    "reduce": "对侧减载 (Reduce)",
    "verify_reduction": "核验减载效果 (VerifyReduction)",
    "exit": "对侧停机 (Exit)",
    "verify_exit": "核验停机低耗 (VerifyExit)",
    "serve": "交接完成 · 服务优先区 (Serve)",
}

HANDOFF_REASON_LABELS = {
    "observe_both_sensors_and_low_power_after_off": "停机后等待双区新测温与连续低耗证据",
    "exit_verified": "对侧停机与低耗证据均已核验通过",
    "yielding_equipment_locked": "待退出设备处于锁定或不可控状态",
    "observe_both_sensors_after_reduction": "减载后等待双区新测温反馈",
    "reduced_or_proven_dormant_opposition": "已减载或证实低耗，准备执行对侧停机",
    "yield_one_bounded_step": "对侧存在有效输出，先执行一步受限减载",
    "yield_at_safe_boundary": "对侧已到软限边界，转为停机退出",
}

ACTUATOR_STATUS_LABELS = {
    "bounded_zone_correction": "区域偏差确认，准备小步加码",
    "start_for_confirmed_zone_error": "持续偏差确认，准备按需启机",
    "start_beside_unpowered_unit": "同区设备未证实输出，准备按需启用另一台",
    "exit_eased_dormant_overrun": "减载后已确认低耗，准备退出",
    "reduce_after_assistance_ended": "跨区辅助需求已结束，准备减载",
    "ease_within_minimum_runtime": "最短运行期内先减载",
    "maintain_existing_zone_assignment": "同区已有运行设备承担需求",
    "wait_opposite_or_observation": "等待反向退出或动作后观察期",
    "no_admissible_start": "当前不满足启机条件或未停机驻留",
    "exit_local_comfort_overrun": "本区超出舒适带，准备保护停机",
    "wait_new_local_feedback_after_reduction": "等待上次减载后的新室温反馈",
    "exit_dormant_opposition": "反向设备已证实低耗，准备停机退出",
    "exit_contrary_or_risky_output": "反向或风险输出持续存在，准备停机退出",
    "high_power_overrun": "持续高功率超理想线，直接减载",
    "reduce_at_assistance_limit": "跨区辅助触及本区跟踪边界，回退减载",
    "reduce_projected_or_neighbor_overrun": "预测过冲或邻区舒适风险，提前减载",
    "wait_verified_output_or_off_proof": "等待实测功率输出或低耗停机证据",
    "wait_coordinated_exit": "等待对侧反向设备协调退出",
    "unmet_conflicting_demand": "双区需求冲突，暂无法满足",
    "observe_local_response": "等待动作后观察期与新室温上报",
    "hold_effective_setting": "维持当前有效设温",
    "verify_previous_yielding_action": "核验前一次交接减载反馈",
}

PURPOSE_LABELS = {
    "zone_strengthen": "区域偏差恢复加码",
    "zone_reduce": "过冲 / 邻区风险减载",
    "min_run_easing": "最短运行保护期间减载",
    "energy_start": "按需启机",
    "energy_off": "保护 / 退载停机",
    "handoff_reduce": "双向交接 · 对侧减载",
    "handoff_off": "双向交接 · 对侧停机",
    "reversal_initial": "启机后补发初始设温",
    "initial_safety_easing": "继承越限设温安全归位",
    "reversal_off": "换向先停机",
    "reversal_mode": "换向切换模式",
    "gentle_temperature": "温和恢复调温",
    "direct_high_easing": "持续高功率直接减载",
    "trend_easing": "预测越界提前减载",
    "coordinated_easing": "协同减载",
    "temperature": "常规设温调整",
}

COMMAND_STATUS_LABELS = {
    "prepared": "已落盘待发",
    "accepted": "已受理 · 等待伴侣回报",
    "proxy_echo": "伴侣已回报",
    "unconfirmed": "回报超时或异常（已加锁）",
    "rejected": "HA 调用被拒绝",
    "canceled": "发令前已取消",
}

OUTCOME_LABELS = {
    "idle": "常规巡检（维持现状）",
    "dispatch": "已下发实机指令",
    "recheck_changed": "发令前复核条件变化，取消发令",
    "permit_inactive": "控制许可未启用或不在窗口内",
    "permit_expiring": "许可临近结束，停止新发令",
    "permit_revoked": "发令前许可撤销，已取消",
}

MODE_LABELS = {
    "heat": "制热",
    "cool": "制冷",
    "off": "关机",
    "hold": "维持",
    "fan_only": "送风",
    "unknown": "未知",
    "conflict": "双向冲突",
    "unavailable": "不可用",
}

ZONE_FRIENDLY_NAMES = {
    "sleep": "卧室区",
    "lounge": "客厅区",
    "office": "书房区",
    "bedroom": "卧室",
    "living": "客厅",
}


@dataclass(frozen=True)
class DashboardPaths:
    """Read-only filesystem locations inspected by the dashboard."""

    shadow_config_dir: Path
    pilot_data_dir: Path
    shadow_data_dir: Path
    shadow_output_dir: Path
    neural_output_dir: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "DashboardPaths":
        ha_dir = repo_root / "home-assistant"
        return cls(
            shadow_config_dir=ha_dir / "shadow",
            pilot_data_dir=ha_dir / "pilot-data",
            shadow_data_dir=ha_dir / "shadow-data",
            shadow_output_dir=repo_root / "outputs" / "climate-shadow",
            neural_output_dir=repo_root / "outputs" / "neural-shadow",
        )

    @classmethod
    def from_env(cls, default_root: Path | None = None) -> "DashboardPaths":
        root = default_root or Path(__file__).resolve().parents[2]
        base = cls.from_repo_root(root)
        return cls(
            shadow_config_dir=Path(os.getenv("THERMOSTAT_SHADOW_CONFIG_DIR", str(base.shadow_config_dir))),
            pilot_data_dir=Path(os.getenv("THERMOSTAT_PILOT_DATA_DIR", str(base.pilot_data_dir))),
            shadow_data_dir=Path(os.getenv("THERMOSTAT_SHADOW_DATA_DIR", str(base.shadow_data_dir))),
            shadow_output_dir=Path(os.getenv("THERMOSTAT_SHADOW_OUTPUT_DIR", str(base.shadow_output_dir))),
            neural_output_dir=Path(os.getenv("THERMOSTAT_NEURAL_OUTPUT_DIR", str(base.neural_output_dir))),
        )


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _parse_ts(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if isfinite(float(value)) else None
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, type(exc).__name__


def _tail_jsonl(path: Path, max_lines: int = 80, max_bytes: int = 262_144) -> tuple[list[dict[str, Any]], str | None]:
    if not path.is_file():
        return [], None
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
                handle.readline()  # discard partial first line
            raw_lines = handle.read().splitlines()
    except OSError as exc:
        return [], type(exc).__name__
    records: list[dict[str, Any]] = []
    for raw in raw_lines[-max_lines:]:
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records, None


def _setpoint_bounds(limits: Any) -> tuple[float, float] | None:
    if not isinstance(limits, (list, tuple)) or len(limits) != 2:
        return None
    lo, hi = _num(limits[0]), _num(limits[1])
    if lo is None or hi is None:
        return None
    lo, hi = max(16.0, lo), min(30.0, hi)
    return (lo, hi) if lo <= hi else None


def _automatic_setpoint(prior: float | None, step: float, limits: Any) -> float | None:
    bounds = _setpoint_bounds(limits)
    if bounds is None or prior is None or not 16 <= prior <= 30:
        return None
    lo, hi = bounds
    if step < 0 and prior > lo:
        return max(lo, min(prior, hi) + step)
    if step > 0 and prior < hi:
        return min(hi, max(prior, lo) + step)
    return None


def _initial_safety_setting(mode: str, setting: float | None, target: float | None, limits: Any) -> float | None:
    bounds = _setpoint_bounds(limits)
    if bounds is None or setting is None or target is None:
        return None
    excess = (mode == "heat" and setting > bounds[1]) or (mode == "cool" and setting < bounds[0])
    if not excess:
        return None
    return max(bounds[0], min(bounds[1], round(target)))


def _learned_initial(
    room: str,
    mode: str,
    zone: dict[str, Any],
    closed_loop: dict[str, Any],
    limits: Any,
    policy: dict[str, Any],
) -> float | None:
    bounds = _setpoint_bounds(limits)
    target = _num(zone.get("target"))
    if bounds is None or target is None:
        return None
    learned = (closed_loop.get("maintenance") or {}).get(room, {}).get(mode)
    chosen = float(round(target))
    step = _num(policy.get("ordinary_step")) or 1.0
    if (
        isinstance(learned, dict)
        and learned.get("source_version") == zone.get("source_version")
        and _num(learned.get("target")) == target
        and _num(learned.get("setpoint")) is not None
    ):
        chosen = max(chosen - step, min(chosen + step, float(learned["setpoint"])))
    return max(bounds[0], min(bounds[1], chosen))


def _load_config(
    paths: DashboardPaths,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    read_errors: list[dict[str, str]] = []
    policy_raw, policy_err = _read_json(paths.shadow_config_dir / "house_policy.json")
    if policy_err is not None:
        read_errors.append({"file": "house_policy.json", "error": policy_err})
    policy = dict(DEFAULT_POLICY)
    if isinstance(policy_raw, dict):
        policy.update(policy_raw)

    rooms_live = paths.shadow_config_dir / "rooms.json"
    rooms_example = paths.shadow_config_dir / "rooms.example.json"
    using_example_rooms = False
    rooms_raw, rooms_err = _read_json(rooms_live)
    if rooms_err is not None:
        read_errors.append({"file": "rooms.json", "error": rooms_err})
    elif not isinstance(rooms_raw, dict) and not rooms_live.exists():
        rooms_raw, _ = _read_json(rooms_example)
        using_example_rooms = isinstance(rooms_raw, dict)
    rooms = rooms_raw if isinstance(rooms_raw, dict) else {}

    topo_live = paths.shadow_config_dir / "topology.json"
    topo_example = paths.shadow_config_dir / "topology.example.json"
    topo_raw, topo_err = _read_json(topo_live)
    if topo_err is not None:
        read_errors.append({"file": "topology.json", "error": topo_err})
        topo_raw = None
    elif not isinstance(topo_raw, dict):
        topo_raw = None
    topology = topo_raw if isinstance(topo_raw, dict) else None
    meta = {
        "rooms_from_example": using_example_rooms,
        "topology_from_example": False,
        "topology_live_present": topo_live.exists(),
        "topology_example_exists": topo_example.is_file(),
        "read_errors": read_errors,
    }
    return policy, rooms, topology, meta


def _load_permit(
    pilot_dir: Path,
    now: float,
    rooms_cfg: dict[str, Any] | None = None,
    topology: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    raw = None
    source = None
    for filename in ("live-permit.json", "continuous-permit.json"):
        data, err = _read_json(pilot_dir / filename)
        if err is not None:
            return None, {
                "present": True,
                "source": filename,
                "active": False,
                "enabled": False,
                "trial": filename == "continuous-permit.json",
                "start": None,
                "end": None,
                "remaining_s": None,
                "isolated_rooms": [],
                "paused_zones": [],
                "reason": f"许可文件解析失败 ({err})",
            }
        if isinstance(data, dict):
            raw = data
            source = filename
            break
    if raw is None:
        return None, {
            "present": False,
            "source": None,
            "active": False,
            "enabled": False,
            "trial": False,
            "start": None,
            "end": None,
            "remaining_s": None,
            "isolated_rooms": [],
            "paused_zones": [],
            "reason": "未找到许可文件 (live-permit.json / continuous-permit.json)",
        }

    enabled = raw.get("enabled") is True
    start = _num(raw.get("start"))
    end_raw = raw.get("end")
    end = _num(end_raw) if end_raw is not None else None
    trial = end_raw is not None
    isolated_raw = raw.get("isolated_rooms", [])
    paused_raw = raw.get("paused_zones", [])
    isolated = list(isolated_raw) if isinstance(isolated_raw, list) else []
    paused_zones = list(paused_raw) if isinstance(paused_raw, list) else []
    permit_rooms = raw.get("rooms")
    expected_rooms = set(rooms_cfg.keys()) if isinstance(rooms_cfg, dict) and rooms_cfg else None
    topo_zones = topology.get("zones") if isinstance(topology, dict) and isinstance(topology.get("zones"), dict) else None

    def _valid_baseline(sig: Any) -> bool:
        if not isinstance(sig, list) or len(sig) != 3:
            return False
        mode, setpoint, _ = sig
        if mode not in ACTIVE_MODES:
            return False
        val = _num(setpoint)
        return val is not None and 16 <= val <= 30

    reason = None
    active = False
    if not enabled:
        reason = "许可开关 enabled=false（已暂停）"
    elif start is None:
        reason = "许可缺少有效起始时间"
    elif end_raw is not None and end is None:
        reason = "许可结束时间无效"
    elif source == "continuous-permit.json" and end is None:
        reason = "试点许可缺少结束时间"
    elif start > now:
        reason = "尚未到达许可生效时间"
    elif end is not None and now >= end:
        reason = "许可窗口已到期"
    elif not isinstance(permit_rooms, dict) or not permit_rooms:
        reason = "许可缺少有效的房间基线 (rooms)"
    elif expected_rooms is not None and set(permit_rooms.keys()) != expected_rooms:
        reason = "许可房间列表与 rooms.json 不一致"
    elif any(not _valid_baseline(sig) for sig in permit_rooms.values()):
        reason = "许可基线格式无效（须为 [cool|heat, 16..30, fan]）"
    elif not isinstance(isolated_raw, list) or any(
        not isinstance(r, str) or (expected_rooms is not None and r not in expected_rooms) for r in isolated_raw
    ):
        reason = "许可 isolated_rooms 包含无效设备"
    elif not isinstance(paused_raw, list) or (
        paused_raw and (topo_zones is None or any(not isinstance(z, str) or z not in topo_zones for z in paused_raw))
    ):
        reason = "许可 paused_zones 与当前拓扑区域不匹配"
    else:
        active = True

    remaining_s = max(0.0, end - now) if (active and end is not None) else None
    return raw, {
        "present": True,
        "source": source,
        "active": active,
        "enabled": enabled,
        "trial": trial,
        "start": start,
        "end": end,
        "remaining_s": round(remaining_s, 1) if remaining_s is not None else None,
        "isolated_rooms": isolated,
        "paused_zones": paused_zones,
        "reason": reason,
    }


def _load_journal(
    pilot_dir: Path, permit_raw: dict[str, Any] | None
) -> tuple[dict[str, Any], str | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    if isinstance(permit_raw, dict) and _num(permit_raw.get("start")) is not None:
        candidate = pilot_dir / f"continuous-{int(float(permit_raw['start']))}.json"
        data, err = _read_json(candidate)
        if err is not None:
            errors.append({"file": candidate.name, "error": err})
        if isinstance(data, dict):
            return data, candidate.name, errors

    candidates: list[tuple[int, Path]] = []
    if pilot_dir.is_dir():
        for path in pilot_dir.glob("continuous-*.json"):
            suffix = path.stem.removeprefix("continuous-")
            if suffix.isdigit():
                candidates.append((int(suffix), path))
    for _, path in sorted(candidates, reverse=True):
        data, err = _read_json(path)
        if err is not None and not any(e["file"] == path.name for e in errors):
            errors.append({"file": path.name, "error": err})
        if isinstance(data, dict):
            return data, path.name, errors
    return {}, None, errors


def _load_shadow_series(shadow_data_dir: Path, rooms: dict[str, Any], now: float, window_s: int = 7200) -> dict[str, list[dict[str, Any]]]:
    db_path = shadow_data_dir / "shadow.sqlite3"
    if not db_path.is_file():
        return {}
    series: dict[str, list[dict[str, Any]]] = {}
    since = int(now - window_s)
    try:
        with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True, timeout=1.5)) as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("PRAGMA temp_store=MEMORY")
            db.row_factory = sqlite3.Row
            for room in rooms:
                rows = db.execute(
                    "SELECT tick, value FROM decisions WHERE room=? AND tick>=? ORDER BY tick ASC LIMIT 180",
                    (room, since),
                ).fetchall()
                points = []
                for row in rows:
                    try:
                        val = json.loads(row["value"])
                    except json.JSONDecodeError:
                        continue
                    points.append({
                        "t": row["tick"],
                        "temp": _num(val.get("measured")) if val.get("source_fresh") else None,
                        "target": _num(val.get("target")),
                        "watts": _num(val.get("power_report_w") if val.get("power_report_w") is not None else val.get("watts")),
                        "mode": val.get("reported_mode"),
                        "setpoint": _num(val.get("from_setpoint")),
                    })
                if points:
                    series[room] = points
    except (sqlite3.Error, OSError):
        return {}
    return series


def _format_value(kind: str, value: Any) -> str:
    if kind == "temperature":
        v = _num(value)
        return f"{v:g}℃" if v is not None else str(value)
    return MODE_LABELS.get(str(value), str(value))


def _format_action_display(room_name: str, kind: str, before_val: Any, to_val: Any) -> str:
    if kind == "temperature":
        if before_val is not None:
            return f"{room_name} · 设温 {_format_value('temperature', before_val)} → {_format_value('temperature', to_val)}"
        return f"{room_name} · 设温 → {_format_value('temperature', to_val)}"
    if before_val is not None:
        return f"{room_name} · 模式 {_format_value('mode', before_val)} → {_format_value('mode', to_val)}"
    return f"{room_name} · 模式 → {_format_value('mode', to_val)}"


def _service_call_text(kind: str, value: Any) -> str:
    if kind == "temperature":
        v = _num(value)
        val_str = f"{v:g}" if v is not None else str(value)
        return f"climate.set_temperature(temperature={val_str})"
    return f"climate.set_hvac_mode(hvac_mode='{value}')"


def _room_display_name(room: str, rooms_cfg: dict[str, Any]) -> str:
    cfg = rooms_cfg.get(room)
    if isinstance(cfg, dict) and cfg.get("name"):
        name = str(cfg["name"])
        zh = {"Bedroom": "卧室", "Living room": "客厅", "Study": "书房"}.get(name)
        return f"{zh} ({room})" if zh else name
    return {"bedroom": "卧室 (bedroom)", "living": "客厅 (living)", "study": "书房 (study)"}.get(room, room)


def _zone_display_name(
    zone_key: str,
    topology: dict[str, Any] | None = None,
    zone_keys: Any = None,
) -> str:
    if zone_key == "lounge":
        known_zones = set(zone_keys or ())
        if isinstance(topology, dict) and isinstance(topology.get("zones"), dict):
            known_zones.update(topology["zones"].keys())
        actuators = topology.get("actuators") if isinstance(topology, dict) and isinstance(topology.get("actuators"), dict) else {}
        study_in_lounge = isinstance(actuators.get("study"), dict) and actuators["study"].get("zone") == "lounge"
        if study_in_lounge and "office" not in known_zones:
            return "客厅 / 书房区 (lounge)"
        return "客厅区 (lounge)"
    friendly = ZONE_FRIENDLY_NAMES.get(zone_key)
    return f"{friendly} ({zone_key})" if friendly else zone_key


def _format_duration_zh(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    secs = max(0, int(round(seconds)))
    if secs < 60:
        return f"{secs}秒"
    mins, rem = divmod(secs, 60)
    if mins < 60:
        return f"{mins}分{rem:02d}秒" if rem else f"{mins}分钟"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}小时{mins}分"


def _normalize_command(action: dict[str, Any], rooms_cfg: dict[str, Any], now: float) -> dict[str, Any]:
    room = str(action.get("room") or "")
    kind = str(action.get("kind") or "")
    value = action.get("value")
    before = action.get("before") if isinstance(action.get("before"), list) else []
    before_mode = before[0] if len(before) > 0 else None
    before_setpoint = _num(before[1]) if len(before) > 1 else None
    before_val = before_setpoint if kind == "temperature" else before_mode
    sent_at = _num(action.get("sent_at"))
    relay_at = _num(action.get("relay_at"))
    proxy_at = _num(action.get("proxy_at"))
    status = str(action.get("status") or "unknown")
    purpose = str(action.get("purpose") or "")
    room_name = _room_display_name(room, rooms_cfg)
    return {
        "room": room,
        "room_name": room_name,
        "kind": kind,
        "before": before,
        "before_mode": before_mode,
        "before_setpoint": before_setpoint,
        "value": value,
        "display": _format_action_display(room_name, kind, before_val, value),
        "short_transition": (
            f"{_format_value(kind, before_val)} → {_format_value(kind, value)}"
            if before_val is not None
            else f"→ {_format_value(kind, value)}"
        ),
        "purpose": purpose,
        "purpose_label": PURPOSE_LABELS.get(purpose, purpose or "—"),
        "status": status,
        "status_label": COMMAND_STATUS_LABELS.get(status, status),
        "sent_at": sent_at,
        "age_s": round(max(0.0, now - sent_at), 1) if sent_at is not None else None,
        "relay_at": relay_at,
        "proxy_at": proxy_at,
        "relay_delay_s": round(relay_at - sent_at, 2) if (relay_at is not None and sent_at is not None) else None,
        "echo_delay_s": round(proxy_at - sent_at, 1) if (proxy_at is not None and sent_at is not None) else None,
        "http": action.get("http"),
        "transaction": action.get("transaction"),
        "target_at_send": _num(action.get("target_at_send")),
        "before_room_c": _num(action.get("before_room_c")),
        "before_watts": _num(action.get("before_watts")),
        "expected": action.get("expected"),
        "followup": action.get("followup") if isinstance(action.get("followup"), dict) else {},
        "physical_verified": bool(action.get("physical_verified")),
        "control": action.get("control") if isinstance(action.get("control"), dict) else None,
    }


def _build_zone_states(
    control: dict[str, Any],
    closed_loop: dict[str, Any],
    policy: dict[str, Any],
    now: float,
    topology: dict[str, Any] | None,
    status_rooms: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    zones_raw = control.get("zones") if isinstance(control.get("zones"), dict) else {}
    owner = control.get("owner")
    feasible = control.get("feasible") if isinstance(control.get("feasible"), dict) else {}
    unmet = control.get("unmet") if isinstance(control.get("unmet"), dict) else {}
    yielding_blocks = control.get("yielding_blocks") if isinstance(control.get("yielding_blocks"), dict) else {}
    trackers = closed_loop.get("trackers") if isinstance(closed_loop.get("trackers"), dict) else {}

    dwell_req = float(policy.get("tracking_dwell_seconds", 1200))
    span_req = float(policy.get("tracking_report_span_seconds", 600))
    settle_req = float(policy.get("recovery_settle_seconds", 600))

    zone_order = list(zones_raw.keys())
    if not zone_order and isinstance(topology, dict) and isinstance(topology.get("zones"), dict):
        zone_order = list(topology["zones"].keys())

    # Legacy pre-topology mode fallback: derive independent bedroom/living views from status.json rooms.
    if not zone_order and isinstance(status_rooms, dict) and status_rooms:
        legacy_result: list[dict[str, Any]] = []
        for room_key in ("bedroom", "living"):
            detail = status_rooms.get(room_key)
            if not isinstance(detail, dict):
                continue
            measured = _num(detail.get("room_c"))
            target = _num(detail.get("target_c"))
            signed_delta = round(measured - target, 3) if (measured is not None and target is not None) else None
            error = round(target - measured, 3) if (measured is not None and target is not None) else None
            cold_s = _num(detail.get("cold_seconds")) or 0.0
            warm_s = _num(detail.get("warm_seconds")) or 0.0
            dwell_s = max(cold_s, warm_s)
            direction = "heat" if (cold_s > 0 or (signed_delta is not None and signed_delta < -policy["comfort_band"])) else (
                "cool" if (warm_s > 0 or (signed_delta is not None and signed_delta > policy["comfort_band"])) else None
            )
            qualified = bool(
                (cold_s >= dwell_req and detail.get("cold_confirmed"))
                or (warm_s >= dwell_req and detail.get("warm_confirmed"))
            )
            if measured is None or target is None:
                intent, reason = "limited", "data_missing"
            elif qualified:
                intent, reason = "recover", "confirmed_bias"
            elif signed_delta is not None and abs(signed_delta) <= float(policy.get("ideal_band", 0.5)) + TEMP_EPSILON:
                intent, reason = "maintain", "near_target"
            else:
                intent, reason = "observe", "waiting_reports_or_dwell"
            pred = _num(detail.get("prediction_10m_c"))
            trend = (
                {"delta_10m_c": round(pred, 3), "uncertainty_c": 0.0, "projected_10m_c": round(measured + pred, 2) if measured is not None else None, "reports": 0, "from_ts": None, "to_ts": None}
                if pred is not None
                else None
            )
            at_ts = _num(detail.get("temperature_report_at"))
            legacy_result.append({
                "zone": room_key,
                "label": _zone_display_name(room_key),
                "state": intent,
                "state_label": INTENT_LABELS.get(intent, intent),
                "reason": reason,
                "reason_label": ZONE_REASON_LABELS.get(reason, reason),
                "direction": direction,
                "direction_label": MODE_LABELS.get(direction, "—") if direction else "无方向需求",
                "qualified": qualified,
                "paused": not bool(detail.get("ready")),
                "is_owner": False,
                "quality": "fresh" if (measured is not None and (_num(detail.get("temperature_age_s")) or 9999) <= 1200) else "stale",
                "measured_c": round(measured, 2) if measured is not None else None,
                "target_c": round(target, 2) if target is not None else None,
                "error_c": error,
                "signed_delta_c": signed_delta,
                "trend": trend,
                "dwell": {
                    "duration_s": round(dwell_s, 1),
                    "required_s": dwell_req,
                    "remaining_s": round(max(0.0, dwell_req - dwell_s), 1),
                    "pct": min(100, int(round(100 * dwell_s / dwell_req))) if dwell_req > 0 else 100,
                },
                "independent_reports": {
                    "count": 2 if qualified else (1 if dwell_s > 0 else 0),
                    "required_count": 2,
                    "span_s": span_req if qualified else 0.0,
                    "required_span_s": span_req,
                    "remaining_span_s": 0.0 if qualified else span_req,
                    "pct": 100 if qualified else 0,
                },
                "settling": {"active": False, "reports": 0, "required_count": 2, "span_s": 0.0, "required_span_s": settle_req, "remaining_span_s": settle_req},
                "recovery_active": qualified,
                "evidence_from": None,
                "last_report_at": at_ts,
                "report_age_s": _num(detail.get("temperature_age_s")),
                "source_version": None,
                "feasible_actuators": [room_key] if detail.get("ready") else [],
                "unmet_reason": None,
                "unmet_reason_label": None,
                "yielding_block": None,
                "points": [],
                "constraints": {},
            })
        return legacy_result

    result: list[dict[str, Any]] = []
    for key in zone_order:
        z = zones_raw.get(key) if isinstance(zones_raw.get(key), dict) else {}
        measured = _num(z.get("measured"))
        target = _num(z.get("target"))
        error = _num(z.get("error"))
        if error is None and measured is not None and target is not None:
            error = round(target - measured, 4)
        signed_delta = round(measured - target, 3) if (measured is not None and target is not None) else None
        intent = str(z.get("intent") or "limited")
        reason = str(z.get("reason") or "data_missing")
        duration = _num(z.get("duration")) or 0.0
        report_span = _num(z.get("report_span")) or 0.0
        reports = int(z.get("reports") or 0)
        settling_reports = int(z.get("settling_reports") or 0)
        settling_span = _num(z.get("settling_span")) or 0.0
        at_ts = _num(z.get("at"))
        trend_raw = z.get("trend") if isinstance(z.get("trend"), dict) else None
        trend = None
        if trend_raw is not None and _num(trend_raw.get("delta")) is not None:
            delta_10m = float(trend_raw["delta"])
            unc = _num(trend_raw.get("uncertainty")) or 0.0
            trend = {
                "delta_10m_c": round(delta_10m, 3),
                "uncertainty_c": round(unc, 3),
                "projected_10m_c": round(measured + delta_10m, 2) if measured is not None else None,
                "reports": int(trend_raw.get("reports") or 0),
                "from_ts": _num(trend_raw.get("from")),
                "to_ts": _num(trend_raw.get("to")),
            }

        tracker = trackers.get(key) if isinstance(trackers.get(key), dict) else {}
        raw_points = tracker.get("points") if isinstance(tracker.get("points"), list) else []
        points = []
        for pt in raw_points[-120:]:
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                t_val, temp_val = _num(pt[0]), _num(pt[1])
                if t_val is not None and temp_val is not None:
                    points.append([t_val, round(temp_val, 3)])

        dwell_rem = max(0.0, dwell_req - duration)
        span_rem = max(0.0, span_req - report_span)
        track_band = _num(policy.get("tracking_band"))
        if reason == "near_target" and track_band is not None:
            reason_label = f"处于目标 ±{track_band:g}℃ 跟踪带内，稳定维持"
        else:
            reason_label = ZONE_REASON_LABELS.get(reason, reason)
        result.append({
            "zone": key,
            "label": _zone_display_name(key, topology=topology, zone_keys=zone_order),
            "state": intent,
            "state_label": INTENT_LABELS.get(intent, intent),
            "reason": reason,
            "reason_label": reason_label,
            "direction": z.get("direction"),
            "direction_label": MODE_LABELS.get(str(z.get("direction")), "—") if z.get("direction") else "无方向需求",
            "qualified": bool(z.get("qualified")),
            "paused": bool(z.get("paused")),
            "is_owner": owner == key,
            "quality": str(z.get("quality") or "unavailable"),
            "measured_c": round(measured, 2) if measured is not None else None,
            "target_c": round(target, 2) if target is not None else None,
            "error_c": round(error, 3) if error is not None else None,
            "signed_delta_c": signed_delta,
            "trend": trend,
            "dwell": {
                "duration_s": round(duration, 1),
                "required_s": dwell_req,
                "remaining_s": round(dwell_rem, 1),
                "pct": min(100, int(round(100 * duration / dwell_req))) if dwell_req > 0 else 100,
            },
            "independent_reports": {
                "count": reports,
                "required_count": 2,
                "span_s": round(report_span, 1),
                "required_span_s": span_req,
                "remaining_span_s": round(span_rem, 1),
                "pct": min(100, int(round(100 * report_span / span_req))) if span_req > 0 else 100,
            },
            "settling": {
                "active": reason == "verify_stable_recovery",
                "reports": settling_reports,
                "required_count": 2,
                "span_s": round(settling_span, 1),
                "required_span_s": settle_req,
                "remaining_span_s": round(max(0.0, settle_req - settling_span), 1),
            },
            "recovery_active": bool(z.get("recovery_active")),
            "evidence_from": _num(z.get("evidence_from")),
            "last_report_at": at_ts,
            "report_age_s": round(max(0.0, now - at_ts), 1) if at_ts is not None else None,
            "source_version": z.get("source_version"),
            "feasible_actuators": list(feasible.get(key) or []),
            "unmet_reason": unmet.get(key),
            "unmet_reason_label": ZONE_REASON_LABELS.get(str(unmet.get(key)), str(unmet.get(key))) if unmet.get(key) else None,
            "yielding_block": yielding_blocks.get(key),
            "points": points,
            "constraints": z.get("constraints") if isinstance(z.get("constraints"), dict) else {},
        })
    return result


def _build_handoff_state(control: dict[str, Any], closed_loop: dict[str, Any], now: float) -> dict[str, Any]:
    handoff = control.get("handoff") if isinstance(control.get("handoff"), dict) else None
    conflict = bool(control.get("conflict"))
    yielding_blocks = control.get("yielding_blocks") if isinstance(control.get("yielding_blocks"), dict) else {}
    if handoff is None:
        persisted = closed_loop.get("handoff") if isinstance(closed_loop.get("handoff"), dict) else None
        if persisted and conflict:
            handoff = dict(persisted, phase="elect", reason=None)
    if handoff is None:
        phase = "limited" if (conflict and yielding_blocks) else "idle"
        return {
            "active": conflict,
            "conflict": conflict,
            "phase": phase,
            "phase_label": HANDOFF_PHASE_LABELS.get(phase, phase),
            "reason": next(iter(yielding_blocks.values()), None) if conflict else None,
            "reason_label": (
                ZONE_REASON_LABELS.get(str(next(iter(yielding_blocks.values()))), str(next(iter(yielding_blocks.values()))))
                if (conflict and yielding_blocks)
                else None
            ),
            "owner": control.get("owner"),
            "receiver": control.get("elected"),
            "blocker": None,
            "direction": None,
            "selected_at": None,
            "elapsed_s": None,
            "yielding_blocks": yielding_blocks,
        }
    phase = str(handoff.get("phase") or "elect")
    reason = handoff.get("reason")
    selected_at = _num(handoff.get("selected_at"))
    return {
        "active": True,
        "conflict": conflict,
        "phase": phase,
        "phase_label": HANDOFF_PHASE_LABELS.get(phase, phase),
        "reason": reason,
        "reason_label": HANDOFF_REASON_LABELS.get(str(reason), str(reason)) if reason else None,
        "owner": handoff.get("owner"),
        "receiver": handoff.get("receiver"),
        "blocker": handoff.get("blocker"),
        "direction": handoff.get("direction"),
        "selected_at": selected_at,
        "elapsed_s": round(max(0.0, now - selected_at), 1) if selected_at is not None else None,
        "yielding_blocks": yielding_blocks,
    }


def _build_actuators(
    rooms_cfg: dict[str, Any],
    status_rooms: dict[str, Any],
    control: dict[str, Any],
    journal: dict[str, Any],
    permit_info: dict[str, Any],
    topology: dict[str, Any] | None,
    shadow_status: dict[str, Any] | None,
    policy: dict[str, Any],
    now: float,
) -> list[dict[str, Any]]:
    topo_actuators = topology.get("actuators") if isinstance(topology, dict) and isinstance(topology.get("actuators"), dict) else {}
    holds = journal.get("holds") if isinstance(journal.get("holds"), dict) else {}
    manual_holds = journal.get("manual_holds") if isinstance(journal.get("manual_holds"), dict) else {}
    parked_map = journal.get("parked") if isinstance(journal.get("parked"), dict) else {}
    closed_loop = journal.get("closed_loop") if isinstance(journal.get("closed_loop"), dict) else {}
    maintenance = closed_loop.get("maintenance") if isinstance(closed_loop.get("maintenance"), dict) else {}
    output_evidence = control.get("output_evidence") if isinstance(control.get("output_evidence"), dict) else {}
    risks_map = control.get("risks") if isinstance(control.get("risks"), dict) else {}
    obs_after = control.get("observations_after") if isinstance(control.get("observations_after"), dict) else {}
    equipment_map = control.get("equipment") if isinstance(control.get("equipment"), dict) else {}
    energy_scores = control.get("energy_scores") if isinstance(control.get("energy_scores"), dict) else {}
    isolated = set(permit_info.get("isolated_rooms") or ())
    shadow_rooms = shadow_status.get("rooms") if isinstance(shadow_status, dict) and isinstance(shadow_status.get("rooms"), dict) else {}

    min_off_req = float(policy.get("min_off_seconds", 300))
    rev_req = float(policy.get("reversal_run_seconds", 3600))

    ordered_rooms = list(rooms_cfg.keys()) or list(status_rooms.keys())
    for extra in status_rooms:
        if extra not in ordered_rooms:
            ordered_rooms.append(extra)

    actuators: list[dict[str, Any]] = []
    for room in ordered_rooms:
        cfg = rooms_cfg.get(room) if isinstance(rooms_cfg.get(room), dict) else {}
        detail = status_rooms.get(room) if isinstance(status_rooms.get(room), dict) else {}
        unit = topo_actuators.get(room) if isinstance(topo_actuators.get(room), dict) else {}
        lock_group = str(
            unit.get("lock_group")
            or (room if topology is not None else "living_study" if room in {"living", "study"} else room)
        )
        home_zone = (
            detail.get("zone")
            or unit.get("zone")
            or (room if topology is not None else "living" if room == "study" else room)
        )
        equip_entry = equipment_map.get(room) if isinstance(equipment_map.get(room), dict) else {}
        served_zone = equip_entry.get("served_zone") or home_zone
        parked_raw = parked_map.get(room) if isinstance(parked_map.get(room), dict) else None
        parked_info = None
        if parked_raw is not None:
            off_at = _num(parked_raw.get("off_at"))
            started_at = _num(parked_raw.get("started_at"))
            off_elapsed = max(0.0, now - off_at) if off_at is not None else 0.0
            rev_elapsed = max(0.0, now - started_at) if started_at is not None else 0.0
            parked_info = {
                "mode": parked_raw.get("mode"),
                "off_at": off_at,
                "started_at": started_at,
                "handoff": bool(parked_raw.get("handoff")),
                "off_elapsed_s": round(off_elapsed, 1),
                "min_off_required_s": min_off_req,
                "min_off_remaining_s": round(max(0.0, min_off_req - off_elapsed), 1),
                "reversal_elapsed_s": round(rev_elapsed, 1),
                "reversal_required_s": rev_req,
                "reversal_remaining_s": round(max(0.0, rev_req - rev_elapsed), 1),
            }
        mode = str(detail.get("mode_reported") or "unknown")
        status_code = str(detail.get("status") or "")
        status_label = (
            ACTUATOR_STATUS_LABELS.get(status_code)
            or HANDOFF_REASON_LABELS.get(status_code)
            or ZONE_REASON_LABELS.get(status_code)
            or status_code
            or "—"
        )

        shadow_entry = shadow_rooms.get(room) if isinstance(shadow_rooms.get(room), dict) else {}
        shadow_latest_raw = shadow_entry.get("latest") if isinstance(shadow_entry.get("latest"), dict) else {}
        shadow_totals = shadow_entry.get("totals") if isinstance(shadow_entry.get("totals"), dict) else {}
        shadow_latest = None
        if shadow_latest_raw or shadow_totals:
            shadow_latest = {
                "activity": shadow_latest_raw.get("activity"),
                "confidence": shadow_latest_raw.get("confidence"),
                "power_reason": shadow_latest_raw.get("power_reason"),
                "within_comfort_pct": _num(shadow_totals.get("within_comfort_pct")),
                "within_ideal_pct": _num(shadow_totals.get("within_ideal_pct")),
                "mae_c": _num(shadow_totals.get("mae_c")),
                "forecast_15m_mae_c": _num(shadow_totals.get("forecast_15m_mae_c")),
                "metered_kwh": _num(shadow_totals.get("metered_kwh")),
                "coverage_pct": _num(shadow_entry.get("coverage_pct")),
                "model_centers_w": (shadow_latest_raw.get("model") or {}).get("centers_w") if isinstance(shadow_latest_raw.get("model"), dict) else None,
                "model_edges_w": (shadow_latest_raw.get("model") or {}).get("edges_w") if isinstance(shadow_latest_raw.get("model"), dict) else None,
            }

        power_bands = detail.get("power_bands") or equip_entry.get("bands") or {}
        if not isinstance(power_bands, dict):
            power_bands = {}

        actuators.append({
            "room": room,
            "name": _room_display_name(room, rooms_cfg),
            "zone": home_zone,
            "served_zone": served_zone,
            "influences": unit.get("influences") if isinstance(unit.get("influences"), dict) else {home_zone: 1.0},
            "lock_group": lock_group,
            "priority": int(unit.get("priority") or 0),
            "mode": mode,
            "mode_label": MODE_LABELS.get(mode, mode),
            "setpoint": _num(detail.get("setpoint_reported")),
            "soft_limits": cfg.get("soft_limits") if isinstance(cfg.get("soft_limits"), dict) else {},
            "room_c": _num(detail.get("room_c")),
            "target_c": _num(detail.get("target_c")),
            "power_w": _num(detail.get("power_w")),
            "power_age_s": _num(detail.get("power_age_s")),
            "temperature_age_s": _num(detail.get("temperature_age_s")),
            "temperature_report_at": _num(detail.get("temperature_report_at")),
            "power_report_at": _num(detail.get("power_report_at")),
            "power_evidence_at": _num(detail.get("power_evidence_at") or equip_entry.get("power_at")),
            "ready": bool(detail.get("ready")),
            "parked": mode == "off" and parked_info is not None,
            "parked_info": parked_info,
            "output_credible": bool(output_evidence.get(room)),
            "hold_reason": holds.get(lock_group),
            "manual_hold_reason": manual_holds.get(room),
            "isolated": room in isolated,
            "status_code": status_code,
            "status_label": status_label,
            "power_bands": power_bands,
            "risks": risks_map.get(room) if isinstance(risks_map.get(room), list) else [],
            "observations_after": obs_after.get(room) if isinstance(obs_after.get(room), dict) else {},
            "learned_maintenance": maintenance.get(room) if isinstance(maintenance.get(room), dict) else {},
            "energy_score_wh_per_c": _num(energy_scores.get(room)),
            "shadow_latest": shadow_latest,
        })
    return actuators


def _build_ledger_state(
    journal: dict[str, Any],
    commands: list[dict[str, Any]],
    permit_info: dict[str, Any],
    actuators: list[dict[str, Any]],
    now: float,
) -> dict[str, Any]:
    pending = [c for c in commands if c["status"] in {"prepared", "accepted"}]
    unconfirmed = [c for c in commands if c["status"] == "unconfirmed"]
    echoed = [c for c in commands if c["status"] == "proxy_echo"]
    holds = journal.get("holds") if isinstance(journal.get("holds"), dict) else {}
    manual_holds = journal.get("manual_holds") if isinstance(journal.get("manual_holds"), dict) else {}

    if any(c["status"] == "prepared" for c in pending):
        active_phase = "prepared"
    elif any(c["status"] == "accepted" for c in pending):
        active_phase = "accepted"
    elif holds:
        active_phase = "unconfirmed"
    elif echoed:
        active_phase = "proxy_echo"
    else:
        active_phase = "idle"

    transactions_raw = journal.get("transitions") if isinstance(journal.get("transitions"), dict) else {}
    active_transactions = []
    for group_name, batch in transactions_raw.items():
        if not isinstance(batch, dict):
            continue
        ready_at = _num(batch.get("ready_at")) or now
        members_list = []
        for room, member in (batch.get("members") or {}).items():
            if not isinstance(member, dict):
                continue
            off_until = _num(member.get("off_until"))
            members_list.append({
                "room": room,
                "phase": member.get("phase"),
                "target": _num(member.get("target")),
                "initial": _num(member.get("initial")),
                "initial_invalidated": bool(member.get("initial_invalidated")),
                "off_echo_at": _num(member.get("off_echo_at")),
                "off_until": off_until,
                "off_remaining_s": round(max(0.0, off_until - now), 1) if off_until is not None else 0.0,
                "control": member.get("control") if isinstance(member.get("control"), dict) else None,
            })
        active_transactions.append({
            "group": group_name,
            "id": batch.get("id"),
            "mode": batch.get("mode"),
            "ready_at": ready_at,
            "ready_in_s": round(max(0.0, ready_at - now), 1),
            "members": members_list,
        })

    parked_list = [
        {"room": a["room"], "room_name": a["name"], **a["parked_info"]}
        for a in actuators
        if a["parked_info"] is not None
    ]
    return {
        "active_phase": active_phase,
        "pending_count": len(pending),
        "unconfirmed_count": len(unconfirmed),
        "echoed_24h_count": len(echoed),
        "active_transactions": active_transactions,
        "locks": {
            "holds": holds,
            "manual_holds": manual_holds,
            "isolated_rooms": list(permit_info.get("isolated_rooms") or []),
            "paused_zones": list(permit_info.get("paused_zones") or []),
        },
        "parked": parked_list,
    }


def _make_action_payload(
    room: str,
    kind: str,
    to_val: Any,
    purpose: str,
    rooms_cfg: dict[str, Any],
    actuators_by_room: dict[str, dict[str, Any]],
    zone: str | None = None,
    before_override: Any = None,
) -> dict[str, Any]:
    act = actuators_by_room.get(room, {})
    room_name = _room_display_name(room, rooms_cfg)
    current_mode = act.get("mode")
    before_val = before_override if before_override is not None else (
        act.get("setpoint") if kind == "temperature" else current_mode
    )
    return {
        "room": room,
        "room_name": room_name,
        "zone": zone or act.get("served_zone") or act.get("zone"),
        "kind": kind,
        "purpose": purpose,
        "purpose_label": PURPOSE_LABELS.get(purpose, purpose or "—"),
        "from_value": before_val,
        "to_value": to_val,
        "current_mode": current_mode,
        "display": _format_action_display(room_name, kind, before_val, to_val),
        "service_call": _service_call_text(kind, to_val),
    }


def _gate_blockers_for_candidate(
    action_tuple: list[Any],
    rejection_reason: str | None,
    actuators_by_room: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    now: float,
) -> tuple[list[dict[str, Any]], float | None]:
    room, kind, value, purpose = action_tuple[0], action_tuple[1], action_tuple[2], action_tuple[3]
    act = actuators_by_room.get(room, {})
    blockers: list[dict[str, Any]] = []
    max_rem = 0.0
    has_timer = False

    if rejection_reason:
        blockers.append({
            "code": "final_gate",
            "label": "最终门禁拦截原因",
            "detail": rejection_reason,
            "met": False,
            "remaining_s": None,
            "progress_pct": 0,
        })

    power_age = act.get("power_age_s")
    power_ok = power_age is not None and 0 <= power_age <= POWER_REPORT_GAP_SECONDS
    blockers.append({
        "code": "power_fresh",
        "label": "本机实时功率证据 (≤120秒)",
        "detail": (
            f"{int(round(power_age))}秒前上报 {act['power_w']:g} W"
            if (power_age is not None and act.get("power_w") is not None)
            else "等待有效实时功率上报"
        ),
        "met": power_ok,
        "remaining_s": 0 if power_ok else None,
        "progress_pct": 100 if power_ok else 0,
    })

    if kind == "mode" and value in ACTIVE_MODES and act.get("parked_info"):
        parked = act["parked_info"]
        min_off_req = float(policy.get("min_off_seconds", 300))
        min_off_rem = float(parked["min_off_remaining_s"])
        min_off_el = float(parked["off_elapsed_s"])
        blockers.append({
            "code": "min_off_time",
            "label": f"最短停机时间 ({_format_duration_zh(min_off_req)})",
            "detail": f"已停机 {_format_duration_zh(min_off_el)} / 需 {_format_duration_zh(min_off_req)}",
            "met": min_off_rem <= 0,
            "remaining_s": round(min_off_rem, 1),
            "progress_pct": min(100, int(round(100 * min_off_el / min_off_req))) if min_off_req else 100,
        })
        if min_off_rem > 0:
            max_rem = max(max_rem, min_off_rem)
            has_timer = True

        low_band = (act.get("power_bands") or {}).get("low") or {}
        low_sec = _num(low_band.get("seconds")) or 0.0
        low_rem = max(0.0, min_off_req - low_sec)
        blockers.append({
            "code": "low_power_proof",
            "label": f"连续停机低耗实测 ({_format_duration_zh(min_off_req)})",
            "detail": f"连续低耗 {_format_duration_zh(low_sec)} / 需 {_format_duration_zh(min_off_req)}",
            "met": low_rem <= 0,
            "remaining_s": round(low_rem, 1),
            "progress_pct": min(100, int(round(100 * low_sec / min_off_req))) if min_off_req else 100,
        })
        if low_rem > 0:
            max_rem = max(max_rem, low_rem)
            has_timer = True

        if parked.get("mode") and parked["mode"] != value:
            rev_req = float(policy.get("reversal_run_seconds", 3600))
            rev_rem = float(parked["reversal_remaining_s"])
            rev_el = float(parked["reversal_elapsed_s"])
            blockers.append({
                "code": "reversal_guard",
                "label": f"冷热换向保护 ({_format_duration_zh(rev_req)})",
                "detail": f"距上次 {MODE_LABELS.get(parked['mode'], parked['mode'])} 启机 {_format_duration_zh(rev_el)} / 需 {_format_duration_zh(rev_req)}",
                "met": rev_rem <= 0,
                "remaining_s": round(rev_rem, 1),
                "progress_pct": min(100, int(round(100 * rev_el / rev_req))) if rev_req else 100,
            })
            if rev_rem > 0:
                max_rem = max(max_rem, rev_rem)
                has_timer = True

    return blockers, (round(max_rem, 1) if has_timer else None)


def _predict_zone_action(
    zone: dict[str, Any],
    actuators: list[dict[str, Any]],
    actuators_by_room: dict[str, dict[str, Any]],
    rooms_cfg: dict[str, Any],
    closed_loop: dict[str, Any],
    policy: dict[str, Any],
    elected: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    direction = zone.get("direction")
    if direction not in ACTIVE_MODES:
        return None, None
    feasible = zone.get("feasible_actuators") or []
    candidates_rooms = list(feasible)
    if not candidates_rooms:
        # Even before qualification, check ready actuators belonging to this zone
        candidates_rooms = [
            a["room"]
            for a in actuators
            if a["zone"] == zone["zone"] and a["ready"] and not a["isolated"]
        ]
    if not candidates_rooms:
        return None, None

    effects_list = list((closed_loop.get("effects") or {}).values()) if isinstance(closed_loop.get("effects"), dict) else []

    def _peer_capable(room_name: str) -> bool:
        act_obj = actuators_by_room[room_name]
        if act_obj["mode"] != direction or act_obj["setpoint"] is None:
            return False
        lim = (rooms_cfg.get(room_name) or {}).get("soft_limits", {}).get(direction)
        space = _automatic_setpoint(act_obj["setpoint"], DIRECTION[direction], lim)
        if space is None:
            return False
        last_eff = next(
            (
                v
                for v in reversed(effects_list)
                if isinstance(v, dict)
                and v.get("unit") == room_name
                and v.get("zone") == zone["zone"]
                and _num(v.get("target")) == zone.get("target_c")
            ),
            None,
        )
        return last_eff is None or bool(last_eff.get("useful"))

    # Match zone_control.py ranking: home zone, energy score (Wh/°C), priority, -influence.
    def _rank_key(room_name: str) -> tuple[Any, ...]:
        act_obj = actuators_by_room[room_name]
        score = act_obj.get("energy_score_wh_per_c")
        influence = _num((act_obj.get("influences") or {}).get(zone["zone"])) or 0.0
        return (
            act_obj["zone"] != zone["zone"],
            round(score, 4) if score is not None else float("inf"),
            act_obj["priority"],
            -influence,
            room_name,
        )

    if elected and elected in candidates_rooms:
        ordered = [elected]
    else:
        ordered = sorted(candidates_rooms, key=_rank_key)
        # Mirror zone_control.py's capable() guard: prefer an active capable same-mode
        # peer over starting an off/parked unit.
        active_capable = [r for r in ordered if _peer_capable(r)]
        if active_capable:
            ordered = active_capable + [r for r in ordered if r not in active_capable and actuators_by_room[r]["mode"] != "off"]

    step = float(policy.get("ordinary_step", 1))
    large_dev = float(policy.get("large_deviation", 2.0))
    large_step = float(policy.get("large_step", 2))
    first_act = actuators_by_room[ordered[0]]

    for chosen_room in ordered:
        act = actuators_by_room[chosen_room]
        mode = act["mode"]
        limits = (rooms_cfg.get(chosen_room) or {}).get("soft_limits", {}).get(direction)

        if mode == "off" and act["parked"]:
            if any(_peer_capable(peer) for peer in candidates_rooms if peer != chosen_room):
                continue
            action = _make_action_payload(
                chosen_room, "mode", direction, "energy_start", rooms_cfg, actuators_by_room, zone=zone["zone"]
            )
            return action, act
        if mode == direction and act["setpoint"] is not None:
            mag = step
            trend = zone.get("trend")
            err = abs(zone["error_c"]) if zone.get("error_c") is not None else 0.0
            if (
                not act["risks"]
                and act["zone"] == zone["zone"]
                and err >= large_dev
                and trend is not None
                and DIRECTION[direction] * trend["delta_10m_c"] <= 0
            ):
                mag = large_step
            proposed = _automatic_setpoint(act["setpoint"], DIRECTION[direction] * mag, limits)
            if proposed is not None:
                action = _make_action_payload(
                    chosen_room, "temperature", proposed, "zone_strengthen", rooms_cfg, actuators_by_room, zone=zone["zone"]
                )
                return action, act
    return None, first_act


def _infer_next_command(
    *,
    control: dict[str, Any],
    zone_states: list[dict[str, Any]],
    handoff: dict[str, Any],
    ledger: dict[str, Any],
    actuators: list[dict[str, Any]],
    commands: list[dict[str, Any]],
    permit_info: dict[str, Any],
    journal: dict[str, Any],
    rooms_cfg: dict[str, Any],
    policy: dict[str, Any],
    next_cycle_in_s: float | None,
    now: float,
) -> dict[str, Any]:
    actuators_by_room = {a["room"]: a for a in actuators}
    zones_by_key = {z["zone"]: z for z in zone_states}
    closed_loop = journal.get("closed_loop") if isinstance(journal.get("closed_loop"), dict) else {}

    raw_candidates = control.get("candidates") if isinstance(control.get("candidates"), list) else []
    raw_rejections = control.get("rejections") if isinstance(control.get("rejections"), list) else []
    rejection_map: dict[tuple[Any, ...], str] = {}
    for entry in raw_rejections:
        if isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], list):
            rejection_map[tuple(entry[0])] = str(entry[1])

    formatted_candidates = []
    for item in raw_candidates:
        if not isinstance(item, list) or len(item) < 4:
            continue
        room, kind, val, purpose = item[0], item[1], item[2], item[3]
        rej = rejection_map.get(tuple(item))
        room_name = _room_display_name(str(room), rooms_cfg)
        act = actuators_by_room.get(str(room), {})
        before_val = act.get("setpoint") if kind == "temperature" else act.get("mode")
        formatted_candidates.append({
            "room": room,
            "room_name": room_name,
            "kind": kind,
            "value": val,
            "purpose": purpose,
            "purpose_label": PURPOSE_LABELS.get(str(purpose), str(purpose)),
            "display": _format_action_display(room_name, str(kind), before_val, val),
            "rejected": rej is not None,
            "rejection_reason": rej,
        })

    # 1. In-flight command awaiting proxy echo
    pending_cmds = [c for c in commands if c["status"] in {"prepared", "accepted"}]
    if pending_cmds:
        cmd = pending_cmds[-1]
        elapsed = cmd["age_s"] if cmd["age_s"] is not None else 0.0
        rem_timeout = max(0.0, ECHO_TIMEOUT_SECONDS - elapsed)
        action = _make_action_payload(
            cmd["room"],
            cmd["kind"],
            cmd["value"],
            cmd["purpose"],
            rooms_cfg,
            actuators_by_room,
            zone=(cmd.get("control") or {}).get("zone"),
            before_override=cmd["before_setpoint"] if cmd["kind"] == "temperature" else cmd["before_mode"],
        )
        in_flight = {
            "room": cmd["room"],
            "room_name": cmd["room_name"],
            "kind": cmd["kind"],
            "before": cmd["before"],
            "value": cmd["value"],
            "purpose": cmd["purpose"],
            "purpose_label": cmd["purpose_label"],
            "status": cmd["status"],
            "status_label": cmd["status_label"],
            "sent_at": cmd["sent_at"],
            "elapsed_s": round(elapsed, 1),
            "timeout_s": ECHO_TIMEOUT_SECONDS,
            "timeout_remaining_s": round(rem_timeout, 1),
            "expected": cmd["expected"],
        }
        return {
            "stage": "in_flight",
            "stage_label": "在途指令 · 等待伴侣回报",
            "headline": f"等待回报：{action['display']}",
            "summary": f"已于 {_format_duration_zh(elapsed)} 前下发（{cmd['status_label']}），在伴侣确认回报前暂停所有新发令。",
            "confidence": "confirmed",
            "action": action,
            "eta_seconds": round(rem_timeout, 1),
            "eta_note": f"回报超时窗口剩余 {_format_duration_zh(rem_timeout)}（每 20 秒检查一次）",
            "waiting_for_new_report": False,
            "blockers": [
                {
                    "code": "proxy_echo",
                    "label": "伴侣状态回显确认",
                    "detail": f"期望回报 {cmd['expected']} · 已等待 {_format_duration_zh(elapsed)} / 上限 3分钟",
                    "met": False,
                    "remaining_s": round(rem_timeout, 1),
                    "progress_pct": min(100, int(round(100 * elapsed / ECHO_TIMEOUT_SECONDS))),
                }
            ],
            "candidates": formatted_candidates,
            "rejections": [{"action": k, "reason": v} for k, v in rejection_map.items()],
            "in_flight": in_flight,
        }

    # 2. Permit inactive — continuous_pilot.cycle() exits before advancing unsent
    # transactions or dispatching any new command when read_permit() is inactive.
    # (Already-sent in-flight commands above are still displayed while awaiting echo.)
    if not permit_info.get("active"):
        has_queued_tx = any(
            m.get("phase") == "initial"
            for tx in ledger["active_transactions"]
            if tx["group"] not in ledger["locks"]["holds"]
            for m in tx.get("members") or ()
        )
        summary = permit_info.get("reason") or "控制许可未开启或已过期，控制器处于只读暂停状态。"
        if has_queued_tx:
            summary += "（账本中存在待续步的开机初值事务，但在许可重新生效前不会下发。）"
        return {
            "stage": "paused_or_locked",
            "stage_label": "已暂停 · 许可未生效",
            "headline": "当前不会下发任何指令",
            "summary": summary,
            "confidence": "none",
            "action": None,
            "eta_seconds": None,
            "eta_note": "需先启用有效控制许可",
            "waiting_for_new_report": False,
            "blockers": [
                {
                    "code": "permit_inactive",
                    "label": "实机控制许可",
                    "detail": permit_info.get("reason") or "未启用",
                    "met": False,
                    "remaining_s": None,
                    "progress_pct": 0,
                }
            ],
            "candidates": formatted_candidates,
            "rejections": [{"action": k, "reason": v} for k, v in rejection_map.items()],
            "in_flight": None,
        }

    # 3. Selected command in current cycle
    selected = control.get("selected")
    if isinstance(selected, list) and len(selected) >= 4:
        action = _make_action_payload(
            str(selected[0]), str(selected[1]), selected[2], str(selected[3]), rooms_cfg, actuators_by_room
        )
        return {
            "stage": "selected",
            "stage_label": "本轮已选定指令",
            "headline": f"本轮执行：{action['display']}",
            "summary": f"用途：{action['purpose_label']}。已通过全部区域分配与最终门禁核验。",
            "confidence": "confirmed",
            "action": action,
            "eta_seconds": 0.0,
            "eta_note": "本轮已提交发令",
            "waiting_for_new_report": False,
            "blockers": [],
            "candidates": formatted_candidates,
            "rejections": [{"action": k, "reason": v} for k, v in rejection_map.items()],
            "in_flight": None,
        }

    # 4. Queued multi-step transaction (e.g. post-start initial setpoint / safety easing)
    for tx in ledger["active_transactions"]:
        if tx["group"] in ledger["locks"]["holds"]:
            continue
        for member in tx["members"]:
            if member["phase"] == "initial":
                room = member["room"]
                act = actuators_by_room.get(room, {})
                mode = str(tx["mode"] or act.get("mode") or "")
                limits = (rooms_cfg.get(room) or {}).get("soft_limits", {}).get(mode)
                target = member["target"] if member["target"] is not None else act.get("target_c")
                safety = _initial_safety_setting(mode, act.get("setpoint"), target, limits)
                if safety is not None:
                    purpose = "initial_safety_easing"
                    next_val = safety
                else:
                    purpose = "reversal_initial"
                    zone_key = (member.get("control") or {}).get("zone") or act.get("zone")
                    zone = zones_by_key.get(str(zone_key), {})
                    next_val = (
                        member["initial"]
                        if member["initial"] is not None
                        else _learned_initial(
                            room,
                            mode,
                            {"target": target, "source_version": zone.get("source_version")},
                            closed_loop,
                            limits,
                            policy,
                        )
                    )
                if next_val is not None:
                    action = _make_action_payload(room, "temperature", next_val, purpose, rooms_cfg, actuators_by_room)
                    rem = float(tx["ready_in_s"])
                    return {
                        "stage": "queued_transaction",
                        "stage_label": "事务续步 · 待补发初值",
                        "headline": f"下步事务：{action['display']}",
                        "summary": f"已确认进入 {MODE_LABELS.get(mode, mode)} 模式，下一轮将自动补发{PURPOSE_LABELS.get(purpose, purpose)}。",
                        "confidence": "confirmed",
                        "action": action,
                        "eta_seconds": round(max(rem, next_cycle_in_s or 0.0), 1),
                        "eta_note": (
                            f"批次间隔剩余 {_format_duration_zh(rem)}"
                            if rem > 0
                            else "下一轮 20 秒巡检即可补发"
                        ),
                        "waiting_for_new_report": False,
                        "blockers": [
                            {
                                "code": "batch_spacing",
                                "label": "同组串行发令间隔",
                                "detail": f"剩余 {_format_duration_zh(rem)}" if rem > 0 else "已满足",
                                "met": rem <= 0,
                                "remaining_s": round(rem, 1),
                                "progress_pct": 100 if rem <= 0 else 40,
                            }
                        ],
                        "candidates": formatted_candidates,
                        "rejections": [{"action": k, "reason": v} for k, v in rejection_map.items()],
                        "in_flight": None,
                    }

    # 5. Candidate generated but blocked by final_gate
    if raw_candidates and raw_rejections:
        first = raw_candidates[0]
        rej_reason = rejection_map.get(tuple(first)) or (str(raw_rejections[0][1]) if raw_rejections else None)
        action = _make_action_payload(
            str(first[0]), str(first[1]), first[2], str(first[3]), rooms_cfg, actuators_by_room
        )
        blockers, eta = _gate_blockers_for_candidate(first, rej_reason, actuators_by_room, policy, now)
        return {
            "stage": "gate_blocked",
            "stage_label": "候选已生成 · 门禁拦截",
            "headline": f"待放行：{action['display']}",
            "summary": f"控制器已生成候选指令（{action['purpose_label']}），因「{rej_reason}」暂未下发。",
            "confidence": "candidate",
            "action": action,
            "eta_seconds": eta,
            "eta_note": (
                f"预计最快 {_format_duration_zh(eta)} 后满足保护门禁"
                if eta is not None
                else "待阻塞条件解除后下一轮 20 秒巡检放行"
            ),
            "waiting_for_new_report": "功率" in (rej_reason or "") or "上报" in (rej_reason or ""),
            "blockers": blockers,
            "candidates": formatted_candidates,
            "rejections": [{"action": k, "reason": v} for k, v in rejection_map.items()],
            "in_flight": None,
        }

    # 6. Active bidirectional handoff waiting phase
    if handoff.get("active") and handoff.get("phase") not in {"idle", "serve"}:
        phase = handoff["phase"]
        blocker_room = handoff.get("blocker")
        receiver_room = handoff.get("receiver")
        owner_key = handoff.get("owner")
        blockers = []
        action = None
        eta = None
        waiting_report = False
        red_wait = float(policy.get("reduction_wait_seconds", 180))
        low_obs_wait = float(policy.get("lower_output_observe_seconds", 180))

        handoff_selected_at = _num(handoff.get("selected_at"))

        def _matches_current_handoff(cmd: dict[str, Any], purpose_name: str) -> bool:
            if cmd["room"] != blocker_room or cmd["purpose"] != purpose_name or cmd["status"] != "proxy_echo":
                return False
            if handoff_selected_at is None:
                return True
            cmd_ctrl = cmd.get("control")
            cmd_handoff_at = _num(cmd_ctrl.get("handoff_at")) if isinstance(cmd_ctrl, dict) else None
            return cmd_handoff_at is not None and abs(cmd_handoff_at - handoff_selected_at) <= 1e-6

        if phase == "verify_reduction" and blocker_room:
            action = _make_action_payload(
                blocker_room, "mode", "off", "handoff_off", rooms_cfg, actuators_by_room
            )
            # `commands` is already sorted newest-first; match the current handoff's selected_at.
            last_red = next((c for c in commands if _matches_current_handoff(c, "handoff_reduce")), None)
            if last_red and last_red["proxy_at"] is not None:
                elapsed = max(0.0, now - last_red["proxy_at"])
                rem = max(0.0, red_wait - elapsed)
                eta = round(rem, 1)
                blockers.append({
                    "code": "reduction_wait",
                    "label": f"减载后观察期 ({_format_duration_zh(red_wait)})",
                    "detail": f"已观察 {_format_duration_zh(elapsed)} / 需 {_format_duration_zh(red_wait)}",
                    "met": rem <= 0,
                    "remaining_s": round(rem, 1),
                    "progress_pct": min(100, int(round(100 * elapsed / red_wait))),
                })
                waiting_report = True
                blockers.append({
                    "code": "both_zone_reports",
                    "label": "双区独立新测温反馈",
                    "detail": "需优先区与对侧区域在减载回显后均产生新测温上报",
                    "met": False,
                    "remaining_s": None,
                    "progress_pct": 0,
                })
        elif phase == "verify_exit" and receiver_room and owner_key:
            owner_zone = zones_by_key.get(owner_key)
            if owner_zone:
                action, _ = _predict_zone_action(
                    owner_zone, actuators, actuators_by_room, rooms_cfg, closed_loop, policy, receiver_room
                )
            last_off = next((c for c in commands if _matches_current_handoff(c, "handoff_off")), None)
            if last_off and last_off["proxy_at"] is not None:
                elapsed = max(0.0, now - last_off["proxy_at"])
                rem = max(0.0, low_obs_wait - elapsed)
                eta = round(rem, 1)
                blockers.append({
                    "code": "exit_low_power",
                    "label": f"停机后连续低耗与双区反馈 ({_format_duration_zh(low_obs_wait)})",
                    "detail": f"已等待 {_format_duration_zh(elapsed)} / 需 {_format_duration_zh(low_obs_wait)}",
                    "met": rem <= 0,
                    "remaining_s": round(rem, 1),
                    "progress_pct": min(100, int(round(100 * elapsed / low_obs_wait))),
                })
                waiting_report = True
        elif phase == "reduce" and blocker_room:
            act = actuators_by_room.get(blocker_room, {})
            b_mode = act.get("mode")
            if b_mode in ACTIVE_MODES and act.get("setpoint") is not None:
                limits = (rooms_cfg.get(blocker_room) or {}).get("soft_limits", {}).get(b_mode)
                step = _automatic_setpoint(
                    act["setpoint"], -DIRECTION[b_mode] * float(policy.get("ordinary_step", 1)), limits
                )
                if step is not None:
                    action = _make_action_payload(
                        blocker_room, "temperature", step, "handoff_reduce", rooms_cfg, actuators_by_room
                    )
                else:
                    action = _make_action_payload(
                        blocker_room, "mode", "off", "handoff_off", rooms_cfg, actuators_by_room
                    )
        elif phase == "exit" and blocker_room:
            action = _make_action_payload(
                blocker_room, "mode", "off", "handoff_off", rooms_cfg, actuators_by_room
            )

        headline = f"交接阶段 [{handoff['phase_label']}]" + (f"：{action['display']}" if action else "")
        return {
            "stage": "handoff_waiting",
            "stage_label": f"双向交接 · {handoff['phase_label']}",
            "headline": headline,
            "summary": handoff.get("reason_label") or "双区反向需求交接中，按安全顺序先退载/停机再服务新优先区。",
            "confidence": "projected" if action else "none",
            "action": action,
            "eta_seconds": eta,
            "eta_note": (
                f"交接观察期剩余约 {_format_duration_zh(eta)}，并需双区新测温"
                if eta is not None
                else handoff.get("reason_label") or "等待交接条件满足"
            ),
            "waiting_for_new_report": waiting_report,
            "blockers": blockers,
            "candidates": formatted_candidates,
            "rejections": [{"action": k, "reason": v} for k, v in rejection_map.items()],
            "in_flight": None,
        }

    # 7. Active locks preventing any feasible action on demanded zone
    holds = ledger["locks"]["holds"]
    manual_holds = ledger["locks"]["manual_holds"]
    demanded_zones = [z for z in zone_states if z["qualified"] or z["recovery_active"]]
    if (holds or manual_holds) and demanded_zones and not any(z["feasible_actuators"] for z in demanded_zones):
        lock_items = [
            {
                "code": f"hold_{grp}",
                "label": f"故障/回报锁 ({grp})",
                "detail": str(msg),
                "met": False,
                "remaining_s": None,
                "progress_pct": 0,
            }
            for grp, msg in holds.items()
        ] + [
            {
                "code": f"manual_{room}",
                "label": f"人工修改暂停 ({_room_display_name(room, rooms_cfg)})",
                "detail": str(msg),
                "met": False,
                "remaining_s": None,
                "progress_pct": 0,
            }
            for room, msg in manual_holds.items()
        ]
        return {
            "stage": "paused_or_locked",
            "stage_label": "存在运行锁 · 暂缓发令",
            "headline": "区域有需求，但可控设备已锁定或被人工接管",
            "summary": "需人工确认空调状态并解除对应故障锁或手动暂停锁后，控制器才会恢复发令。",
            "confidence": "none",
            "action": None,
            "eta_seconds": None,
            "eta_note": "等待人工复核解锁",
            "waiting_for_new_report": False,
            "blockers": lock_items,
            "candidates": formatted_candidates,
            "rejections": [{"action": k, "reason": v} for k, v in rejection_map.items()],
            "in_flight": None,
        }

    # 8. Projected next action: qualified owner waiting feedback/observation OR weakening OR approaching dwell
    owner_key = control.get("owner")
    owner_zone = zones_by_key.get(owner_key) if owner_key else None
    feedback_wait = float(policy.get("feedback_wait_seconds", 600))
    reduction_wait = float(policy.get("reduction_wait_seconds", 180))

    if owner_zone and owner_zone["qualified"]:
        action, chosen_act = _predict_zone_action(
            owner_zone, actuators, actuators_by_room, rooms_cfg, closed_loop, policy, control.get("elected")
        )
        blockers = []
        max_rem = 0.0
        has_timer = False
        waiting_report = False

        if chosen_act is not None:
            obs = chosen_act.get("observations_after") or {}
            last_inc = _num(obs.get("last_increase"))
            own_cmd = _num(obs.get("own_command"))
            zone_rep = _num(obs.get("zone_report")) or owner_zone.get("last_report_at")
            anchor = max((t for t in (last_inc, own_cmd) if t is not None), default=None)
            if anchor is not None:
                elapsed = max(0.0, now - anchor)
                rem = max(0.0, feedback_wait - elapsed)
                blockers.append({
                    "code": "feedback_wait",
                    "label": f"动作后反馈观察期 ({_format_duration_zh(feedback_wait)})",
                    "detail": f"距上次动作回显 {_format_duration_zh(elapsed)} / 需满 {_format_duration_zh(feedback_wait)}",
                    "met": rem <= 0,
                    "remaining_s": round(rem, 1),
                    "progress_pct": min(100, int(round(100 * elapsed / feedback_wait))),
                })
                if rem > 0:
                    max_rem = max(max_rem, rem)
                    has_timer = True

                got_new_report = zone_rep is not None and zone_rep > anchor
                if not got_new_report:
                    waiting_report = True
                blockers.append({
                    "code": "new_zone_report",
                    "label": "动作后的独立新室温上报",
                    "detail": "已收到动作后新测温" if got_new_report else "需收到晚于上次动作回显的新测温上报",
                    "met": got_new_report,
                    "remaining_s": None,
                    "progress_pct": 100 if got_new_report else 0,
                })

            if chosen_act["parked"] and chosen_act["parked_info"]:
                parked = chosen_act["parked_info"]
                if parked["min_off_remaining_s"] > 0:
                    rem = float(parked["min_off_remaining_s"])
                    max_rem = max(max_rem, rem)
                    has_timer = True
                    blockers.append({
                        "code": "min_off",
                        "label": "最短停机保护",
                        "detail": f"剩余 {_format_duration_zh(rem)}",
                        "met": False,
                        "remaining_s": round(rem, 1),
                        "progress_pct": min(100, int(round(100 * parked["off_elapsed_s"] / parked["min_off_required_s"]))),
                    })
                if parked["mode"] and parked["mode"] != owner_zone.get("direction") and parked["reversal_remaining_s"] > 0:
                    rem = float(parked["reversal_remaining_s"])
                    max_rem = max(max_rem, rem)
                    has_timer = True
                    blockers.append({
                        "code": "reversal_run",
                        "label": "冷热换向保护",
                        "detail": f"剩余 {_format_duration_zh(rem)}",
                        "met": False,
                        "remaining_s": round(rem, 1),
                        "progress_pct": min(100, int(round(100 * parked["reversal_elapsed_s"] / parked["reversal_required_s"]))),
                    })

        if action is not None:
            eta = round(max_rem, 1) if has_timer else (round(next_cycle_in_s, 1) if next_cycle_in_s is not None else None)
            eta_note = (
                f"预计最快 {_format_duration_zh(max_rem)} 后"
                + ("（且需收到新测温上报）" if waiting_report else "进入下一轮评估发令")
                if has_timer
                else ("等待收到动作后的新独立室温上报" if waiting_report else "条件已接近就绪，下一轮巡检评估")
            )
            delta_str = f"{owner_zone['signed_delta_c']:+.2f}℃" if owner_zone.get("signed_delta_c") is not None else "—"
            return {
                "stage": "projected",
                "stage_label": "预计下条指令 · 等待响应窗口",
                "headline": f"预计下发：{action['display']}",
                "summary": f"{owner_zone['label']}已确认{owner_zone['direction_label']}需求（偏差 {delta_str}），满足观察条件后预计执行该指令。",
                "confidence": "projected",
                "action": action,
                "eta_seconds": eta,
                "eta_note": eta_note,
                "waiting_for_new_report": waiting_report,
                "blockers": blockers,
                "candidates": formatted_candidates,
                "rejections": [{"action": k, "reason": v} for k, v in rejection_map.items()],
                "in_flight": None,
            }

    # 8b. Weakening zone waiting reduction_wait_seconds
    weaken_zones = [z for z in zone_states if z["state"] == "weaken"]
    if weaken_zones:
        wz = weaken_zones[0]
        local_active = [a for a in actuators if a["zone"] == wz["zone"] and a["mode"] in ACTIVE_MODES and a["ready"]]
        action = None
        blockers = []
        eta = None
        waiting_report = False
        if local_active:
            act = local_active[0]
            limits = (rooms_cfg.get(act["room"]) or {}).get("soft_limits", {}).get(act["mode"])
            proposed = _automatic_setpoint(
                act["setpoint"], -DIRECTION[act["mode"]] * float(policy.get("ordinary_step", 1)), limits
            )
            if proposed is not None:
                action = _make_action_payload(
                    act["room"], "temperature", proposed, "zone_reduce", rooms_cfg, actuators_by_room, zone=wz["zone"]
                )
            else:
                action = _make_action_payload(
                    act["room"], "mode", "off", "energy_off", rooms_cfg, actuators_by_room, zone=wz["zone"]
                )
            own_cmd = _num((act.get("observations_after") or {}).get("own_command"))
            if own_cmd is not None:
                elapsed = max(0.0, now - own_cmd)
                rem = max(0.0, reduction_wait - elapsed)
                eta = round(rem, 1)
                waiting_report = (wz.get("last_report_at") or 0) <= own_cmd
                blockers.append({
                    "code": "reduction_wait",
                    "label": f"减载后等待 ({_format_duration_zh(reduction_wait)})",
                    "detail": f"已过 {_format_duration_zh(elapsed)} / 需 {_format_duration_zh(reduction_wait)}",
                    "met": rem <= 0,
                    "remaining_s": round(rem, 1),
                    "progress_pct": min(100, int(round(100 * elapsed / reduction_wait))),
                })
        return {
            "stage": "projected",
            "stage_label": "预计减载指令",
            "headline": f"预计减载：{action['display']}" if action else f"{wz['label']} 处于减载阶段",
            "summary": f"{wz['label']}触发过冲或风险减载，等待上次动作反馈后继续减载。",
            "confidence": "projected" if action else "none",
            "action": action,
            "eta_seconds": eta,
            "eta_note": (
                f"预计 {_format_duration_zh(eta)} 后评估下一步减载"
                if eta is not None
                else "等待新的本区测温反馈"
            ),
            "waiting_for_new_report": waiting_report,
            "blockers": blockers,
            "candidates": formatted_candidates,
            "rejections": [{"action": k, "reason": v} for k, v in rejection_map.items()],
            "in_flight": None,
        }

    # 8c. Zone accumulating dwell & independent reports toward Recover
    approaching = [
        z
        for z in zone_states
        if not z["paused"]
        and z["quality"] == "fresh"
        and z["direction"] in ACTIVE_MODES
        and z["reason"] == "waiting_reports_or_dwell"
    ]
    if approaching:
        approaching.sort(key=lambda z: (-abs(z["error_c"] or 0.0), -z["dwell"]["duration_s"]))
        az = approaching[0]
        action, _ = _predict_zone_action(
            az, actuators, actuators_by_room, rooms_cfg, closed_loop, policy, None
        )
        dwell_rem = float(az["dwell"]["remaining_s"])
        span_rem = float(az["independent_reports"]["remaining_span_s"])
        eta = round(max(dwell_rem, span_rem), 1)
        need_more_reports = az["independent_reports"]["count"] < 2 or span_rem > 0
        blockers = [
            {
                "code": "tracking_dwell",
                "label": f"持续偏差时长 ({_format_duration_zh(az['dwell']['required_s'])})",
                "detail": f"已持续 {_format_duration_zh(az['dwell']['duration_s'])} / 需 {_format_duration_zh(az['dwell']['required_s'])}",
                "met": dwell_rem <= 0,
                "remaining_s": round(dwell_rem, 1),
                "progress_pct": az["dwell"]["pct"],
            },
            {
                "code": "independent_reports",
                "label": f"跨时段独立测温上报 (≥2次且跨度≥{_format_duration_zh(az['independent_reports']['required_span_s'])})",
                "detail": f"当前 {az['independent_reports']['count']} 次，跨度 {_format_duration_zh(az['independent_reports']['span_s'])}",
                "met": not need_more_reports,
                "remaining_s": round(span_rem, 1),
                "progress_pct": az["independent_reports"]["pct"],
            },
        ]
        meas_str = f"{az['measured_c']:.2f}℃" if az.get("measured_c") is not None else "—"
        tgt_str = f"{az['target_c']:.1f}℃" if az.get("target_c") is not None else "—"
        dlt_str = f"{az['signed_delta_c']:+.2f}℃" if az.get("signed_delta_c") is not None else "—"
        return {
            "stage": "projected",
            "stage_label": "偏差累计中 · 预计下条指令",
            "headline": (
                f"资格达成后预计：{action['display']}"
                if action
                else f"{az['label']} 偏差累计中（{az['direction_label']}）"
            ),
            "summary": (
                f"{az['label']}当前 {meas_str}（目标 {tgt_str}，偏差 {dlt_str}），"
                f"需持续满 {_format_duration_zh(az['dwell']['required_s'])} 且具备跨 "
                f"{_format_duration_zh(az['independent_reports']['required_span_s'])} 的独立测温上报才会转入恢复状态。"
            ),
            "confidence": "projected" if action else "none",
            "action": action,
            "eta_seconds": eta,
            "eta_note": f"若偏差持续不回归，最快约 {_format_duration_zh(eta)} 后进入恢复并触发指令",
            "waiting_for_new_report": need_more_reports,
            "blockers": blockers,
            "candidates": formatted_candidates,
            "rejections": [{"action": k, "reason": v} for k, v in rejection_map.items()],
            "in_flight": None,
        }

    # 8d. Zone observing credible return or verifying stable recovery
    observing = [z for z in zone_states if z["state"] == "observe" and z["reason"] in {"returning", "verify_stable_recovery"}]
    if observing:
        oz = observing[0]
        if oz["reason"] == "returning":
            return {
                "stage": "idle_maintain",
                "stage_label": "自然回归中 · 暂缓发令",
                "headline": f"{oz['label']}温度正在可信回归，暂不发令",
                "summary": (
                    f"10 分钟预测变化 {oz['trend']['delta_10m_c']:+.2f}℃（±{oz['trend']['uncertainty_c']:.2f}℃），"
                    "满足回归判据，避免过度调节。"
                    if oz.get("trend")
                    else "温度趋势正在回归目标带，暂缓加码。"
                ),
                "confidence": "none",
                "action": None,
                "eta_seconds": round(next_cycle_in_s, 1) if next_cycle_in_s is not None else None,
                "eta_note": "持续观测温度变化，无需干预",
                "waiting_for_new_report": False,
                "blockers": [],
                "candidates": formatted_candidates,
                "rejections": [{"action": k, "reason": v} for k, v in rejection_map.items()],
                "in_flight": None,
            }
        settle = oz["settling"]
        track_band = _num(policy.get("tracking_band")) or 0.3
        settle_req_txt = _format_duration_zh(settle["required_span_s"])
        return {
            "stage": "idle_maintain",
            "stage_label": "稳定确认中 · 暂不加码",
            "headline": f"{oz['label']}已回到目标 ±{track_band:g}℃ 带内，确认稳定中",
            "summary": f"已收到带内独立上报 {settle['reports']}/2 次，跨度 {_format_duration_zh(settle['span_s'])} / {settle_req_txt}。",
            "confidence": "none",
            "action": None,
            "eta_seconds": settle["remaining_span_s"],
            "eta_note": f"稳定确认剩余约 {_format_duration_zh(settle['remaining_span_s'])} 后转入 Maintain",
            "waiting_for_new_report": settle["reports"] < 2 or settle["remaining_span_s"] > 0,
            "blockers": [
                {
                    "code": "recovery_settle",
                    "label": f"回归稳定确认 (≥2次上报且跨度≥{settle_req_txt})",
                    "detail": f"{settle['reports']} 次上报 · 跨度 {_format_duration_zh(settle['span_s'])} / {_format_duration_zh(settle['required_span_s'])}",
                    "met": settle["reports"] >= 2 and settle["remaining_span_s"] <= 0,
                    "remaining_s": settle["remaining_span_s"],
                    "progress_pct": min(100, int(round(100 * settle["span_s"] / settle["required_span_s"]))) if settle["required_span_s"] else 100,
                }
            ],
            "candidates": formatted_candidates,
            "rejections": [{"action": k, "reason": v} for k, v in rejection_map.items()],
            "in_flight": None,
        }

    # 9. Default: steady maintain / hold
    return {
        "stage": "idle_maintain",
        "stage_label": "稳定维持 · 暂无待发指令",
        "headline": "当前无待发指令，全屋维持现有运行设置",
        "summary": "所有监测区域均在目标跟踪带内或无须调整，控制器每 20 秒自动巡检一次。",
        "confidence": "none",
        "action": None,
        "eta_seconds": round(next_cycle_in_s, 1) if next_cycle_in_s is not None else None,
        "eta_note": (
            f"距下一次 20 秒巡检约 {_format_duration_zh(next_cycle_in_s)}"
            if next_cycle_in_s is not None
            else "控制器周期巡检中"
        ),
        "waiting_for_new_report": False,
        "blockers": [],
        "candidates": formatted_candidates,
        "rejections": [{"action": k, "reason": v} for k, v in rejection_map.items()],
        "in_flight": None,
    }


def _summarize_audit_record(entry: dict[str, Any], rooms_cfg: dict[str, Any]) -> dict[str, Any]:
    event = str(entry.get("event") or "unknown")
    outcome = entry.get("outcome")
    assessment = entry.get("assessment") if isinstance(entry.get("assessment"), dict) else {}
    final = entry.get("final")
    proposed = entry.get("proposed")
    summary = ""
    if event == "startup":
        ver = entry.get("observation_version") or "—"
        summary = f"控制器启动 · 观测版本 {ver} · commit {str(entry.get('commit') or 'dev')[:8]}"
    elif event == "cycle":
        if isinstance(final, list) and len(final) >= 4:
            rname = _room_display_name(str(final[0]), rooms_cfg)
            summary = f"执行 {_format_action_display(rname, str(final[1]), None, final[2])}（{PURPOSE_LABELS.get(str(final[3]), final[3])}）"
        elif outcome == "recheck_changed" and isinstance(proposed, list) and len(proposed) >= 4:
            rname = _room_display_name(str(proposed[0]), rooms_cfg)
            summary = f"复核取消 {_format_action_display(rname, str(proposed[1]), None, proposed[2])}"
        else:
            need = assessment.get("need")
            owner = assessment.get("owner")
            cands = assessment.get("candidates") or []
            rejs = assessment.get("rejections") or []
            parts = [OUTCOME_LABELS.get(str(outcome), str(outcome or "cycle"))]
            if need:
                parts.append(f"全屋需求={MODE_LABELS.get(str(need), str(need))}")
            if owner:
                parts.append(f"优先区={owner}")
            if rejs:
                parts.append(f"门禁拦截 {len(rejs)} 项")
            elif cands:
                parts.append(f"候选 {len(cands)} 项")
            summary = " · ".join(parts)
    elif event in {"relay", "receipt"}:
        room = str(entry.get("room") or "")
        st = str(entry.get("status") or "")
        http_code = entry.get("http")
        summary = f"{_room_display_name(room, rooms_cfg)} · {COMMAND_STATUS_LABELS.get(st, st)}" + (
            f" (HTTP {http_code})" if http_code is not None else ""
        )
    elif event.endswith("_error"):
        summary = f"异常 {entry.get('error') or 'Error'}" + (f" ({entry.get('room')})" if entry.get("room") else "")
    else:
        summary = event

    return {
        "at": _num(entry.get("at")),
        "event": event,
        "outcome": outcome,
        "outcome_label": OUTCOME_LABELS.get(str(outcome), str(outcome)) if outcome else event,
        "need": assessment.get("need"),
        "owner": assessment.get("owner"),
        "final": final,
        "candidates_count": len(assessment.get("candidates") or []),
        "rejections_count": len(assessment.get("rejections") or []),
        "error": entry.get("error"),
        "summary": summary,
    }


def _build_shadow_summary(shadow_status: dict[str, Any] | None, shadow_health: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(shadow_status, dict) and not isinstance(shadow_health, dict):
        return {"available": False}
    st = shadow_status if isinstance(shadow_status, dict) else {}
    hl = shadow_health if isinstance(shadow_health, dict) else {}
    return {
        "available": True,
        "health_state": hl.get("state") or ("stale" if st.get("stale") else "healthy"),
        "health_error": hl.get("error") or st.get("source_error"),
        "stale": bool(st.get("stale") or hl.get("state") == "stale"),
        "checked_at": st.get("checked_at"),
        "published_at": st.get("published_at"),
        "policy_revision": st.get("policy_revision"),
        "release_commit": st.get("release_commit"),
        "observed_home_kwh": _num(st.get("observed_home_kwh")),
        "observed_home_minutes": st.get("observed_home_minutes"),
        "observed_home_coverage_pct": _num(st.get("observed_home_coverage_pct")),
        "assessment_completed_minutes": st.get("assessment_completed_minutes"),
        "data_ready_for_24h_review": bool(st.get("data_ready_for_24h_review")),
        "live_actuations": st.get("live_actuations") if isinstance(st.get("live_actuations"), dict) else None,
        "completed_days": (st.get("completed_days") or [])[:9] if isinstance(st.get("completed_days"), list) else [],
    }


def _build_neural_summary(neural_status: dict[str, Any] | None, neural_health: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(neural_status, dict) and not isinstance(neural_health, dict):
        return {"available": False}
    st = neural_status if isinstance(neural_status, dict) else {}
    hl = neural_health if isinstance(neural_health, dict) else {}
    model = st.get("model") if isinstance(st.get("model"), dict) else {}
    counts = st.get("counts") if isinstance(st.get("counts"), dict) else {}
    latest = st.get("latest_sealed") if isinstance(st.get("latest_sealed"), dict) else None
    return {
        "available": True,
        "health_state": hl.get("state") or "unknown",
        "last_error": hl.get("last_error"),
        "model_state": model.get("model_state"),
        "deployment_eligible": bool(model.get("deployment_eligible")),
        "model_arch_version": model.get("model_arch_version"),
        "feature_version": model.get("feature_version"),
        "counts": counts,
        "latest_sealed": latest,
    }


def _build_topology_summary(
    topology: dict[str, Any] | None,
    cfg_meta: dict[str, Any],
    zone_states: list[dict[str, Any]],
    actuators: list[dict[str, Any]],
) -> dict[str, Any]:
    zone_keys = [z["zone"] for z in zone_states]
    zone_labels = {z["zone"]: z["label"] for z in zone_states}
    if not isinstance(topology, dict):
        return {
            "present": False,
            "from_example": False,
            "observation_version": None,
            "zones": [
                {
                    "zone": z["zone"],
                    "label": z["label"],
                    "actuators": [a["room"] for a in actuators if a["zone"] == z["zone"]],
                }
                for z in zone_states
            ],
            "connections": [],
            "opposite_operation": {"enabled": False, "max_cross_influence": 0.1},
        }
    topo_zones = topology.get("zones") if isinstance(topology.get("zones"), dict) else {}
    topo_actuators = topology.get("actuators") if isinstance(topology.get("actuators"), dict) else {}
    all_keys = list(topo_zones.keys()) or zone_keys
    zones_out = []
    for key in all_keys:
        label = zone_labels.get(key) or _zone_display_name(key, topology=topology, zone_keys=all_keys)
        members = [
            room
            for room, unit in topo_actuators.items()
            if isinstance(unit, dict) and unit.get("zone") == key
        ]
        zones_out.append({"zone": key, "label": label, "actuators": members})
    conns_out = []
    for link in topology.get("connections") or ():
        if not isinstance(link, dict):
            continue
        src = str(link.get("from") or "")
        dst = str(link.get("to") or "")
        coupling = _num(link.get("coupling"))
        if not src or not dst:
            continue
        conns_out.append({
            "from": src,
            "from_label": zone_labels.get(src) or _zone_display_name(src, topology=topology, zone_keys=all_keys),
            "to": dst,
            "to_label": zone_labels.get(dst) or _zone_display_name(dst, topology=topology, zone_keys=all_keys),
            "coupling": coupling,
        })
    opposing = topology.get("opposite_operation") if isinstance(topology.get("opposite_operation"), dict) else {}
    return {
        "present": True,
        "from_example": bool(cfg_meta.get("topology_from_example")),
        "observation_version": topology.get("observation_version"),
        "zones": zones_out,
        "connections": conns_out,
        "opposite_operation": {
            "enabled": bool(opposing.get("enabled", False)),
            "max_cross_influence": _num(opposing.get("max_cross_influence")) if opposing.get("max_cross_influence") is not None else 0.1,
        },
    }


def collect_snapshot(paths: DashboardPaths, *, now: float | None = None) -> dict[str, Any]:
    """Assemble a complete, read-only dashboard snapshot from runtime files."""
    now_ts = time.time() if now is None else float(now)
    policy, rooms_cfg, topology, cfg_meta = _load_config(paths)
    read_errors: list[dict[str, str]] = list(cfg_meta.get("read_errors") or [])
    status_raw, status_err = _read_json(paths.pilot_data_dir / "status.json")
    if status_err is not None:
        read_errors.append({"file": "status.json", "error": status_err})
    startup_raw, startup_err = _read_json(paths.pilot_data_dir / "startup.json")
    if startup_err is not None:
        read_errors.append({"file": "startup.json", "error": startup_err})
    audit_raw, audit_err = _tail_jsonl(paths.pilot_data_dir / "live-audit.jsonl", max_lines=80)
    if audit_err is not None:
        read_errors.append({"file": "live-audit.jsonl", "error": audit_err})
    shadow_status, _ = _read_json(paths.shadow_output_dir / "shadow-status.json")
    shadow_health, _ = _read_json(paths.shadow_output_dir / "shadow-health.json")
    neural_status, _ = _read_json(paths.neural_output_dir / "neural-shadow-status.json")
    neural_health, _ = _read_json(paths.neural_output_dir / "neural-shadow-health.json")

    status = status_raw if isinstance(status_raw, dict) else {}
    status_rooms = status.get("rooms") if isinstance(status.get("rooms"), dict) else {}
    control = status.get("control") if isinstance(status.get("control"), dict) else {}

    # Enrich control with latest cycle audit trace fields not duplicated in status.json
    # (such as equipment, observations_after, demand_order, feasible, output_evidence, energy_scores, recent_effects)
    latest_cycle = next((r for r in reversed(audit_raw) if r.get("event") == "cycle"), None)
    latest_error = next((r for r in reversed(audit_raw) if str(r.get("event", "")).endswith("_error")), None)
    if latest_cycle:
        trace_src = latest_cycle.get("verification") or latest_cycle.get("assessment")
        if isinstance(trace_src, dict):
            merged_control = dict(trace_src)
            merged_control.update({k: v for k, v in control.items() if v is not None})
            control = merged_control

    if (
        topology is None
        and not cfg_meta.get("topology_live_present")
        and control.get("topology")
        and cfg_meta.get("topology_example_exists")
    ):
        topo_ex, _ = _read_json(paths.shadow_config_dir / "topology.example.json")
        control_zones = control.get("zones") if isinstance(control.get("zones"), dict) else {}
        if isinstance(topo_ex, dict):
            ex_zones = topo_ex.get("zones") if isinstance(topo_ex.get("zones"), dict) else {}
            if not control_zones or set(control_zones).issubset(ex_zones):
                topology = topo_ex
                cfg_meta["topology_from_example"] = True

    permit_raw, permit_info = _load_permit(paths.pilot_data_dir, now_ts, rooms_cfg=rooms_cfg, topology=topology)
    journal, journal_file, journal_errors = _load_journal(paths.pilot_data_dir, permit_raw)
    read_errors.extend(journal_errors)

    status_ts = _num(control.get("at")) or _parse_ts(status.get("at"))
    latest_cycle_ts = _num(latest_cycle.get("at")) if latest_cycle else None
    controller_reports_inactive = (
        status.get("status") == "实机控制已暂停或验证窗口已结束" and not status_rooms and not control
    ) or (
        latest_cycle is not None
        and latest_cycle.get("outcome") == "permit_inactive"
        and (status_ts is None or latest_cycle_ts is None or latest_cycle_ts >= status_ts - 5.0)
    )
    if permit_info["active"] and controller_reports_inactive:
        permit_info = dict(
            permit_info,
            active=False,
            reason="控制器本轮判定许可未生效 (permit_inactive)",
        )
    status_age_s = max(0.0, now_ts - status_ts) if status_ts is not None else None
    stale = status_ts is None or (status_age_s is not None and status_age_s > STATUS_STALE_SECONDS)
    next_cycle_in_s = (
        round(max(0.0, CYCLE_INTERVAL_SECONDS - (status_age_s % CYCLE_INTERVAL_SECONDS)), 1)
        if (status_age_s is not None and not stale)
        else None
    )

    closed_loop = journal.get("closed_loop") if isinstance(journal.get("closed_loop"), dict) else {}
    raw_commands = journal.get("commands") if isinstance(journal.get("commands"), list) else []
    commands = [_normalize_command(c, rooms_cfg, now_ts) for c in raw_commands if isinstance(c, dict)]
    commands.sort(key=lambda c: c["sent_at"] or 0.0, reverse=True)

    zone_states = _build_zone_states(control, closed_loop, policy, now_ts, topology, status_rooms=status_rooms)
    handoff = _build_handoff_state(control, closed_loop, now_ts)
    actuators = _build_actuators(
        rooms_cfg, status_rooms, control, journal, permit_info, topology, shadow_status, policy, now_ts
    )
    ledger = _build_ledger_state(journal, commands, permit_info, actuators, now_ts)
    next_command = _infer_next_command(
        control=control,
        zone_states=zone_states,
        handoff=handoff,
        ledger=ledger,
        actuators=actuators,
        commands=commands,
        permit_info=permit_info,
        journal=journal,
        rooms_cfg=rooms_cfg,
        policy=policy,
        next_cycle_in_s=next_cycle_in_s,
        now=now_ts,
    )

    # Environment sensors
    env_out: dict[str, Any] = {}
    env_raw = control.get("environment") if isinstance(control.get("environment"), dict) else {}
    for key, item in env_raw.items():
        if not isinstance(item, dict):
            continue
        at_val = _num(item.get("at"))
        env_out[key] = {
            "value": _num(item.get("value")),
            "at": at_val,
            "age_s": round(max(0.0, now_ts - at_val), 1) if at_val is not None else None,
            "quality": str(item.get("quality") or "unavailable"),
        }

    # Controller overall health
    data_present = bool(status or journal or audit_raw)
    if not data_present:
        health = "offline"
    elif stale:
        health = "stale"
    elif ledger["locks"]["holds"]:
        health = "fault_locked"
    elif not permit_info["active"]:
        health = "paused"
    else:
        health = "healthy"

    # Controller mode detection
    if control.get("topology") or control.get("zones"):
        ctrl_mode = "unified_topology"
    elif isinstance(control.get("gentle"), dict) or isinstance(journal.get("gentle"), dict) or status_rooms:
        ctrl_mode = "legacy_gentle"
    else:
        ctrl_mode = "unknown"

    startup = startup_raw if isinstance(startup_raw, dict) else {}
    effects_raw = list((closed_loop.get("effects") or {}).values()) if isinstance(closed_loop.get("effects"), dict) else []
    effects = [e for e in effects_raw if isinstance(e, dict)]
    effects.sort(key=lambda e: _num(e.get("ended_at")) or 0.0, reverse=True)

    shadow_series = _load_shadow_series(paths.shadow_data_dir, rooms_cfg, now_ts)
    topology_summary = _build_topology_summary(topology, cfg_meta, zone_states, actuators)

    return {
        "generated_at": round(now_ts, 3),
        "mode": "live",
        "scenario": None,
        "available_scenarios": [],
        "controller": {
            "mode": ctrl_mode,
            "status_text": status.get("status") or ("未检测到运行中的控制器数据" if not data_present else "—"),
            "status_at": status.get("at"),
            "status_ts": status_ts,
            "status_age_s": round(status_age_s, 1) if status_age_s is not None else None,
            "status_error": status_err,
            "read_errors": read_errors,
            "cycle_interval_s": CYCLE_INTERVAL_SECONDS,
            "next_cycle_in_s": next_cycle_in_s,
            "health": health,
            "stale": stale,
            "data_present": data_present,
            "journal_file": journal_file,
            "last_cycle_outcome": latest_cycle.get("outcome") if latest_cycle else None,
            "last_cycle_outcome_label": (
                OUTCOME_LABELS.get(str(latest_cycle.get("outcome")), str(latest_cycle.get("outcome")))
                if latest_cycle and latest_cycle.get("outcome")
                else None
            ),
            "last_cycle_at": _num(latest_cycle.get("at")) if latest_cycle else None,
            "last_error": (
                {"at": _num(latest_error.get("at")), "event": latest_error.get("event"), "error": latest_error.get("error")}
                if latest_error
                else None
            ),
            "release": {
                "commit": startup.get("commit") or (latest_cycle.get("commit") if latest_cycle else None),
                "code_sha256": startup.get("code_sha256") or (latest_cycle.get("code_sha256") if latest_cycle else None),
                "config_sha256": startup.get("config_sha256") or (latest_cycle.get("config_sha256") if latest_cycle else None),
                "observation_version": startup.get("observation_version") or control.get("topology"),
                "startup_at": _num(startup.get("at")),
            },
            "permit": permit_info,
            "house_need": control.get("need") or next(
                (r.get("house_need") for r in status_rooms.values() if isinstance(r, dict) and r.get("house_need")),
                "unknown",
            ),
            "house_need_label": MODE_LABELS.get(
                str(
                    control.get("need")
                    or next(
                        (r.get("house_need") for r in status_rooms.values() if isinstance(r, dict) and r.get("house_need")),
                        "unknown",
                    )
                ),
                "未知",
            ),
            "owner": control.get("owner"),
            "lead": control.get("lead"),
            "elected": control.get("elected"),
            "conflict": bool(control.get("conflict")),
            "demand_order": list(control.get("demand_order") or []),
            "uncertain_zones": list(control.get("uncertain_zones") or []),
            "pending_rooms": list(control.get("pending_rooms") or []),
            "environment": env_out,
        },
        "state_machines": {
            "zone_states": zone_states,
            "handoff": handoff,
            "ledger": ledger,
            "gentle": control.get("gentle") if isinstance(control.get("gentle"), dict) else journal.get("gentle"),
        },
        "next_command": next_command,
        "actuators": actuators,
        "commands": commands[:40],
        "effects": effects[:12],
        "audit_recent": [_summarize_audit_record(r, rooms_cfg) for r in reversed(audit_raw[-40:])],
        "shadow": _build_shadow_summary(shadow_status, shadow_health),
        "shadow_series": shadow_series,
        "neural_shadow": _build_neural_summary(neural_status, neural_health),
        "topology": topology_summary,
        "policy": policy,
    }
