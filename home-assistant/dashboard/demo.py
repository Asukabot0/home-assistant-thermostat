"""Deterministic multi-scenario demo fixtures for previewing the dashboard UI.

Only activated when explicitly requested via `--demo`. Never used as a silent
fallback in live production mode so operators can never confuse simulated data
with real house state.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

from .collector import DashboardPaths, collect_snapshot

SCENARIOS: list[dict[str, str]] = [
    {
        "id": "recover_strengthen",
        "title": "恢复加码 · 等待观察窗",
        "subtitle": "卧室偏冷已确认，预计下发 22→23℃",
    },
    {
        "id": "handoff_conflict",
        "title": "双区冲突 · 双向交接",
        "subtitle": "冷热需求冲突，对侧先减载再停机",
    },
    {
        "id": "inflight_echo",
        "title": "在途指令 · 等待回报",
        "subtitle": "已下发启机指令，等待伴侣回显",
    },
    {
        "id": "gate_blocked",
        "title": "候选拦截 · 门禁保护",
        "subtitle": "最短停机低耗未满 + 外部手动锁",
    },
    {
        "id": "dwell_building",
        "title": "偏差累计 · 资格倒计时",
        "subtitle": "客厅偏热累计中，距恢复还差 6 分钟",
    },
    {
        "id": "maintain_quiet",
        "title": "全屋维持 · 稳态运行",
        "subtitle": "双区在 ±0.3℃ 跟踪带内稳定维持",
    },
]

SCENARIO_IDS = {item["id"] for item in SCENARIOS}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def _base_shadow_status(now: float) -> tuple[dict[str, Any], dict[str, Any]]:
    status = {
        "started_at": _iso(now - 5 * 86400),
        "checked_at": _iso(now - 35),
        "published_at": _iso(now - 32),
        "latest_observation_at": _iso(now - 60),
        "stale_after": _iso(now + 180),
        "stale": False,
        "policy_revision": 9,
        "release_commit": "259bc5f49e1c3a8d7f02b918a4c5e123456789ab",
        "assessment_completed_minutes": 1440,
        "data_ready_for_24h_review": True,
        "observed_home_kwh": 4.682,
        "observed_home_minutes": 1392,
        "observed_home_coverage_pct": 96.7,
        "live_actuations": {"attempts": 8, "proxy_echoes": 8, "starts": 2, "stops": 1, "reversals": 0},
        "rooms": {
            "bedroom": {
                "coverage_pct": 98.4,
                "latest": {
                    "activity": "稳定制热",
                    "confidence": "高",
                    "power_reason": "功率处于中载区间且连续上报",
                    "model": {"centers_w": [11.2, 185.0, 510.0], "edges_w": [45.0, 340.0]},
                },
                "totals": {
                    "within_comfort_pct": 95.8,
                    "within_ideal_pct": 84.2,
                    "mae_c": 0.28,
                    "forecast_15m_mae_c": 0.14,
                    "metered_kwh": 2.154,
                    "metered_seconds": 84820,
                },
            },
            "living": {
                "coverage_pct": 97.9,
                "latest": {
                    "activity": "低频维持",
                    "confidence": "高",
                    "power_reason": "功率稳定高于待机阈值",
                    "model": {"centers_w": [13.5, 160.0, 490.0], "edges_w": [48.0, 325.0]},
                },
                "totals": {
                    "within_comfort_pct": 97.1,
                    "within_ideal_pct": 89.5,
                    "mae_c": 0.21,
                    "forecast_15m_mae_c": 0.11,
                    "metered_kwh": 1.892,
                    "metered_seconds": 84600,
                },
            },
            "study": {
                "coverage_pct": 97.9,
                "latest": {
                    "activity": "停机待机",
                    "confidence": "高",
                    "power_reason": "连续低于停机低耗阈值",
                    "model": {"centers_w": [8.4, 150.0, 420.0], "edges_w": [32.0, 285.0]},
                },
                "totals": {
                    "within_comfort_pct": 97.1,
                    "within_ideal_pct": 89.5,
                    "mae_c": 0.21,
                    "forecast_15m_mae_c": 0.11,
                    "metered_kwh": 0.636,
                    "metered_seconds": 85100,
                },
            },
        },
        "completed_days": [
            {"localday": "2026-11-27", "room": "bedroom", "metrics": {"valid_temperature_minutes": 1418, "within_comfort_pct": 96.1, "mae_c": 0.26, "hypothetical_beeps": 5}},
            {"localday": "2026-11-27", "room": "living", "metrics": {"valid_temperature_minutes": 1425, "within_comfort_pct": 97.8, "mae_c": 0.19, "hypothetical_beeps": 4}},
            {"localday": "2026-11-27", "room": "study", "metrics": {"valid_temperature_minutes": 1425, "within_comfort_pct": 97.8, "mae_c": 0.19, "hypothetical_beeps": 1}},
        ],
    }
    health = {"at": now - 12, "last_success_tick": int(now // 60 * 60 - 60), "stale_after": now + 180, "state": "healthy"}
    return status, health


def _base_neural_status(now: float) -> tuple[dict[str, Any], dict[str, Any]]:
    status = {
        "model": {
            "model_state": "uncalibrated",
            "deployment_eligible": False,
            "model_arch_version": "ns-gru-v1",
            "feature_version": "ns-feat-v1",
        },
        "counts": {"sealed_predictions": 148, "settled_horizons": 412, "factual_hold": 340},
        "latest_sealed": {
            "prediction_id": "pred_demo_01",
            "cutoff_ts": now - 180,
            "declined": True,
            "decline_reason": "model_not_ready_to_use",
        },
    }
    health = {"state": "healthy", "last_error": None, "at": now - 25}
    return status, health


def _populate_scenario(paths: DashboardPaths, scenario: str, now: float) -> None:
    permit_start = int(now - 4 * 86400)
    startup = {
        "commit": "259bc5f49e1c3a8d7f02b918a4c5e123456789ab",
        "code_sha256": "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0",
        "config_sha256": "9f8e7d6c5b4a3210fedcba9876543210abcdef0123456789abcdef0123456789",
        "at": now - 6 * 3600,
        "event": "startup",
        "observation_version": "site-1",
        "observation_fingerprint": "f0e1d2c3b4a5968778695a4b3c2d1e0f0123456789abcdef0123456789abcdef",
    }
    permit = {
        "enabled": True,
        "start": permit_start,
        "end": None,
        "isolated_rooms": [],
        "paused_zones": [],
        "rooms": {
            "bedroom": ["heat", 22, "auto"],
            "living": ["heat", 23, "auto"],
            "study": ["cool", 25, "auto"],
        },
    }

    sleep_points = [
        [now - 3000, 23.85], [now - 2400, 23.72], [now - 1800, 23.60],
        [now - 1380, 23.52], [now - 960, 23.45], [now - 540, 23.38], [now - 95, 23.35],
    ]
    lounge_points = [
        [now - 3000, 24.02], [now - 2400, 23.98], [now - 1800, 24.05],
        [now - 1200, 23.96], [now - 600, 23.94], [now - 80, 23.95],
    ]

    env = {"outdoor_reference": {"value": 11.6, "at": now - 140, "quality": "fresh"}}
    base_commands = [
        {
            "room": "bedroom",
            "kind": "temperature",
            "value": 22,
            "before": ["heat", 21, "auto"],
            "sent_at": now - 462,
            "relay_at": now - 461.55,
            "proxy_at": now - 458,
            "status": "proxy_echo",
            "http": 200,
            "purpose": "zone_strengthen",
            "transaction": None,
            "target_at_send": 24.0,
            "expected": ["heat", 22, "auto"],
            "before_room_c": 23.45,
            "before_watts": 148.0,
            "before_mode_at": now - 7200,
            "followup": {
                "5": {"at": now - 160, "room_c": 23.38, "room_age_s": 55, "watts": 212.0, "power_age_s": 14, "reported": ["heat", 22, "auto"]},
            },
            "physical_verified": True,
            "control": {"zone": "sleep", "mode": "heat", "target": 24.0, "before_c": 23.45},
        },
        {
            "room": "study",
            "kind": "mode",
            "value": "off",
            "before": ["heat", 22, "auto"],
            "sent_at": now - 2800,
            "relay_at": now - 2799.6,
            "proxy_at": now - 2795,
            "status": "proxy_echo",
            "http": 200,
            "purpose": "energy_off",
            "transaction": None,
            "target_at_send": 24.0,
            "expected": ["off"],
            "before_room_c": 24.35,
            "before_watts": 92.0,
            "before_mode_at": now - 9600,
            "followup": {
                "5": {"at": now - 2500, "room_c": 24.22, "room_age_s": 40, "watts": 8.4, "power_age_s": 11, "reported": ["off", None, "auto"]},
                "15": {"at": now - 1900, "room_c": 24.05, "room_age_s": 32, "watts": 8.3, "power_age_s": 9, "reported": ["off", None, "auto"]},
            },
            "physical_verified": True,
            "control": {"zone": "lounge", "mode": "heat", "target": 24.0, "before_c": 24.35},
        },
    ]

    journal: dict[str, Any] = {
        "start": permit_start,
        "baselines": {"bedroom": ["heat", 22, "auto"], "living": ["heat", 23, "auto"], "study": ["off", None, "auto"]},
        "holds": {},
        "manual_holds": {},
        "parked": {"study": {"off_at": now - 2795, "mode": "heat", "started_at": now - 9600}},
        "transitions": {},
        "house_lead": "bedroom",
        "commands": base_commands,
        "closed_loop": {
            "owner": "sleep",
            "selected_at": now - 1380,
            "trackers": {
                "sleep": {"points": sleep_points, "recovery": {"direction": "heat", "since": now - 1380, "report_span": 840, "reports": 4, "last_outside": now - 95}},
                "lounge": {"points": lounge_points},
            },
            "maintenance": {
                "living": {"heat": {"setpoint": 23, "target": 24.0, "source_version": "demo-lounge-v1", "at": now - 900, "observed_seconds": 1560}},
                "bedroom": {"heat": {"setpoint": 22, "target": 24.0, "source_version": "demo-sleep-v1", "at": now - 3600, "observed_seconds": 1320}},
            },
            "effects": {
                str(now - 5400): {
                    "unit": "bedroom",
                    "zone": "sleep",
                    "mode": "heat",
                    "target": 24.0,
                    "ended_at": now - 4200,
                    "start_c": 23.3,
                    "end_c": 23.75,
                    "useful": True,
                    "improvement": 0.45,
                    "thermal_delta": 0.45,
                    "watt_hours": 58.2,
                    "covered_seconds": 1200,
                    "interval_seconds": 1200,
                    "weather_c": 11.4,
                },
            },
        },
    }

    # Default scenario: recover_strengthen
    status_text = "持续实机控制运行中"
    rooms_detail = {
        "bedroom": {
            "room_c": 23.35, "target_c": 24.0, "mode_reported": "heat", "setpoint_reported": 22,
            "power_w": 218.0, "ready": True, "temperature_age_s": 95.0, "power_age_s": 14.0,
            "temperature_report_at": now - 95, "power_report_at": now - 14, "power_evidence_at": now - 14,
            "power_bands": {"low": {"threshold": 22.0, "seconds": 0}, "above_low": {"threshold": 22.0, "seconds": 3240}, "high": {"threshold": 340.0, "seconds": 0}},
            "house_need": "heat", "status": "observe_local_response",
            "zone": "sleep", "zone_intent": "observe", "zone_reason": "confirmed_bias", "zone_target_c": 24.0, "zone_temperature_c": 23.35,
        },
        "living": {
            "room_c": 23.95, "target_c": 24.0, "mode_reported": "heat", "setpoint_reported": 23,
            "power_w": 146.0, "ready": True, "temperature_age_s": 80.0, "power_age_s": 11.0,
            "temperature_report_at": now - 80, "power_report_at": now - 11, "power_evidence_at": now - 11,
            "power_bands": {"low": {"threshold": 24.0, "seconds": 0}, "above_low": {"threshold": 24.0, "seconds": 4100}, "high": {"threshold": 325.0, "seconds": 0}},
            "house_need": "heat", "status": "hold_effective_setting",
            "zone": "lounge", "zone_intent": "maintain", "zone_reason": "near_target", "zone_target_c": 24.0, "zone_temperature_c": 23.95,
        },
        "study": {
            "room_c": 23.95, "target_c": 24.0, "mode_reported": "off", "setpoint_reported": None,
            "power_w": 8.4, "ready": True, "temperature_age_s": 80.0, "power_age_s": 18.0,
            "temperature_report_at": now - 80, "power_report_at": now - 18, "power_evidence_at": now - 18,
            "power_bands": {"low": {"threshold": 18.0, "seconds": 2760}, "above_low": {"threshold": 18.0, "seconds": 0}, "high": {"threshold": 285.0, "seconds": 0}},
            "house_need": "heat", "status": "no_admissible_start",
            "zone": "lounge", "zone_intent": "maintain", "zone_reason": "near_target", "zone_target_c": 24.0, "zone_temperature_c": 23.95,
        },
    }
    control: dict[str, Any] = {
        "at": now - 6,
        "selected": None,
        "topology": "site-1",
        "definition": startup["observation_fingerprint"],
        "need": "heat",
        "owner": "sleep",
        "lead": "bedroom",
        "elected": None,
        "conflict": False,
        "demand_order": ["sleep"],
        "feasible": {"sleep": ["bedroom"]},
        "output_evidence": {"bedroom": True, "living": True, "study": False},
        "risks": {"bedroom": [], "living": [], "study": []},
        "candidates": [],
        "rejections": [],
        "pending_rooms": [],
        "unmet": {},
        "handoff": None,
        "yielding_blocks": {},
        "uncertain_zones": [],
        "environment": env,
        "equipment": {
            "bedroom": {"served_zone": "sleep", "power_at": now - 14, "bands": rooms_detail["bedroom"]["power_bands"]},
            "living": {"served_zone": "lounge", "power_at": now - 11, "bands": rooms_detail["living"]["power_bands"]},
            "study": {"served_zone": "lounge", "power_at": now - 18, "bands": rooms_detail["study"]["power_bands"]},
        },
        "observations_after": {
            "bedroom": {"zone_report": now - 95, "last_increase": now - 458, "own_command": now - 458},
            "living": {"zone_report": now - 80, "last_increase": now - 458, "own_command": None},
            "study": {"zone_report": now - 80, "last_increase": now - 458, "own_command": now - 2795},
        },
        "zones": {
            "sleep": {
                "key": "sleep", "measured": 23.35, "target": 24.0, "error": 0.65, "at": now - 95,
                "quality": "fresh", "paused": False, "intent": "recover", "reason": "confirmed_bias",
                "direction": "heat", "duration": 1380, "report_span": 840, "reports": 4,
                "qualified": True, "recovery_active": True, "evidence_from": now - 1380,
                "settling_reports": 0, "settling_span": 0, "source_version": "demo-sleep-v1",
                "trend": {"delta": -0.06, "uncertainty": 0.05, "from": now - 960, "to": now - 95, "reports": 4},
            },
            "lounge": {
                "key": "lounge", "measured": 23.95, "target": 24.0, "error": 0.05, "at": now - 80,
                "quality": "fresh", "paused": False, "intent": "maintain", "reason": "near_target",
                "direction": None, "duration": 0, "report_span": 0, "reports": 0,
                "qualified": False, "recovery_active": False, "evidence_from": None,
                "settling_reports": 0, "settling_span": 0, "source_version": "demo-lounge-v1",
                "trend": {"delta": 0.02, "uncertainty": 0.04, "from": now - 1200, "to": now - 80, "reports": 3},
            },
        },
    }

    if scenario == "handoff_conflict":
        journal["commands"].append({
            "room": "living", "kind": "temperature", "value": 25, "before": ["cool", 24, "auto"],
            "sent_at": now - 102, "relay_at": now - 101.6, "proxy_at": now - 98,
            "status": "proxy_echo", "http": 200, "purpose": "handoff_reduce",
            "transaction": None, "target_at_send": 24.0, "expected": ["cool", 25, "auto"],
            "before_room_c": 24.62, "before_watts": 312.0, "before_mode_at": now - 4200,
            "followup": {}, "physical_verified": False,
            "control": {"zone": "lounge", "mode": "cool", "handoff_at": now - 160, "handoff_phase": "reduce"},
        })
        rooms_detail["living"].update(mode_reported="cool", setpoint_reported=25, room_c=24.62, power_w=245.0,
                                      status="observe_both_sensors_after_reduction", zone_intent="limited", zone_reason="conflicting_priority")
        rooms_detail["bedroom"].update(room_c=23.15, status="wait_coordinated_exit", zone_intent="recover")
        control.update(
            conflict=True, owner="sleep", elected="bedroom", demand_order=["sleep", "lounge"],
            feasible={"sleep": ["bedroom"], "lounge": ["living"]},
            unmet={"sleep": "handoff_pending", "lounge": "conflicting_priority"},
            handoff={
                "owner": "sleep", "receiver": "bedroom", "blocker": "living", "direction": "heat",
                "phase": "verify_reduction", "reason": "observe_both_sensors_after_reduction",
                "selected_at": now - 160,
            },
        )
        control["zones"]["sleep"].update(measured=23.15, error=0.85, duration=1620)
        control["zones"]["lounge"].update(
            measured=24.62, error=-0.62, intent="limited", reason="conflicting_priority",
            direction="cool", duration=1290, report_span=720, reports=3, qualified=True, recovery_active=True,
            trend={"delta": 0.04, "uncertainty": 0.05, "from": now - 900, "to": now - 80, "reports": 3},
        )
        journal["closed_loop"]["trackers"]["lounge"]["points"] = [
            [now - 2400, 24.28], [now - 1800, 24.42], [now - 1200, 24.51], [now - 600, 24.58], [now - 80, 24.62],
        ]

    elif scenario == "inflight_echo":
        status_text = "已记录一条米家实机指令，等待伴侣回报"
        sent = now - 34
        journal["commands"].append({
            "room": "living", "kind": "mode", "value": "cool", "before": ["off", None, "auto"],
            "sent_at": sent, "relay_at": sent + 0.48, "proxy_at": None,
            "status": "accepted", "http": 200, "purpose": "energy_start",
            "transaction": sent, "target_at_send": 24.0, "expected": ["cool"],
            "before_room_c": 24.78, "before_watts": 12.1, "before_mode_at": now - 3900,
            "followup": {}, "physical_verified": False,
            "control": {"zone": "lounge", "mode": "cool", "target": 24.0, "before_c": 24.78},
        })
        journal["transitions"] = {
            "living_study": {
                "id": sent, "mode": "cool", "ready_at": sent + 5,
                "members": {
                    "living": {"phase": "queued", "target": 24.0, "control": {"zone": "lounge", "target": 24.0}},
                },
            }
        }
        rooms_detail["living"].update(mode_reported="off", setpoint_reported=None, room_c=24.78, power_w=12.1,
                                      ready=False, status="等待米家指令回报", zone_intent="recover")
        rooms_detail["bedroom"].update(room_c=23.98, status="hold_effective_setting", zone_intent="maintain")
        control.update(
            need="cool", owner="lounge", lead="living", demand_order=["lounge"],
            feasible={"lounge": ["living"]}, pending_rooms=["living"],
            selected=["living", "mode", "cool", "energy_start"],
        )
        control["zones"]["sleep"].update(measured=23.98, error=0.02, intent="maintain", reason="near_target", direction=None, qualified=False, recovery_active=False)
        control["zones"]["lounge"].update(
            measured=24.78, error=-0.78, intent="recover", reason="confirmed_bias", direction="cool",
            duration=1510, report_span=900, reports=4, qualified=True, recovery_active=True,
            trend={"delta": 0.11, "uncertainty": 0.05, "from": now - 900, "to": now - 45, "reports": 4},
        )

    elif scenario == "gate_blocked":
        journal["parked"]["bedroom"] = {"off_at": now - 195, "mode": "cool", "started_at": now - 2500}
        journal["manual_holds"]["living"] = "检测到外部修改了空调设置，已暂停该机"
        rooms_detail["bedroom"].update(
            mode_reported="off", setpoint_reported=None, power_w=10.8, power_age_s=16.0,
            power_bands={"low": {"threshold": 22.0, "seconds": 175}, "above_low": {"threshold": 22.0, "seconds": 0}, "high": {"threshold": 340.0, "seconds": 0}},
            status="等待本机停机时间和独立低耗证据",
        )
        rooms_detail["living"].update(ready=False, status="检测到外部修改了空调设置，已暂停该机")
        control.update(
            candidates=[["bedroom", "mode", "heat", "energy_start"]],
            rejections=[[["bedroom", "mode", "heat", "energy_start"], "等待本机停机时间和独立低耗证据"]],
            output_evidence={"bedroom": False, "living": True, "study": False},
        )

    elif scenario == "dwell_building":
        rooms_detail["bedroom"].update(room_c=23.92, status="hold_effective_setting", zone_intent="maintain", zone_reason="near_target")
        rooms_detail["living"].update(mode_reported="cool", setpoint_reported=25, room_c=24.55, power_w=132.0,
                                      status="hold_effective_setting", zone_intent="observe", zone_reason="waiting_reports_or_dwell")
        rooms_detail["study"].update(room_c=24.55, zone_intent="observe", zone_reason="waiting_reports_or_dwell")
        control.update(
            need="hold", owner=None, lead="living", demand_order=[], feasible={},
        )
        control["zones"]["sleep"].update(
            measured=23.92, error=0.08, intent="maintain", reason="near_target",
            direction=None, duration=0, report_span=0, reports=0, qualified=False, recovery_active=False,
        )
        control["zones"]["lounge"].update(
            measured=24.55, error=-0.55, intent="observe", reason="waiting_reports_or_dwell",
            direction="cool", duration=820, report_span=410, reports=2, qualified=False, recovery_active=False,
            trend={"delta": 0.06, "uncertainty": 0.05, "from": now - 600, "to": now - 70, "reports": 3},
        )
        journal["closed_loop"]["trackers"]["lounge"]["points"] = [
            [now - 2400, 24.18], [now - 1800, 24.25], [now - 1200, 24.34], [now - 820, 24.42], [now - 410, 24.50], [now - 70, 24.55],
        ]

    elif scenario == "maintain_quiet":
        rooms_detail["bedroom"].update(room_c=23.94, status="hold_effective_setting", zone_intent="maintain", zone_reason="near_target", power_w=138.0)
        rooms_detail["living"].update(room_c=24.06, status="hold_effective_setting", zone_intent="maintain", zone_reason="near_target", power_w=122.0)
        rooms_detail["study"].update(room_c=24.06, status="no_admissible_start", zone_intent="maintain", zone_reason="near_target")
        control.update(need="hold", owner=None, lead="bedroom", demand_order=[], feasible={})
        control["zones"]["sleep"].update(
            measured=23.94, error=0.06, intent="maintain", reason="near_target",
            direction=None, duration=0, report_span=0, reports=0, qualified=False, recovery_active=False,
            trend={"delta": 0.01, "uncertainty": 0.04, "from": now - 1000, "to": now - 60, "reports": 4},
        )
        control["zones"]["lounge"].update(
            measured=24.06, error=-0.06, intent="maintain", reason="near_target",
            direction=None, duration=0, report_span=0, reports=0, qualified=False, recovery_active=False,
            trend={"delta": -0.02, "uncertainty": 0.04, "from": now - 1000, "to": now - 50, "reports": 4},
        )
        journal["closed_loop"]["trackers"]["sleep"]["points"] = [
            [now - 3000, 23.88], [now - 2400, 23.91], [now - 1800, 23.95], [now - 1200, 23.93], [now - 600, 23.96], [now - 60, 23.94],
        ]

    audit_rows = [
        {**startup, "at": now - 3600},
        {"at": now - 462, "event": "cycle", "outcome": "dispatch", "final": ["bedroom", "temperature", 22, "zone_strengthen"], "assessment": {"need": "heat", "owner": "sleep", "candidates": [["bedroom", "temperature", 22, "zone_strengthen"]], "rejections": []}},
        {"at": now - 461.5, "event": "relay", "room": "bedroom", "status": "accepted", "http": 200, "command_at": now - 462},
        {"at": now - 458, "event": "receipt", "room": "bedroom", "status": "proxy_echo", "http": 200, "command_at": now - 462},
        {"at": now - 46, "event": "cycle", "outcome": "idle", "assessment": control},
        {"at": now - 26, "event": "cycle", "outcome": "idle", "assessment": control},
        {
            "at": now - 6,
            "event": "cycle",
            "outcome": "dispatch" if control.get("selected") else "idle",
            "final": control.get("selected"),
            "assessment": control,
        },
    ]

    _write_json(paths.pilot_data_dir / "startup.json", startup)
    _write_json(paths.pilot_data_dir / "live-permit.json", permit)
    _write_json(paths.pilot_data_dir / f"continuous-{permit_start}.json", journal)
    _write_json(paths.pilot_data_dir / "status.json", {"at": _iso(now - 6), "status": status_text, "rooms": rooms_detail, "control": control})
    _write_jsonl(paths.pilot_data_dir / "live-audit.jsonl", audit_rows)

    shadow_status, shadow_health = _base_shadow_status(now)
    _write_json(paths.shadow_output_dir / "shadow-status.json", shadow_status)
    _write_json(paths.shadow_output_dir / "shadow-health.json", shadow_health)

    neural_status, neural_health = _base_neural_status(now)
    _write_json(paths.neural_output_dir / "neural-shadow-status.json", neural_status)
    _write_json(paths.neural_output_dir / "neural-shadow-health.json", neural_health)


def build_demo_snapshot(
    scenario: str = "recover_strengthen",
    *,
    config_dir: Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Generate a live-schema snapshot for the requested demo scenario."""
    selected = scenario if scenario in SCENARIO_IDS else "recover_strengthen"
    now_ts = time.time() if now is None else float(now)
    shadow_cfg = config_dir or (Path(__file__).resolve().parents[1] / "shadow")

    with TemporaryDirectory(prefix="thermostat-demo-") as tmp:
        root = Path(tmp)
        paths = DashboardPaths(
            shadow_config_dir=shadow_cfg,
            pilot_data_dir=root / "pilot-data",
            shadow_data_dir=root / "shadow-data",
            shadow_output_dir=root / "climate-shadow",
            neural_output_dir=root / "neural-shadow",
        )
        _populate_scenario(paths, selected, now_ts)
        snapshot = collect_snapshot(paths, now=now_ts)
        snapshot["mode"] = "demo"
        snapshot["scenario"] = selected
        snapshot["available_scenarios"] = list(SCENARIOS)
        return snapshot
