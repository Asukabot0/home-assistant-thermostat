"""Unified hard-constraint checks and V1 shadow candidate generator.

V1 scope restriction:
- ONLY `hold` and same-mode `±1°C` adjustments on currently active (`heat`/`cool`) units.
- Start (`off -> heat/cool`), stop (`heat/cool -> off`), and mode reversals (`heat <-> cool`)
  may appear in historical training data, but are STRICTLY prohibited as v1 candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import CONSTRAINT_VERSION
from .config import SiteConfig, shadow_model
from .data_pipeline import FeatureSpec, build_hold_plan, build_step_action_plan

ACTIVE_MODES = {"heat", "cool"}
V1_ALLOWED_KINDS = {"hold", "same_mode_setpoint"}


@dataclass(frozen=True)
class CandidateAction:
    """Single candidate action plan evaluated by the shadow model."""

    candidate_id: str
    kind: str  # "hold" | "same_mode_setpoint"
    unit: str | None
    mode: str | None
    from_setpoint: float | None
    to_setpoint: float | None
    step_c: float
    step_plans: list[dict[str, tuple[str | None, float | None]]]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "unit": self.unit,
            "mode": self.mode,
            "from_setpoint": self.from_setpoint,
            "to_setpoint": self.to_setpoint,
            "step_c": self.step_c,
            "constraint_version": CONSTRAINT_VERSION,
        }


def validate_v1_action_proposal(
    site: SiteConfig,
    meta: dict[str, Any],
    *,
    unit: str,
    proposed_mode: str,
    proposed_setpoint: float | None,
) -> tuple[bool, str]:
    """Validate an individual action against V1 scope and unified hard safety constraints."""
    if unit not in site.unit_names:
        return False, "unknown_unit"
    act = site.actuator_by_name(unit)
    cur_mode = meta["unit_modes"].get(unit)
    cur_sp = meta["unit_setpoints"].get(unit)

    # Hard V1 scope rule: NO start, NO stop, NO mode reversal.
    if cur_mode not in ACTIVE_MODES or proposed_mode != cur_mode:
        return False, "v1_action_scope_prohibits_start_stop_or_reversal"

    if cur_sp is None or proposed_setpoint is None:
        return False, "missing_setpoint"

    step = float(proposed_setpoint) - float(cur_sp)
    if abs(abs(step) - 1.0) > 1e-6:
        return False, "v1_step_must_be_exactly_plus_or_minus_1c"

    # Device protection: manual hold, failure hold, isolation, quiet mode
    holds = (meta.get("unit_holds") or {}).get(unit) or {}
    if holds.get("isolated"):
        return False, "unit_isolated"
    if holds.get("manual_hold"):
        return False, "unit_manual_hold"
    if holds.get("failure_hold"):
        return False, "unit_failure_hold"
    if holds.get("quiet"):
        return False, "unit_quiet_preset_active"

    # Zone pause & fresh temperature
    home_zone = act.home_zone
    if (meta.get("zone_paused") or {}).get(home_zone):
        return False, "home_zone_paused"
    if (meta.get("zone_temps") or {}).get(home_zone) is None or (meta.get("zone_targets") or {}).get(home_zone) is None:
        return False, "home_zone_observation_or_target_missing"

    # Hardware bounds and room soft limits (reuses home-assistant/shadow/model.py)
    limits = act.soft_limits.get(cur_mode)
    bounds = shadow_model.setpoint_bounds(limits)
    if bounds is None:
        return False, "invalid_soft_limits"
    auto_sp = shadow_model.automatic_setpoint(cur_sp, int(round(step)), limits)
    if auto_sp is None or abs(float(auto_sp) - float(proposed_setpoint)) > 1e-6:
        return False, "outside_mode_soft_limits"
    if not (bounds[0] <= proposed_setpoint <= bounds[1]):
        return False, "outside_mode_soft_limits"

    # Fresh power evidence required for equipment safety checks
    if (meta.get("unit_watts") or {}).get(unit) is None:
        return False, "power_evidence_missing"

    increases = shadow_model.increases_output(cur_mode, cur_sp, proposed_setpoint)
    cutoff_ts = float(meta["cutoff_ts"])
    policy = site.policy

    # Post-action feedback wait & cooldown
    last_act_at = (meta.get("unit_last_action_at") or {}).get(unit)
    home_rep_at = (meta.get("zone_report_at") or {}).get(home_zone)
    wait_s = float(policy.get("feedback_wait_seconds", 600) if increases else policy.get("reduction_wait_seconds", 180))
    if last_act_at is not None:
        if cutoff_ts - last_act_at < wait_s:
            return False, "waiting_post_action_cooldown"
        if home_rep_at is None or home_rep_at <= last_act_at:
            return False, "waiting_new_temperature_report_after_action"

    # Strengthening checks: opposing-mode interlock and peer comfort risk
    if increases:
        opp_enabled = bool(site.opposite_operation.get("enabled", False))
        for peer_u, peer_m in (meta.get("unit_modes") or {}).items():
            if peer_u != unit and peer_m in ACTIVE_MODES and peer_m != cur_mode and not opp_enabled:
                return False, "opposing_mode_active"

        direction = 1.0 if cur_mode == "heat" else -1.0
        comfort_band = float(policy.get("comfort_band", 0.7))
        for peer_z, influence in act.influences.items():
            if peer_z == home_zone or influence <= 0.0:
                continue
            pt = (meta.get("zone_temps") or {}).get(peer_z)
            ptgt = (meta.get("zone_targets") or {}).get(peer_z)
            if pt is None or ptgt is None:
                return False, f"peer_observation_missing:{peer_z}"
            adverse = direction * (pt - ptgt)
            if adverse > comfort_band + 1e-6:
                return False, f"peer_comfort_risk:{peer_z}"

    return True, "ok"


def generate_v1_candidates(
    site: SiteConfig,
    spec: FeatureSpec,
    meta: dict[str, Any],
) -> tuple[list[CandidateAction], list[dict[str, Any]]]:
    """Generate all valid V1 candidates (hold + constraint-checked same-mode ±1°C) plus rejection log."""
    hold_plan = build_hold_plan(spec, meta["unit_modes"], meta["unit_setpoints"])
    candidates: list[CandidateAction] = [
        CandidateAction(
            candidate_id="hold",
            kind="hold",
            unit=None,
            mode=None,
            from_setpoint=None,
            to_setpoint=None,
            step_c=0.0,
            step_plans=hold_plan,
        )
    ]
    rejections: list[dict[str, Any]] = []

    for unit in spec.unit_names:
        cur_mode = meta["unit_modes"].get(unit)
        cur_sp = meta["unit_setpoints"].get(unit)
        if cur_mode not in ACTIVE_MODES:
            rejections.append(
                {
                    "unit": unit,
                    "candidate": f"{unit}:mode_not_active",
                    "reason": "v1_action_scope_prohibits_start_stop_or_reversal",
                }
            )
            continue
        if cur_sp is None:
            rejections.append({"unit": unit, "candidate": f"{unit}:no_setpoint", "reason": "missing_setpoint"})
            continue

        for delta in (-1.0, 1.0):
            proposed = round(float(cur_sp) + delta, 1)
            ok, reason = validate_v1_action_proposal(
                site, meta, unit=unit, proposed_mode=cur_mode, proposed_setpoint=proposed
            )
            cid = f"{unit}:{cur_mode}:{int(cur_sp)}->{int(proposed)}"
            if not ok:
                rejections.append({"unit": unit, "candidate": cid, "reason": reason})
                continue
            plan = build_step_action_plan(
                spec,
                meta["unit_modes"],
                meta["unit_setpoints"],
                unit=unit,
                new_mode=cur_mode,
                new_setpoint=proposed,
            )
            candidates.append(
                CandidateAction(
                    candidate_id=cid,
                    kind="same_mode_setpoint",
                    unit=unit,
                    mode=cur_mode,
                    from_setpoint=float(cur_sp),
                    to_setpoint=float(proposed),
                    step_c=delta,
                    step_plans=plan,
                )
            )

    return candidates, rejections
