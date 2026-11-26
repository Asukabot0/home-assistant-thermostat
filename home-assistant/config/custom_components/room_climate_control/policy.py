"""Command policy for inverter units behind a potentially one-way adapter."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import isfinite

ACTIVE = {"heat", "cool"}
SUPPORTED_MODES = ACTIVE | {"off", "auto"}
MIN_RUN = timedelta(minutes=15)
MIN_OFF = timedelta(minutes=5)
MIN_REVERSE = timedelta(hours=1)
MIN_DAY = timedelta(minutes=10)
MIN_QUIET = timedelta(minutes=30)
CONFIRM_TIMEOUT = timedelta(minutes=3)

@dataclass(frozen=True)
class Reading:
    value: float | None = None
    time: datetime | None = None

    def fresh(self, now: datetime, max_age: timedelta, minimum: float, maximum: float) -> bool:
        return (
            self.value is not None and isfinite(self.value) and minimum <= self.value <= maximum
            and self.time is not None and timedelta(0) <= now - self.time <= max_age
        )

@dataclass(frozen=True)
class Request:
    mode: str = "off"
    temperature: float = 24.0
    quiet: bool = False

@dataclass(frozen=True)
class Evidence:
    room: Reading = field(default_factory=Reading)
    power: Reading = field(default_factory=Reading)
    adapter_mode: str | None = None
    adapter_setpoint: float | None = None
    independent_mode: str | None = None
    phase_since: datetime | None = None
    last_active_mode: str | None = None
    last_active_start: datetime | None = None
    power_stable_since: datetime | None = None
    adapter_available: bool = False
    history_ready_since: datetime | None = None
    independent_setpoint: float | None = None
    proxy_report_fresh: bool = False

@dataclass(frozen=True)
class Limits:
    authorized: bool = False
    absolute_mode_verified: bool = False
    ir_setpoint_validated: bool = False
    room_freshness: timedelta = timedelta(minutes=20)
    power_freshness: timedelta = timedelta(minutes=12)
    min_setpoint: float = 16
    max_setpoint: float = 30
    step: float = 1
    active_watts: float = 100
    low_watts: float = 55

@dataclass(frozen=True)
class Command:
    kind: str
    value: str | float

@dataclass(frozen=True)
class Pending:
    command: Command
    sent: datetime

@dataclass(frozen=True)
class History:
    last_command: Pending | None = None
    pending: Pending | None = None
    manual_until: datetime | None = None
    manual_latched: bool = False
    retry_after: datetime | None = None

@dataclass(frozen=True)
class Decision:
    status: str
    activity: str
    confidence: str
    command: Command | None = None

def activity(now: datetime, data: Evidence, limits: Limits) -> tuple[str, str]:
    if data.independent_mode in ACTIVE | {"off"}:
        return data.independent_mode, "verified"
    if not data.power.fresh(now, limits.power_freshness, 0, 10_000):
        return "unknown", "unknown"
    if (data.power_stable_since is None
        or not timedelta(minutes=1) <= now - data.power_stable_since):
        return "settling", "unknown"
    if data.power.value >= limits.active_watts:
        return "load_detected", "inferred"
    if data.power.value <= limits.low_watts:
        return "low_draw", "inferred"
    return "unknown", "unknown"

def has_transport_evidence(now: datetime, proxy_reported: datetime | None, power: Reading,
                           limits: Limits) -> bool:
    return proxy_reported is not None and (
        timedelta(0) <= now - proxy_reported <= timedelta(minutes=30)
        or power.fresh(now, limits.power_freshness, 0, 10_000)
    )

def demand(request: Request, measured: float, existing_mode: str | None) -> str | None:
    if request.mode != "auto":
        return request.mode
    band = 0.8 if request.quiet else 0.5
    if measured > request.temperature + band:
        return "cool"
    if measured < request.temperature - band:
        return "heat"
    return existing_mode if existing_mode in ACTIVE else None

def step_toward(current: float, desired: float, limits: Limits) -> float:
    # Quantize against the appliance's lower bound, then move by one step at most.
    maximum_steps = int((limits.max_setpoint - limits.min_setpoint) / limits.step + 1e-6)
    child = round((desired - limits.min_setpoint) / limits.step)
    desired = limits.min_setpoint + min(maximum_steps,max(0,child)) * limits.step
    index = round((current-limits.min_setpoint)/limits.step)
    if abs(current-(limits.min_setpoint+index*limits.step))>1e-4:
        return current
    moves = int((1+1e-6)/limits.step)
    destination = round((desired-limits.min_setpoint)/limits.step)
    offset = max(-moves,min(moves,destination-index))
    return round(limits.min_setpoint+(index+offset)*limits.step,3)

def decide(
    now: datetime, request: Request, data: Evidence, limits: Limits,
    history: History, regulated_setpoint: float | None = None, offset: float = 0,
) -> Decision:
    observed, quality = activity(now, data, limits)
    def result(status, command=None):
        return Decision(status, observed, quality, command)

    if not limits.authorized:
        return result("observing")
    if history.manual_latched or (history.manual_until and now < history.manual_until):
        return result("manual_hold")
    if not (data.adapter_available and data.room.fresh(now, limits.room_freshness, 0, 50)):
        return result("stale_feedback")
    if request.mode not in SUPPORTED_MODES or not 18 <= request.temperature <= 30:
        return result("invalid_target")
    if history.pending:
        window = now-history.pending.sent
        return result("confirmation_timeout" if window >= CONFIRM_TIMEOUT else "awaiting_evidence")
    if history.retry_after and now < history.retry_after:
        return result("communication_hold")
    if quality == "unknown":
        return result("uncertain_activity")
    mode = data.independent_mode or data.adapter_mode
    wanted = demand(request, data.room.value, mode)
    if wanted is None:
        return result("in_band")
    if history.last_command and (now-history.last_command.sent <
                                 (MIN_QUIET if request.quiet else MIN_DAY)):
        return result("rate_limit")
    if wanted != mode:
        if not limits.absolute_mode_verified or data.independent_mode is None:
            return result("mode_change_needs_independent_feedback")
        phase = now-data.phase_since if data.phase_since else timedelta(0)
        if mode in ACTIVE:
            if phase < MIN_RUN:
                return result("minimum_run")
            # Opposing mode changes always go via a verified absolute off.
            wanted = "off"
        elif mode == "off":
            if phase < MIN_OFF:
                return result("minimum_off")
            if (data.last_active_mode is None and data.history_ready_since is not None
                and now-data.history_ready_since < MIN_REVERSE):
                return result("unknown_reverse_history")
            if (data.last_active_mode in ACTIVE and data.last_active_mode != wanted
                and (data.last_active_start is None or now-data.last_active_start < MIN_REVERSE
                     or (data.history_ready_since is not None
                         and now-data.history_ready_since < MIN_REVERSE))):
                return result("reverse_lock")
        else:
            return result("unrecognized_mode")
        cmd = Command("mode",wanted)
        if history.last_command and history.last_command.command==cmd:
            return result("deduplicated")
        return result("ready",cmd)

    if wanted == "off":
        return result("off")
    if observed not in ACTIVE | {"load_detected"}:
        return result("runtime_not_established")
    reported = data.independent_setpoint if data.independent_mode in ACTIVE else (
        data.adapter_setpoint if limits.ir_setpoint_validated and data.proxy_report_fresh else None)
    if reported is None or regulated_setpoint is None:
        return result("missing_setpoint")
    target = step_toward(reported, regulated_setpoint+offset, limits)
    cmd = Command("setpoint",target)
    if target == reported or (history.last_command and history.last_command.command==cmd):
        return result("deduplicated")
    return result("ready",cmd)

def choose_group(decisions: list[tuple[str, Request, Evidence, Decision]]) -> str | None:
    """Allow checked stops during reversals and block opposing group starts."""
    proposed = [(key, req, data, decision) for key, req, data, decision in decisions
                if decision.command and data.room.value is not None]
    stops = [key for key, _, _, decision in proposed if decision.command == Command("mode", "off")]
    if stops:
        return stops[0]

    wanted = {demand(req, data.room.value, data.independent_mode or data.adapter_mode)
              for _, req, data, _ in proposed} & ACTIVE
    possible_active = {data.independent_mode or data.adapter_mode for _, _, data, decision in decisions
                       if decision.activity in ACTIVE | {"load_detected", "settling", "unknown"}} & ACTIVE
    if len(wanted | possible_active) > 1:
        return None
    return max(proposed, key=lambda item: abs(item[1].temperature-item[2].room.value),
               default=(None, None, None, None))[0]
