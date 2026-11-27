"""Join live, dated power reports to the durable Recorder evidence ledger."""
from model import numeric
from service import POWER_REPORT_GAP_SECONDS, power_evidence


LIVE_HISTORY_SECONDS = 2100


def attach(now, live, shadow, state):
    adapter = live["adapter"]
    context = live.get("power_context")
    at = adapter.get("power_report_at")
    if at is None:
        return shadow  # Historical replays whose feedback lacks exact live timestamps.
    if state.get("context") != context:
        state.clear()
        state.update(context=context, reports=[], evidence=None)
    if context is None:
        return dict(shadow, power_evidence=None)

    watts = numeric(adapter["power"])
    if watts is None or not 0 <= watts <= 10000:
        watts = None
    # Staleness is an explicit break. Never replace a previously seen valid
    # value by a current status that carries an older report timestamp.
    if 0 <= now-at <= POWER_REPORT_GAP_SECONDS:
        points = dict(state["reports"])
        if at not in points or watts is None:
            points[at] = watts
        state["reports"] = [(stamp, value) for stamp, value in sorted(points.items()) if now-stamp <= LIVE_HISTORY_SECONDS]
        # Persist invalid points as separators. Subsequent same-value polls are
        # deduplicated, so they neither accumulate time nor overwrite history.
    if not 0 <= now-at <= POWER_REPORT_GAP_SECONDS or watts is None:
        return dict(shadow, power_evidence=None)
    persisted = shadow.get("power_evidence")
    older = state.get("evidence")
    if persisted and persisted.get("context") == context and (older is None or
            (persisted.get("last_report") or 0) >= (older.get("last_report") or 0)):
        older = persisted
    edge = shadow.get("edges") or ()
    low = shadow.get("reversal_low_cutoff")
    thresholds = {"low": low, "above_low": low, "high": edge[-1] if edge else None,
                  "learning": shadow.get("low_power_cutoff")}
    # Two independent sources have recorded these exact source/epoch/report
    # pairs. When Recorder coalesces last_reported, a live pair survives.
    reports = dict(shadow.get("ledger_reports", ()))
    for stamp, value in state["reports"]:
        if stamp not in reports or value is None or reports[stamp] is not None:
            reports[stamp] = value
    if reports.get(at) != watts:
        return dict(shadow, power_evidence=None)
    evidence = power_evidence(now, reports.items(), context, thresholds, older)
    if evidence["last_report"] != at:
        return dict(shadow, power_evidence=None)
    state["evidence"] = evidence
    return dict(shadow, power_evidence=evidence, live_power_endpoint=at, live_power_reports=sorted(reports.items()))
