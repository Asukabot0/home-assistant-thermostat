"""Offline dispatch, final gating, replay protection and serialization."""
import unittest

from test_p0_p1_dispatch import DispatchTests


CASES = (
    "test_direct_high_advice_passes_real_gate_and_sends_once",
    "test_direct_high_rechecks_freshness_context_and_ignores_beep_budget",
    "test_high_advice_drops_when_live_power_falls_after_suggestion",
    "test_permission_removed_in_second_assessment_never_dispatches",
    "test_unconfirmed_prepared_is_locked_after_restart_without_duplicate",
    "test_new_feedback_blocks_cross_unit_escalation_and_allows_assist_later",
)


if __name__ == "__main__":
    suite = unittest.TestSuite(DispatchTests(name) for name in CASES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
