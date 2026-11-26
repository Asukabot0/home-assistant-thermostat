"""Offline mixed modes, parked-state recovery, echo and failed continuation."""
import unittest

from test_p0_p1_dispatch import DispatchTests


CASES = (
    "test_mixed_modes_inside_comfort_exit_the_opposite_side_gradually",
    "test_mixed_mode_shutdown_cannot_bypass_minimum_runtime",
    "test_normal_mode_change_needs_measured_post_shutdown_low_segment",
    "test_energy_off_persists_and_recovers_for_new_need_not_old_plan",
    "test_mode_success_then_initial_http_500_leaves_confirmed_baseline",
)


if __name__ == "__main__":
    suite = unittest.TestSuite(DispatchTests(name) for name in CASES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
