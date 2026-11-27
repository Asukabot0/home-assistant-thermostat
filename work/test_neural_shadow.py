"""Comprehensive offline regression tests for the neural-network thermostat shadow system.

Covers every required regression category:
1. Time leakage (future mutation invariance, embargoed forward-chaining split, train-only normalization).
2. Shared sensors (study sharing living sensor is not double-counted; future sensor mapping configurable).
3. Duplicate cache vs same-value new reports (temperature and power, int-truncated reconciliation).
4. Missing measurements (power gaps never zero energy; missing temp triggers abstention).
5. Mid-horizon action/target changes & strict factual vs counterfactual separation.
6. Model version mismatch & explicit lifecycle states (untrained / insufficient_data / uncalibrated / ready_to_use).
7. Exception & timeout isolation (runtime cycle + hourly telemetry export hook).
8. No actuation path (AST audit + compose security/network/RO mount audit + V1 start/stop/reversal prohibition).
9. Model architecture invariants (100k-300k params, non-negative monotone cumulative Wh, worst-zone comfort priority).
10. Production read-only SQLite ingestion + end-to-end synthetic pipeline verification.
"""
from __future__ import annotations

import ast
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import time
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "home-assistant"))
sys.path.insert(0, str(REPO_ROOT / "home-assistant" / "telemetry"))
sys.path.insert(0, str(REPO_ROOT / "work"))

import evidence as telemetry_evidence  # noqa: E402
from neural_shadow import DataOrigin, FEATURE_VERSION, MODEL_ARCH_VERSION, ModelState  # noqa: E402
from neural_shadow.config import build_site_config, load_default_example_config  # noqa: E402
from neural_shadow.constraints import generate_v1_candidates, validate_v1_action_proposal  # noqa: E402
from neural_shadow.data_pipeline import (  # noqa: E402
    ActionEvent,
    PowerReport,
    ProxyStateEvent,
    TempReport,
    _HistoryIndex,
    _finalize_canonical,
    build_dataset,
    build_feature_spec,
    deduplicate_temp_reports,
    extract_history_window,
    ingest_production_readonly,
    integrate_power_interval,
)
from neural_shadow.nn_model import (  # noqa: E402
    FeatureNormalizer,
    ModelArtifact,
    ModelVersionMismatchError,
    build_untrained_artifact,
    load_checkpoint,
    predict_numpy,
    save_checkpoint,
)
from neural_shadow.runtime import ShadowRuntime, ShadowRuntimeConfig  # noqa: E402
from neural_shadow.scoring import ScoringPolicy, evaluate_and_rank_candidates, score_candidate_prediction  # noqa: E402
from neural_shadow.synthetic import generate_synthetic_history  # noqa: E402
from neural_shadow.train import ReadinessGates, forward_chain_split, train_and_evaluate  # noqa: E402
from run_neural_shadow_e2e import run_e2e  # noqa: E402


def _make_clean_history(site, *, base_ts=1_800_000_000.0, minutes=300, mid_action_at=None, mid_target_at=None, same_value_temp=False):
    """Deterministic helper constructing a clean canonical history for focused regression tests."""
    raw_temp = []
    raw_env = []
    raw_power = []
    proxy_by_unit = {u: [] for u in site.unit_names}
    actions = []

    for m in range(minutes + 1):
        t = base_ts + m * 60.0
        # Outdoor every 10m
        if m % 10 == 0:
            raw_env.append(
                TempReport("outdoor_reference", "environment", "outdoor_reference", t, 18.0, t, "fresh", is_exact_precision=True)
            )
        # Independent zones every 5m
        if m % 5 == 0:
            for z_idx, z in enumerate(site.zones):
                val = 24.0 if same_value_temp else round(23.5 + 0.01 * (m % 30) + 0.2 * z_idx, 2)
                raw_temp.append(
                    TempReport(z.name, "zone_primary", z.primary_entity, t + 0.2, val, t + 0.5, "fresh", is_exact_precision=True)
                )
        for u_idx, act in enumerate(site.actuators):
            raw_power.append(PowerReport(act.name, act.power_entity, "ep1", t + 10.0, 220.0 + 30.0 * u_idx))
            mode = "heat" if u_idx == 0 else "cool" if u_idx == 1 else "off"
            sp = 23.0 if mode == "heat" else 24.0
            if mid_action_at is not None and t >= mid_action_at and u_idx == 0:
                sp = 24.0
            tgt = 24.0
            if mid_target_at is not None and t >= mid_target_at:
                tgt = 25.0
            proxy_by_unit[act.name].append(ProxyStateEvent(act.name, t, mode, sp, tgt, source_fresh=True))

    if mid_action_at is not None:
        actions.append(
            ActionEvent(
                at=mid_action_at,
                sent_at=mid_action_at - 2.0,
                unit=site.unit_names[0],
                kind="temperature",
                before_mode="heat",
                before_setpoint=23.0,
                after_mode="heat",
                after_setpoint=24.0,
                source="controller",
                receipt_status="proxy_echo",
                physical_verified=False,
                quality_flags=("proxy_only_unverified",),
            )
        )

    return _finalize_canonical(
        site=site,
        data_origin=DataOrigin.SYNTHETIC.value,
        source_description="unit_test_fixture",
        raw_temp=raw_temp,
        raw_env=raw_env,
        raw_power=raw_power,
        proxy_by_unit=proxy_by_unit,
        actions=actions,
        config_versions=[],
        quality_counts={},
    )


class NeuralShadowRegressionTests(unittest.TestCase):
    def setUp(self):
        self.site = load_default_example_config()
        self.spec = build_feature_spec(self.site)

    # 1. Model size and physical energy monotonicity invariants
    def test_parameter_count_and_energy_invariants(self):
        art = build_untrained_artifact(self.site, self.spec, seed=7)
        pcount = art.model.parameter_count()
        self.assertGreaterEqual(pcount, 100_000)
        self.assertLessEqual(pcount, 300_000)
        self.assertEqual(art.model_state, ModelState.UNTRAINED)

        rng = np.random.default_rng(123)
        xh = rng.normal(0.0, 2.0, size=(16, self.spec.n_hist_steps, self.spec.hist_dim)).astype(np.float32)
        xa = rng.normal(0.0, 2.0, size=(16, self.spec.n_future_steps, self.spec.act_dim)).astype(np.float32)
        out = predict_numpy(art, xh, xa)

        # Non-negative Wh, non-decreasing cumulative Wh across 15m/30m/60m, total == sum(units)
        self.assertTrue(np.all(out["unit_wh"] >= 0.0))
        self.assertTrue(np.all(out["total_wh"] >= 0.0))
        self.assertTrue(np.all(np.diff(out["unit_wh"], axis=-1) >= -1e-6))
        self.assertTrue(np.all(np.diff(out["total_wh"], axis=-1) >= -1e-6))
        self.assertTrue(np.allclose(out["total_wh"], out["unit_wh"].sum(axis=1), atol=1e-4))
        self.assertTrue(np.all(out["temp_sigma"] >= 0.05 - 1e-6))

    # 2. Time leakage regression
    def test_time_leakage_future_mutation_and_embargo_and_train_only_normalizer(self):
        hist_a = generate_synthetic_history(self.site, hours=24.0, seed=11)
        t0 = float(hist_a.zone_temp_reports["sleep"][len(hist_a.zone_temp_reports["sleep"]) // 2].report_at)
        t0 = float(int(t0 // 300) * 300)

        # Mutate ALL measurements strictly AFTER t0 by dramatic offsets
        hist_b = deepcopy(hist_a)
        for z in hist_b.zone_temp_reports:
            hist_b.zone_temp_reports[z] = [
                TempReport(r.zone, r.role, r.source_entity, r.report_at, r.value_c + 20.0, r.observed_at, r.quality)
                if r.report_at > t0 or r.observed_at > t0
                else r
                for r in hist_b.zone_temp_reports[z]
            ]
        for e in hist_b.env_temp_reports:
            hist_b.env_temp_reports[e] = [
                TempReport(r.zone, r.role, r.source_entity, r.report_at, r.value_c + 30.0, r.observed_at, r.quality)
                if r.report_at > t0 or r.observed_at > t0
                else r
                for r in hist_b.env_temp_reports[e]
            ]
        for u in hist_b.power_reports:
            hist_b.power_reports[u] = [
                PowerReport(r.unit, r.source_entity, r.observer_epoch, r.report_at, 9999.0)
                if r.report_at > t0
                else r
                for r in hist_b.power_reports[u]
            ]

        xh_a, meta_a = extract_history_window(_HistoryIndex(hist_a), self.spec, t0)
        xh_b, meta_b = extract_history_window(_HistoryIndex(hist_b), self.spec, t0)
        np.testing.assert_array_equal(np.nan_to_num(xh_a, nan=-999.0), np.nan_to_num(xh_b, nan=-999.0))
        self.assertEqual(meta_a["zone_temps"], meta_b["zone_temps"])
        self.assertEqual(meta_a["unit_watts"], meta_b["unit_watts"])
        self.assertEqual(meta_a["env_temps"], meta_b["env_temps"])

        # Forward-chaining split embargo check (60m label horizon cannot cross into val or test)
        ds = build_dataset(hist_a)
        split = forward_chain_split(ds.cutoff_ts, max_horizon_seconds=3600.0)
        self.assertGreater(split.train_idx.size, 0)
        self.assertGreater(split.val_idx.size, 0)
        self.assertGreater(split.test_idx.size, 0)
        self.assertLessEqual(float(ds.cutoff_ts[split.train_idx].max()) + 3600.0, float(ds.cutoff_ts[split.val_idx].min()) + 1e-5)
        self.assertLessEqual(float(ds.cutoff_ts[split.val_idx].max()) + 3600.0, float(ds.cutoff_ts[split.test_idx].min()) + 1e-5)

        # Normalizer fitted on train split MUST be completely unaffected by mutating test split by +100
        norm_1 = FeatureNormalizer.fit(
            ds.X_hist[split.train_idx],
            ds.X_act_actual[split.train_idx],
            [ds.metadata[int(i)] for i in split.train_idx],
            ds.spec,
        )
        ds_mut = deepcopy(ds)
        ds_mut.X_hist[split.test_idx] += 100.0
        norm_2 = FeatureNormalizer.fit(
            ds_mut.X_hist[split.train_idx],
            ds_mut.X_act_actual[split.train_idx],
            [ds_mut.metadata[int(i)] for i in split.train_idx],
            ds_mut.spec,
        )
        np.testing.assert_array_equal(norm_1.hist_mean, norm_2.hist_mean)
        np.testing.assert_array_equal(norm_1.hist_std, norm_2.hist_std)
        np.testing.assert_array_equal(norm_1.hist_fill, norm_2.hist_fill)

    # 3. Shared sensors & future sensor mapping configurability
    def test_shared_sensor_not_double_counted_and_future_sensors_configurable(self):
        # Default topology: 3 units (bedroom, living, study), ONLY 2 independent temp labels (sleep, lounge)
        self.assertEqual(self.site.zone_names, ("sleep", "lounge"))
        self.assertEqual(self.site.unit_names, ("bedroom", "living", "study"))
        self.assertEqual(self.site.actuator_by_name("study").home_zone, "lounge")
        self.assertEqual(
            self.site.actuator_by_name("study").temperature_entity,
            self.site.actuator_by_name("living").temperature_entity,
        )

        hist = _make_clean_history(self.site, minutes=180)
        ds = build_dataset(hist)
        # Exactly 2 independent temperature targets, NEVER 3!
        self.assertEqual(ds.y_temp_delta.shape[1], 2)
        self.assertEqual(ds.y_unit_wh.shape[1], 3)

        # Now enable future office_primary (3rd independent zone) and future balcony outdoor sensor
        rooms = deepcopy(self.site.raw_rooms)
        rooms["study"]["temperature"] = "sensor.example_office_temperature"
        topo = deepcopy(self.site.raw_topology)
        topo["observation_version"] = "site-2-with-office-and-balcony"
        topo["sensors"]["office_primary"]["enabled"] = True
        topo["sensors"]["balcony"]["enabled"] = True
        topo["zones"]["office"] = {"primary": "office_primary", "fusion": "primary", "target": {"actuator": "study"}}
        topo["actuators"]["study"]["zone"] = "office"
        topo["actuators"]["study"]["influences"] = {"office": 1.0, "lounge": 0.45, "sleep": 0.25}
        site2 = build_site_config(rooms, topo, self.site.policy)
        self.assertEqual(len(site2.zone_names), 3)
        self.assertIn("office", site2.zone_names)
        self.assertIn("balcony", [s.name for s in site2.active_environment_sensors])
        spec2 = build_feature_spec(site2)
        self.assertEqual(spec2.n_zones, 3)
        self.assertIn("balcony", spec2.env_sensor_names)
        self.assertNotEqual(site2.config_fingerprint, self.site.config_fingerprint)

    # 4. Duplicate cache vs same-value new reports (temperature & power)
    def test_duplicate_cache_vs_same_value_new_report(self):
        base = 1_800_000_000.0
        # 1 real report at t=base+0.3, plus 10 minute-level cache copies (int-truncated) with same report_at,
        # plus 1 genuinely NEW report 15m later at t=base+900.4 with the EXACT SAME value 24.0C!
        raw_reps = [
            TempReport("sleep", "zone_primary", "sensor.bed", base + 0.3, 24.0, base + 1.0, is_exact_precision=True),
            *[
                TempReport("sleep", "zone_primary", "sensor.bed", float(int(base)), 24.0, base + 60.0 * m, is_exact_precision=False)
                for m in range(1, 11)
            ],
            TempReport("sleep", "zone_primary", "sensor.bed", base + 900.4, 24.0, base + 901.0, is_exact_precision=True),
        ]
        deduped, skipped = deduplicate_temp_reports(raw_reps)
        self.assertEqual(len(deduped), 2)  # NOT 11 and NOT 1!
        self.assertEqual(skipped, 10)
        self.assertAlmostEqual(deduped[0].report_at, base + 0.3, places=4)
        self.assertAlmostEqual(deduped[1].report_at, base + 900.4, places=4)
        self.assertEqual(deduped[0].value_c, 24.0)
        self.assertEqual(deduped[1].value_c, 24.0)

        # Verify in causal feature extraction: cache copies do NOT trigger new_temp_report or future settlement
        hist_cache_only = _finalize_canonical(
            site=self.site,
            data_origin=DataOrigin.SYNTHETIC.value,
            source_description="cache_test",
            raw_temp=[
                TempReport("sleep", "zone_primary", self.site.zone_by_name("sleep").primary_entity, base + 0.3, 24.0, base + 1.0, is_exact_precision=True),
                *[
                    TempReport("sleep", "zone_primary", self.site.zone_by_name("sleep").primary_entity, float(int(base)), 24.0, base + 60.0 * m, is_exact_precision=False)
                    for m in range(1, 20)
                ],
            ],
            raw_env=[],
            raw_power=[],
            proxy_by_unit={},
            actions=[],
            config_versions=[],
            quality_counts={},
        )
        idx_cache = _HistoryIndex(hist_cache_only)
        # Future settlement at +15m MUST return None when only cached copies exist!
        self.assertIsNone(idx_cache.settle_future_temp("sleep", base + 60.0, base + 0.3, 900.0))

        # Now add a genuinely NEW report at +15m with the EXACT SAME value 24.0C
        hist_same_val = deepcopy(hist_cache_only)
        hist_same_val.zone_temp_reports["sleep"].append(
            TempReport("sleep", "zone_primary", self.site.zone_by_name("sleep").primary_entity, base + 960.2, 24.0, base + 961.0, is_exact_precision=True)
        )
        idx_same = _HistoryIndex(hist_same_val)
        settled_rep = idx_same.settle_future_temp("sleep", base + 60.0, base + 0.3, 900.0)
        self.assertIsNotNone(settled_rep)
        self.assertEqual(settled_rep.value_c, 24.0)
        self.assertGreater(settled_rep.report_at, base + 0.3)

        # Power: repeated cache at same report_at = 0 coverage (wh=None); same-value new report = covered Wh!
        p_cache = [PowerReport("bedroom", "sensor.p", "ep1", base, 300.0)] * 5
        int_cache = integrate_power_interval(p_cache, base, base + 60.0)
        self.assertIsNone(int_cache.wh)
        self.assertEqual(int_cache.covered_seconds, 0.0)
        self.assertTrue(int_cache.has_gap)

        p_new_same = [
            PowerReport("bedroom", "sensor.p", "ep1", base, 300.0),
            PowerReport("bedroom", "sensor.p", "ep1", base + 60.0, 300.0),
        ]
        int_new = integrate_power_interval(p_new_same, base, base + 60.0)
        self.assertAlmostEqual(int_new.wh, 5.0, places=4)
        self.assertEqual(int_new.coverage_ratio, 1.0)
        self.assertFalse(int_new.has_gap)

    # 5. Missing measurements & power gap never zero energy
    def test_missing_measurements_power_gaps_never_zero_energy_and_abstention(self):
        base = 1_800_000_000.0
        # Gap > 120s (at t=60 -> t=360) and explicit None separator
        reps = [
            PowerReport("bedroom", "sensor.p", "ep1", base, 600.0),
            PowerReport("bedroom", "sensor.p", "ep1", base + 60.0, 600.0),
            PowerReport("bedroom", "sensor.p", "ep1", base + 120.0, None),
            PowerReport("bedroom", "sensor.p", "ep1", base + 400.0, 600.0),
            PowerReport("bedroom", "sensor.p", "ep1", base + 460.0, 600.0),
        ]
        res = integrate_power_interval(reps, base, base + 600.0)
        # Only [0, 60] and [400, 460] = 120s covered out of 600s (20%)
        self.assertAlmostEqual(res.covered_seconds, 120.0, places=3)
        self.assertAlmostEqual(res.coverage_ratio, 0.2, places=3)
        self.assertTrue(res.has_gap)
        self.assertAlmostEqual(res.gap_seconds, 480.0, places=3)
        self.assertAlmostEqual(res.wh, 600.0 * 120.0 / 3600.0, places=4)

        # Completely missing interval -> wh is None, NEVER 0.0!
        res_empty = integrate_power_interval([], base, base + 900.0)
        self.assertIsNone(res_empty.wh)
        self.assertEqual(res_empty.coverage_ratio, 0.0)
        self.assertTrue(res_empty.has_gap)

        # Missing zone temp triggers runtime abstention even if model is READY_TO_USE
        hist = _make_clean_history(self.site, minutes=200)
        # Drop all sleep temp reports so sleep zone is missing
        hist.zone_temp_reports["sleep"] = []
        art = build_untrained_artifact(self.site, self.spec)
        art.model_state = ModelState.READY_TO_USE
        art.metadata["model_state"] = ModelState.READY_TO_USE.value
        art.metadata["training_cutoff_ts"] = base
        art.normalizer = FeatureNormalizer(
            hist_fill=np.zeros(self.spec.hist_dim, dtype=np.float32),
            hist_mean=np.zeros(self.spec.hist_dim, dtype=np.float32),
            hist_std=np.ones(self.spec.hist_dim, dtype=np.float32),
            act_mean=np.zeros(self.spec.act_dim, dtype=np.float32),
            act_std=np.ones(self.spec.act_dim, dtype=np.float32),
            coverage_envelope={},
        )
        with TemporaryDirectory() as tmp:
            rt = ShadowRuntime(self.site, ShadowRuntimeConfig(Path(tmp) / "st", Path(tmp) / "out"), art)
            sealed = rt.seal_prediction(hist, base + 120 * 60.0)
            self.assertTrue(sealed["declined"])
            self.assertTrue(any("missing_zone_observation:sleep" in r for r in sealed["decline_reasons"]))

    # B1 regression: partial electrical coverage (>= 0.85) must NEVER be scored as full-interval Wh
    def test_b1_partial_power_coverage_masked_not_zero_filled(self):
        base = 1_800_000_000.0
        cutoff_gap = base + 120 * 60.0  # [cutoff_gap, cutoff_gap + 900] has a 130s gap on bedroom
        cutoff_full = base + 180 * 60.0  # [cutoff_full, cutoff_full + 900] has 100% coverage
        hist = _make_clean_history(self.site, base_ts=base, minutes=260)

        # Replace bedroom power stream with constant 1000W and a single 130s reporting gap inside [cutoff_gap, cutoff_gap + 900]
        bed_entity = self.site.actuator_by_name("bedroom").power_entity
        bed_power = []
        t = base
        while t <= base + 260 * 60.0:
            # Skip report at cutoff_gap + 360 and emit next at cutoff_gap + 430 -> 130s gap [300, 430]
            if abs(t - (cutoff_gap + 360.0)) < 1e-6:
                t += 60.0
                continue
            if abs(t - (cutoff_gap + 420.0)) < 1e-6:
                bed_power.append(PowerReport("bedroom", bed_entity, "ep1", cutoff_gap + 430.0, 1000.0))
                t += 60.0
                continue
            bed_power.append(PowerReport("bedroom", bed_entity, "ep1", t, 1000.0))
            t += 60.0
        hist.power_reports["bedroom"] = sorted(bed_power, key=lambda r: r.report_at)

        # Direct integration check: coverage ~ 770/900 = 0.8556, covered_wh ~ 213.889, full_interval_wh MUST be None
        pint_gap = integrate_power_interval(hist.power_reports["bedroom"], cutoff_gap, cutoff_gap + 900.0)
        self.assertAlmostEqual(pint_gap.coverage_ratio, 770.0 / 900.0, places=4)
        self.assertTrue(pint_gap.has_gap)
        self.assertAlmostEqual(pint_gap.gap_seconds, 130.0, places=1)
        self.assertAlmostEqual(pint_gap.wh, 1000.0 * 770.0 / 3600.0, places=3)
        self.assertIsNone(pint_gap.full_interval_wh)

        # Complete coverage window: exact 250.0 Wh
        pint_full = integrate_power_interval(hist.power_reports["bedroom"], cutoff_full, cutoff_full + 900.0)
        self.assertFalse(pint_full.has_gap)
        self.assertAlmostEqual(pint_full.coverage_ratio, 1.0, places=6)
        self.assertAlmostEqual(pint_full.full_interval_wh, 250.0, places=3)

        # Dataset labels MUST mask the 0.856-covered window even if min_wh_coverage=0.85 is requested
        for req_cov in (1.0, 0.85):
            ds = build_dataset(hist, min_wh_coverage=req_cov)
            idx_gap = int(np.where(np.isclose(ds.cutoff_ts, cutoff_gap, rtol=0.0, atol=1e-3))[0][0])
            idx_full = int(np.where(np.isclose(ds.cutoff_ts, cutoff_full, rtol=0.0, atol=1e-3))[0][0])
            bed_u = ds.spec.unit_names.index("bedroom")
            h15 = ds.spec.horizons_minutes.index(15)

            self.assertAlmostEqual(float(ds.y_unit_cov[idx_gap, bed_u, h15]), 770.0 / 900.0, places=4)
            self.assertFalse(bool(ds.y_unit_valid[idx_gap, bed_u, h15]))
            self.assertTrue(np.isnan(ds.y_unit_wh[idx_gap, bed_u, h15]))
            self.assertFalse(bool(ds.y_total_valid[idx_gap, h15]))
            self.assertTrue(np.isnan(ds.y_total_wh[idx_gap, h15]))

            self.assertTrue(bool(ds.y_unit_valid[idx_full, bed_u, h15]))
            self.assertAlmostEqual(float(ds.y_unit_wh[idx_full, bed_u, h15]), 250.0, places=3)
            self.assertTrue(bool(ds.y_total_valid[idx_full, h15]))

        # Runtime settlement MUST also mask partial coverage (actual_total_wh=None, factual_hold_wh_error=None)
        art = build_untrained_artifact(self.site, self.spec)
        with TemporaryDirectory() as tmp:
            rt = ShadowRuntime(
                self.site,
                ShadowRuntimeConfig(Path(tmp) / "st", Path(tmp) / "out", min_wh_label_coverage=0.85),
                art,
            )
            p_gap = rt.seal_prediction(hist, cutoff_gap, sealed_at=cutoff_gap + 1.0)
            p_full = rt.seal_prediction(hist, cutoff_full, sealed_at=cutoff_full + 1.0)
            settled = rt.settle_predictions(hist, now_ts=cutoff_full + 4500.0)
            by_key = {(r["prediction_id"], r["horizon_minutes"]): r for r in settled}

            row_gap = by_key[(p_gap["prediction_id"], 15)]
            self.assertTrue(row_gap["actual_units_wh"]["bedroom"]["has_gap"])
            self.assertFalse(row_gap["actual_units_wh"]["bedroom"]["label_valid"])
            self.assertIsNone(row_gap["actual_units_wh"]["bedroom"]["wh"])
            self.assertAlmostEqual(row_gap["actual_units_wh"]["bedroom"]["covered_wh_raw"], 1000.0 * 770.0 / 3600.0, places=2)
            self.assertAlmostEqual(row_gap["joint_wh_coverage_ratio"], 770.0 / 900.0, places=4)
            self.assertTrue(row_gap["joint_wh_has_gap"])
            self.assertFalse(row_gap["joint_wh_label_valid"])
            self.assertIsNone(row_gap["actual_total_wh"])
            self.assertIsNone(row_gap["factual_hold_wh_error"])

            row_full = by_key[(p_full["prediction_id"], 15)]
            self.assertFalse(row_full["joint_wh_has_gap"])
            self.assertTrue(row_full["joint_wh_label_valid"])
            self.assertAlmostEqual(row_full["actual_units_wh"]["bedroom"]["wh"], 250.0, places=2)
            self.assertIsNotNone(row_full["actual_total_wh"])
            self.assertIsNotNone(row_full["factual_hold_wh_error"])

    # 6. Mid-horizon action/target changes & strict factual vs counterfactual separation
    def test_mid_horizon_action_and_target_changes_and_counterfactual_separation(self):
        base = 1_800_000_000.0
        cutoff = base + 180 * 60.0
        art = build_untrained_artifact(self.site, self.spec)
        art.model_state = ModelState.READY_TO_USE
        art.metadata["model_state"] = ModelState.READY_TO_USE.value
        art.metadata["training_cutoff_ts"] = base
        art.normalizer = FeatureNormalizer(
            hist_fill=np.zeros(self.spec.hist_dim, dtype=np.float32),
            hist_mean=np.zeros(self.spec.hist_dim, dtype=np.float32),
            hist_std=np.ones(self.spec.hist_dim, dtype=np.float32),
            act_mean=np.zeros(self.spec.act_dim, dtype=np.float32),
            act_std=np.ones(self.spec.act_dim, dtype=np.float32),
            coverage_envelope={},
        )

        # Case A: mid-horizon action at cutoff + 20m -> 15m is factual_hold, 30m & 60m MUST be intervened_hold_unscored!
        hist_act = _make_clean_history(self.site, base_ts=base, minutes=260, mid_action_at=cutoff + 20 * 60.0)
        with TemporaryDirectory() as tmp:
            rt = ShadowRuntime(self.site, ShadowRuntimeConfig(Path(tmp) / "st", Path(tmp) / "out"), art)
            sealed = rt.seal_prediction(hist_act, cutoff, sealed_at=cutoff + 1.0)
            # Sealed BEFORE settlement with tamper-evident digest
            self.assertIn("seal_sha256", sealed)
            self.assertEqual(sealed["cutoff_ts"], cutoff)
            self.assertIn("zones", sealed["true_report_times"])

            # Force a non-hold recommendation on a second sealed record to test counterfactual tagging
            settled = rt.settle_predictions(hist_act, now_ts=cutoff + 4500.0)
            by_h = {r["horizon_minutes"]: r for r in settled}
            self.assertFalse(by_h[15]["intervened"])
            self.assertEqual(by_h[15]["factual_settlement_kind"], "factual_hold")
            self.assertTrue(len(by_h[15]["factual_hold_temp_errors"]) == 2)
            self.assertIsNotNone(by_h[15]["factual_hold_wh_error"])

            # 30m and 60m occurred AFTER the mid-horizon action -> MUST NOT score hold plan!
            for h in (30, 60):
                self.assertTrue(by_h[h]["intervened"])
                self.assertEqual(by_h[h]["factual_settlement_kind"], "intervened_hold_unscored")
                self.assertEqual(by_h[h]["factual_hold_temp_errors"], {})
                self.assertIsNone(by_h[h]["factual_hold_wh_error"])

        # Case B: mid-horizon target change at cutoff + 10m -> all horizons 15/30/60m marked intervened!
        hist_tgt = _make_clean_history(self.site, base_ts=base, minutes=260, mid_target_at=cutoff + 10 * 60.0)
        with TemporaryDirectory() as tmp:
            rt = ShadowRuntime(self.site, ShadowRuntimeConfig(Path(tmp) / "st", Path(tmp) / "out"), art)
            rt.seal_prediction(hist_tgt, cutoff, sealed_at=cutoff + 1.0)
            settled = rt.settle_predictions(hist_tgt, now_ts=cutoff + 4500.0)
            for row in settled:
                self.assertTrue(row["intervened"])
                self.assertEqual(row["factual_settlement_kind"], "intervened_hold_unscored")
                self.assertEqual(row["factual_hold_temp_errors"], {})

        # Case C: counterfactual recommendation (not executed) stays counterfactual_unexecuted
        hist_clean = _make_clean_history(self.site, base_ts=base, minutes=260)
        with TemporaryDirectory() as tmp:
            rt = ShadowRuntime(self.site, ShadowRuntimeConfig(Path(tmp) / "st", Path(tmp) / "out"), art)
            sealed = rt.seal_prediction(hist_clean, cutoff, sealed_at=cutoff + 1.0)
            # Simulate a sealed prediction whose recommended_candidate_id was a non-hold action that never ran
            with closing(sqlite3.connect(rt.db_path)) as db, db:
                row = db.execute("SELECT payload_json FROM sealed_predictions WHERE prediction_id=?", (sealed["prediction_id"],)).fetchone()
                p = json.loads(row[0])
                p.pop("seal_sha256")
                p["recommended_candidate_id"] = "bedroom:heat:23->24"
                p["declined"] = False
                from neural_shadow.config import canonical_hash
                new_sig = canonical_hash(p)
                p["seal_sha256"] = new_sig
                db.execute(
                    "UPDATE sealed_predictions SET recommended_candidate_id=?, declined=0, seal_sha256=?, payload_json=? WHERE prediction_id=?",
                    ("bedroom:heat:23->24", new_sig, json.dumps(p, sort_keys=True), sealed["prediction_id"]),
                )
            settled = rt.settle_predictions(hist_clean, now_ts=cutoff + 4500.0)
            for row in settled:
                self.assertEqual(row["factual_settlement_kind"], "factual_hold")
                self.assertEqual(row["recommended_execution_status"], "counterfactual_unexecuted")

    # 7. Model version mismatch & explicit model states
    def test_model_version_mismatch_and_lifecycle_states(self):
        art = build_untrained_artifact(self.site, self.spec)
        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "model.pt"
            save_checkpoint(art, ckpt)

            # Random weights MUST remain UNTRAINED and declined at runtime
            loaded = load_checkpoint(ckpt, expected_site=self.site, expected_spec=self.spec, strict=True)
            self.assertEqual(loaded.model_state, ModelState.UNTRAINED)
            hist = _make_clean_history(self.site, minutes=200)
            rt = ShadowRuntime(self.site, ShadowRuntimeConfig(Path(tmp) / "st", Path(tmp) / "out"), loaded)
            sealed = rt.seal_prediction(hist, 1_800_000_000.0 + 180 * 60.0)
            self.assertTrue(sealed["declined"])
            self.assertIn("model_state_not_ready:untrained", sealed["decline_reasons"])

            # Refusing to package random weights as READY_TO_USE without normalizer/training cutoff
            bad_art = build_untrained_artifact(self.site, self.spec)
            bad_art.model_state = ModelState.READY_TO_USE
            with self.assertRaises(ValueError):
                save_checkpoint(bad_art, Path(tmp) / "bad.pt")

            # Version mismatch: wrong feature_version, arch_version, or config_fingerprint -> ModelVersionMismatchError
            with self.assertRaises(ModelVersionMismatchError):
                load_checkpoint(ckpt, expected_site=self.site, expected_spec=self.spec, expected_feature_version="ns-feat-v999")
            with self.assertRaises(ModelVersionMismatchError):
                load_checkpoint(ckpt, expected_site=self.site, expected_spec=self.spec, expected_arch_version="ns-gru-v999")

            # Modified site topology fingerprint -> ModelVersionMismatchError
            topo2 = deepcopy(self.site.raw_topology)
            topo2["observation_version"] = "site-changed"
            site2 = build_site_config(self.site.raw_rooms, topo2, self.site.policy)
            with self.assertRaises(ModelVersionMismatchError):
                load_checkpoint(ckpt, expected_site=site2, expected_spec=self.spec, strict=True)

            # Short dataset -> INSUFFICIENT_DATA
            short_hist = generate_synthetic_history(self.site, hours=10.0, seed=5)
            short_ds = build_dataset(short_hist)
            short_art, _ = train_and_evaluate(short_ds, epochs=2, seed=5, gates=ReadinessGates())
            self.assertEqual(short_art.model_state, ModelState.INSUFFICIENT_DATA)

    # 8. V1 candidates hard constraints & worst-zone comfort priority (no whole-house average cancellation!)
    def test_v1_candidates_and_worst_zone_comfort_priority(self):
        meta = {
            "cutoff_ts": 1_800_010_000.0,
            "zone_temps": {"sleep": 22.8, "lounge": 25.2},  # -1.2C over-cool in bedroom, +1.2C over-heat in living!
            "zone_targets": {"sleep": 24.0, "lounge": 24.0},
            "zone_report_at": {"sleep": 1_800_009_900.0, "lounge": 1_800_009_900.0},
            "zone_age_s": {"sleep": 100.0, "lounge": 100.0},
            "zone_paused": {"sleep": False, "lounge": False},
            "env_temps": {"outdoor_reference": 22.0},
            "env_report_at": {"outdoor_reference": 1_800_009_500.0},
            "unit_modes": {"bedroom": "cool", "living": "heat", "study": "off"},
            "unit_setpoints": {"bedroom": 23.0, "living": 23.0, "study": 24.0},
            "unit_watts": {"bedroom": 320.0, "living": 350.0, "study": 2.0},
            "unit_power_report_at": {"bedroom": 1_800_009_950.0, "living": 1_800_009_950.0, "study": 1_800_009_950.0},
            "unit_holds": {u: {"manual_hold": False, "failure_hold": False, "isolated": False, "quiet": False} for u in self.site.unit_names},
            "unit_last_action_at": {"bedroom": None, "living": None, "study": None},
            "quality_flags": {},
        }
        cands, rejs = generate_v1_candidates(self.site, self.spec, meta)
        # study is OFF -> must NEVER get start candidate; no stop or mode reversal candidates ever!
        for c in cands:
            self.assertIn(c.kind, {"hold", "same_mode_setpoint"})
            if c.kind == "same_mode_setpoint":
                self.assertNotEqual(c.unit, "study")
                self.assertEqual(c.mode, meta["unit_modes"][c.unit])
                self.assertEqual(abs(c.step_c), 1.0)
        # Opposing mode active: strengthening bedroom cool (23->22) or living heat (23->24) is blocked; weakening allowed!
        cand_ids = {c.candidate_id for c in cands}
        self.assertIn("hold", cand_ids)
        self.assertIn("bedroom:cool:23->24", cand_ids)  # weakening cool is allowed
        self.assertIn("living:heat:23->22", cand_ids)  # weakening heat is allowed
        self.assertNotIn("bedroom:cool:23->22", cand_ids)  # blocked by opposing mode / peer guard
        self.assertNotIn("living:heat:23->24", cand_ids)

        # Direct check: start, stop, reversal, ±2C are all strictly rejected
        self.assertFalse(validate_v1_action_proposal(self.site, meta, unit="study", proposed_mode="cool", proposed_setpoint=24.0)[0])
        self.assertFalse(validate_v1_action_proposal(self.site, meta, unit="bedroom", proposed_mode="off", proposed_setpoint=23.0)[0])
        self.assertFalse(validate_v1_action_proposal(self.site, meta, unit="bedroom", proposed_mode="heat", proposed_setpoint=23.0)[0])
        self.assertFalse(validate_v1_action_proposal(self.site, meta, unit="bedroom", proposed_mode="cool", proposed_setpoint=25.0)[0])

        # Worst-zone comfort priority: bedroom -1.2C + living +1.2C MUST NOT cancel out!
        policy = ScoringPolicy()
        opposed_score = score_candidate_prediction(
            self.spec,
            {"sleep": 22.8, "lounge": 25.2},
            {"sleep": 24.0, "lounge": 24.0},
            np.zeros((2, 3), dtype=np.float32),
            np.full((2, 3), 0.1, dtype=np.float32),
            np.full((3, 3), 50.0, dtype=np.float32),
            np.full((3,), 150.0, dtype=np.float32),
            np.full((3,), 10.0, dtype=np.float32),
            policy=policy,
        )
        balanced_score = score_candidate_prediction(
            self.spec,
            {"sleep": 24.0, "lounge": 24.0},
            {"sleep": 24.0, "lounge": 24.0},
            np.zeros((2, 3), dtype=np.float32),
            np.full((2, 3), 0.1, dtype=np.float32),
            np.full((3, 3), 50.0, dtype=np.float32),
            np.full((3,), 150.0, dtype=np.float32),
            np.full((3,), 10.0, dtype=np.float32),
            policy=policy,
        )
        self.assertGreater(opposed_score["worst_zone_discomfort"], 4.0)
        self.assertAlmostEqual(balanced_score["worst_zone_discomfort"], 0.0, places=5)

        # Comparable comfort tie-breaking selects lower 3-unit Wh; comfort improvement beats lower Wh!
        # Mock predict_numpy to verify ranking behavior cleanly.
        from unittest.mock import patch

        art = build_untrained_artifact(self.site, self.spec)
        art.model_state = ModelState.READY_TO_USE
        art.metadata["model_state"] = ModelState.READY_TO_USE.value
        art.normalizer = FeatureNormalizer(
            hist_fill=np.zeros(self.spec.hist_dim, dtype=np.float32),
            hist_mean=np.zeros(self.spec.hist_dim, dtype=np.float32),
            hist_std=np.ones(self.spec.hist_dim, dtype=np.float32),
            act_mean=np.zeros(self.spec.act_dim, dtype=np.float32),
            act_std=np.ones(self.spec.act_dim, dtype=np.float32),
            coverage_envelope={},
        )
        # 3 candidates: [hold, bedroom:cool:23->24, living:heat:23->22]
        # Scenario 1: living:heat:23->22 dramatically improves comfort (even with higher Wh) -> MUST win!
        mock_out = {
            "temp_delta": np.array(
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # hold: sleep=-1.2, lounge=+1.2
                    [[0.4, 0.5, 0.6], [0.0, 0.0, 0.0]],  # bedroom eases, lounge still +1.2 (worst stays +1.2)
                    [[0.9, 1.0, 1.1], [-0.9, -1.0, -1.1]],  # both zones recover toward 24C!
                ],
                dtype=np.float32,
            ),
            "temp_sigma": np.full((3, 2, 3), 0.15, dtype=np.float32),
            "unit_wh": np.array(
                [
                    [[30, 60, 120], [30, 60, 120], [0, 0, 0]],  # hold: 240 Wh
                    [[10, 20, 40], [10, 20, 40], [0, 0, 0]],  # cheaper: 80 Wh, but lounge still +1.2C!
                    [[40, 80, 160], [40, 80, 160], [0, 0, 0]],  # 320 Wh, much better worst-zone comfort
                ],
                dtype=np.float32,
            ),
            "unit_wh_sigma": np.full((3, 3, 3), 10.0, dtype=np.float32),
            "total_wh": np.array([[60, 120, 240], [20, 40, 80], [80, 160, 320]], dtype=np.float32),
            "total_wh_sigma": np.full((3, 3), 15.0, dtype=np.float32),
        }
        xh0 = np.zeros((self.spec.n_hist_steps, self.spec.hist_dim), dtype=np.float32)
        with patch("neural_shadow.scoring.predict_numpy", return_value=mock_out):
            rank1 = evaluate_and_rank_candidates(art, self.site, self.spec, xh0, meta, cands[:3], policy=policy)
        self.assertFalse(rank1["declined"])
        self.assertEqual(rank1["recommended_candidate_id"], cands[2].candidate_id)

        # Scenario 2: all 3 candidates have comparable comfort inside tracking band -> lowest total Wh wins!
        meta_comfy = deepcopy(meta)
        meta_comfy["zone_temps"] = {"sleep": 24.0, "lounge": 24.0}
        mock_out_energy = deepcopy(mock_out)
        mock_out_energy["temp_delta"] = np.zeros((3, 2, 3), dtype=np.float32)
        with patch("neural_shadow.scoring.predict_numpy", return_value=mock_out_energy):
            rank2 = evaluate_and_rank_candidates(art, self.site, self.spec, xh0, meta_comfy, cands[:3], policy=policy)
        self.assertEqual(rank2["recommended_candidate_id"], cands[1].candidate_id)  # 80 Wh < 240 Wh

    # 9. Exception & timeout isolation + hourly telemetry hook
    def test_exception_timeout_isolation_and_telemetry_hook(self):
        art = build_untrained_artifact(self.site, self.spec)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = ShadowRuntime(
                self.site,
                ShadowRuntimeConfig(root / "st", root / "out", cycle_timeout_seconds=0.3),
                art,
            )

            # Exception inside loader is isolated -> returns ok=False, writes health JSON, never crashes!
            def _boom():
                raise sqlite3.OperationalError("simulated locked db")

            res_err = rt.run_isolated_cycle(_boom)
            self.assertFalse(res_err["ok"])
            self.assertEqual(res_err["error"]["type"], "OperationalError")
            health = json.loads((root / "out" / "neural-shadow-health.json").read_text(encoding="utf-8"))
            self.assertEqual(health["state"], "degraded")

            # Timeout inside loader is isolated -> ShadowCycleTimeout caught cleanly!
            def _hang():
                time.sleep(1.0)
                return _make_clean_history(self.site, minutes=60)

            res_to = rt.run_isolated_cycle(_hang)
            self.assertFalse(res_to["ok"])
            self.assertEqual(res_to["error"]["type"], "ShadowCycleTimeout")

            # Hourly telemetry hook test: valid export (with anonymization!) + corrupt isolation!
            prod = root / "prod"
            (prod / "pilot-data").mkdir(parents=True)
            (prod / "shadow").mkdir(parents=True)
            (prod / "shadow" / "rooms.json").write_text(json.dumps(self.site.raw_rooms))
            (prod / "shadow" / "topology.json").write_text(json.dumps(self.site.raw_topology))
            (prod / "shadow" / "house_policy.json").write_text(json.dumps(self.site.policy))
            (prod / "pilot-data" / "live-audit.jsonl").write_text(
                json.dumps({"event": "cycle", "at": 1_800_011_000.0, "rooms": {}}) + "\n"
            )

            # Create genuine sealed + settled shadow outputs in root/out
            base = 1_800_000_000.0
            cutoff = base + 180 * 60.0
            hist = _make_clean_history(self.site, base_ts=base, minutes=260)
            rt_clean = ShadowRuntime(self.site, ShadowRuntimeConfig(root / "st2", root / "out2"), art)
            rt_clean.seal_prediction(hist, cutoff, sealed_at=cutoff + 1.0)
            rt_clean.settle_predictions(hist, now_ts=cutoff + 4500.0)
            rt_clean.build_status_and_reports(now_ts=cutoff + 4500.0)

            # Hooked collect includes neural_shadow summary and anonymized raw/neural-shadow-scores.jsonl
            snap = telemetry_evidence.collect(
                prod,
                cutoff,
                cutoff + 4000.0,
                {"status": "verified"},
                "a" * 64,
                neural_shadow_dir=root / "out2",
            )
            self.assertIn("neural_shadow", snap["summary"])
            self.assertEqual(snap["summary"]["neural_shadow"]["status"], "available")
            self.assertEqual(snap["summary"]["neural_shadow"]["versions"]["feature_version"], FEATURE_VERSION)
            self.assertEqual(len(snap["raw"]["neural-shadow-scores"]), 3)  # 15m, 30m, 60m
            # Must pass strict public export privacy check (no raw room/entity names!)
            manifest, files = telemetry_evidence.bundle((cutoff, cutoff + 4000.0), "hourly", snap, None, None)
            self.assertIn("raw/neural-shadow-scores.jsonl", files)
            for private_tok in (b"sensor.example_", b"sleep", b"lounge"):
                self.assertNotIn(private_tok, files["raw/neural-shadow-scores.jsonl"])

            # Corrupt neural shadow directory MUST NOT break standard hourly evidence collection!
            corrupt_dir = root / "corrupt_ns"
            corrupt_dir.mkdir()
            (corrupt_dir / "neural-shadow-status.json").write_text("{broken json")
            snap_bad = telemetry_evidence.collect(
                prod,
                cutoff,
                cutoff + 4000.0,
                {"status": "verified"},
                "a" * 64,
                neural_shadow_dir=corrupt_dir,
            )
            self.assertEqual(snap_bad["summary"]["neural_shadow"]["status"], "unavailable")
            self.assertIn("window", snap_bad["summary"])

    # B2 regression: well-formed JSON with wrong types / Inf / unexpected errors must NEVER break hourly export
    def test_b2_telemetry_hook_isolates_wrong_types_and_unexpected_exceptions(self):
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            prod = root / "prod"
            (prod / "pilot-data").mkdir(parents=True)
            (prod / "shadow").mkdir(parents=True)
            (prod / "shadow" / "rooms.json").write_text(json.dumps(self.site.raw_rooms))
            (prod / "shadow" / "topology.json").write_text(json.dumps(self.site.raw_topology))
            (prod / "shadow" / "house_policy.json").write_text(json.dumps(self.site.policy))
            start = 1_800_010_800.0
            end = start + 3600.0
            (prod / "pilot-data" / "live-audit.jsonl").write_text(
                json.dumps({"event": "cycle", "at": start + 60.0, "rooms": {}}) + "\n"
            )

            # Case 1: scores.jsonl contains array line, horizon_minutes=1e999 (inf), list temp_errors, plus 1 valid line
            d1 = root / "ns_mixed_scores"
            d1.mkdir()
            (d1 / "neural-shadow-status.json").write_text(
                json.dumps(
                    {
                        "generated_at": end + 10.0,
                        "generated_at_utc": datetime.fromtimestamp(end + 10.0, timezone.utc).isoformat(),
                        "model": {"model_id": "m1", "model_state": "uncalibrated"},
                    }
                )
            )
            valid_row = {
                "prediction_id": "p_ok",
                "horizon_minutes": 15,
                "cutoff_ts": start + 60.0,
                "horizon_end_ts": start + 960.0,
                "intervened": False,
                "factual_settlement_kind": "factual_hold",
                "recommended_execution_status": "declined",
                "model_state": "uncalibrated",
                "factual_hold_temp_errors": {
                    "sleep": {"abs_error_c": 0.12, "signed_error_c": -0.12, "sigma_c": 0.2, "within_pi80": True}
                },
                "factual_hold_wh_error": None,
                "joint_wh_coverage_ratio": 1.0,
                "joint_wh_has_gap": False,
                "joint_wh_label_valid": True,
            }
            lines = [
                "[1, 2]",
                '{"prediction_id": "p_inf", "horizon_minutes": 1e999, "horizon_end_ts": ' + str(start + 1000.0) + "}",
                json.dumps({**valid_row, "prediction_id": "p_list_err", "factual_hold_temp_errors": [1, 2]}),
                json.dumps({**valid_row, "prediction_id": "p_bad_zval", "factual_hold_temp_errors": {"sleep": [1, 2]}}),
                json.dumps(valid_row),
            ]
            (d1 / "neural-shadow-scores.jsonl").write_text("\n".join(lines) + "\n")
            snap1 = telemetry_evidence.collect(prod, start, end, {"status": "verified"}, "a" * 64, neural_shadow_dir=d1)
            self.assertEqual(snap1["summary"]["neural_shadow"]["status"], "available")
            # Bad lines skipped individually; valid row (and row with valid dict minus bad zval) preserved cleanly
            self.assertGreaterEqual(len(snap1["raw"]["neural-shadow-scores"]), 1)
            self.assertGreaterEqual(snap1["watermarks"]["neural_shadow"]["skipped_malformed_rows"], 3)
            telemetry_evidence.bundle((start, end), "hourly", snap1, None, None)

            # Case 2: status.json is a JSON array [] -> must NOT raise AttributeError
            d2 = root / "ns_status_array"
            d2.mkdir()
            (d2 / "neural-shadow-status.json").write_text("[]")
            snap2 = telemetry_evidence.collect(prod, start, end, {"status": "verified"}, "a" * 64, neural_shadow_dir=d2)
            self.assertEqual(snap2["summary"]["neural_shadow"]["status"], "unavailable")
            self.assertIn("window", snap2["summary"])
            telemetry_evidence.bundle((start, end), "hourly", snap2, None, None)

            # Case 3: status.model is a string -> must NOT raise AttributeError
            d3 = root / "ns_model_str"
            d3.mkdir()
            (d3 / "neural-shadow-status.json").write_text(json.dumps({"model": "unexpected_string"}))
            snap3 = telemetry_evidence.collect(prod, start, end, {"status": "verified"}, "a" * 64, neural_shadow_dir=d3)
            self.assertEqual(snap3["summary"]["neural_shadow"]["status"], "unavailable")
            self.assertIn("window", snap3["summary"])
            telemetry_evidence.bundle((start, end), "hourly", snap3, None, None)

            # Case 4: collect() outer boundary isolates ANY unexpected Exception from read_neural_shadow
            with patch.object(telemetry_evidence, "read_neural_shadow", side_effect=RuntimeError("unexpected hook failure")):
                snap4 = telemetry_evidence.collect(prod, start, end, {"status": "verified"}, "a" * 64, neural_shadow_dir=d1)
            self.assertEqual(snap4["summary"]["neural_shadow"]["status"], "unavailable")
            self.assertEqual(snap4["summary"]["neural_shadow"]["error"], "RuntimeError")
            self.assertIn("window", snap4["summary"])
            telemetry_evidence.bundle((start, end), "hourly", snap4, None, None)

    # 10. Static AST & Compose verification: ZERO actuation path!
    def test_no_actuation_path_static_and_compose_audit(self):
        ns_dir = REPO_ROOT / "home-assistant" / "neural_shadow"
        forbidden_imports = {"urllib", "http", "requests", "aiohttp", "httpx", "socket", "continuous_pilot", "heating_pilot"}
        forbidden_literals = ("services/climate", "set_temperature", "set_hvac_mode", "ha_client")

        py_files = sorted(ns_dir.glob("*.py"))
        self.assertGreaterEqual(len(py_files), 8)
        for py in py_files:
            src = py.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(py))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        self.assertNotIn(top, forbidden_imports, f"Forbidden import {top} in {py.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    self.assertNotIn(top, forbidden_imports, f"Forbidden import-from {top} in {py.name}")
            for lit in forbidden_literals:
                self.assertNotIn(lit, src, f"Forbidden actuation string {lit} in {py.name}")

        compose_text = (REPO_ROOT / "home-assistant" / "compose.neural-shadow.yaml").read_text(encoding="utf-8")
        self.assertIn("network_mode: none", compose_text)
        self.assertIn("read_only: true", compose_text)
        for ro_mount in (
            "./shadow:/opt/shadow-config:ro",
            "./config:/ha-readonly:ro",
            "./shadow-data:/shadow-data-ro:ro",
            "./pilot-data:/pilot-data-ro:ro",
            "./neural-shadow-models:/models:ro",
        ):
            self.assertIn(ro_mount, compose_text)

    # 11. Read-only production SQLite/journal ingestion & proxy_echo quality flag
    def test_production_readonly_sqlite_ingestion_and_proxy_echo_flags(self):
        with TemporaryDirectory() as tmp:
            prod = Path(tmp)
            (prod / "shadow").mkdir()
            (prod / "shadow-data").mkdir()
            (prod / "pilot-data").mkdir()
            (prod / "config" / ".storage").mkdir(parents=True)

            (prod / "shadow" / "rooms.json").write_text(json.dumps(self.site.raw_rooms))
            (prod / "shadow" / "topology.json").write_text(json.dumps(self.site.raw_topology))
            (prod / "shadow" / "house_policy.json").write_text(json.dumps(self.site.policy))

            base = 1_800_000_000.0
            sdb = prod / "shadow-data" / "shadow.sqlite3"
            with closing(sqlite3.connect(sdb)) as db, db:
                db.execute(
                    "CREATE TABLE power_reports (source TEXT, observer_epoch TEXT, report_at REAL, watts REAL)"
                )
                for m in range(10):
                    for act in self.site.actuators:
                        db.execute(
                            "INSERT INTO power_reports VALUES (?,?,?,?)",
                            (act.power_entity, "ep1", base + m * 60.0, 150.0),
                        )

            rdb = prod / "config" / ".storage" / "room_climate_control.sqlite3"
            with closing(sqlite3.connect(rdb)) as db, db:
                db.execute(
                    "CREATE TABLE samples (room TEXT, tick INTEGER, measured REAL, desired REAL, reported REAL, adapter_mode TEXT, watts REAL, outdoor REAL, temp_report INTEGER, quality TEXT)"
                )
                for m in range(10):
                    t = int(base + m * 60)
                    # Repeated temp_report on bedroom; shared sensor on living & study
                    db.execute("INSERT INTO samples VALUES ('bedroom',?,23.5,24.0,23.0,'heat',150.0,16.0,?,'observing')", (t, int(base)))
                    db.execute("INSERT INTO samples VALUES ('living',?,24.1,24.0,24.0,'off',2.0,16.0,?,'observing')", (t, int(base + (m // 5) * 300)))
                    db.execute("INSERT INTO samples VALUES ('study',?,24.1,24.0,24.0,'off',2.0,16.0,?,'observing')", (t, int(base + (m // 5) * 300)))

            # Pilot command journal with proxy_echo (must NOT be physical_verified!) + failed command
            journal = prod / "pilot-data" / "continuous-1800000000.json"
            journal.write_text(
                json.dumps(
                    {
                        "start": base,
                        "commands": [
                            {
                                "room": "bedroom",
                                "kind": "temperature",
                                "value": 24,
                                "before": ["heat", 23, "auto"],
                                "sent_at": base + 120.0,
                                "proxy_at": base + 123.0,
                                "status": "proxy_echo",
                                "physical_verified": True,  # Even if set erroneously, proxy_echo must remain unverified!
                            },
                            {
                                "room": "living",
                                "kind": "temperature",
                                "value": 25,
                                "before": ["cool", 24, "auto"],
                                "sent_at": base + 240.0,
                                "status": "unconfirmed",
                                "physical_verified": False,
                            },
                        ],
                    }
                )
            )
            before_mtimes = {p: p.stat().st_mtime_ns for p in (sdb, rdb, journal)}
            hist = ingest_production_readonly(prod)
            after_mtimes = {p: p.stat().st_mtime_ns for p in (sdb, rdb, journal)}
            self.assertEqual(before_mtimes, after_mtimes)  # Strictly read-only!
            # Bedroom had 10 minute copies of same temp_report -> deduplicated to 1!
            self.assertEqual(len(hist.zone_temp_reports["sleep"]), 1)
            # Living + Study shared sensor had 2 distinct report timestamps -> 2 (NOT 4 or 20)!
            self.assertEqual(len(hist.zone_temp_reports["lounge"]), 2)
            # proxy_echo explicitly unverified; failed call flagged
            echo_cmd = next(a for a in hist.actions if a.receipt_status == "proxy_echo")
            self.assertFalse(echo_cmd.physical_verified)
            self.assertIn("proxy_only_unverified", echo_cmd.quality_flags)
            fail_cmd = next(a for a in hist.actions if a.receipt_status == "unconfirmed")
            self.assertIn("failed_call", fail_cmd.quality_flags)

    # 12. Full E2E synthetic pipeline verification
    def test_e2e_synthetic_pipeline_runs_cleanly_and_marks_synthetic_not_deployable(self):
        with TemporaryDirectory() as tmp:
            summary = run_e2e(Path(tmp), synthetic_hours=48.0, epochs=4, seed=42)
            self.assertEqual(summary["data_origin"], DataOrigin.SYNTHETIC.value)
            self.assertFalse(summary["deployment_eligible"])
            self.assertEqual(summary["model_state"], ModelState.INSUFFICIENT_DATA.value)
            self.assertGreaterEqual(summary["parameter_count"], 100_000)
            self.assertLessEqual(summary["parameter_count"], 300_000)
            self.assertGreater(summary["sealed_count"], 0)
            self.assertGreater(summary["settled_horizon_count"], 0)
            for fname in ("neural-shadow-status.json", "neural-shadow-scores.jsonl", "neural-shadow-health.json", "neural-shadow-report.md"):
                self.assertIn(fname, summary["report_files"])


if __name__ == "__main__":
    unittest.main()
