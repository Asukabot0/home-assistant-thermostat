"""Reference baselines for offline comparison against ThermostatShadowGRU.

Required baselines:
1. PersistenceBaseline: zero temperature delta, constant-power Wh extrapolation.
2. ExistingTrendBaseline: reuses the repository's existing 15-minute trend prediction
   (home-assistant/shadow/model.py:predict) scaled across 15/30/60m horizons.
3. ThermalInertiaLinearBaseline: ridge regression over thermal-inertia, outdoor gradient,
   zone coupling, recent power, and future candidate action features (fitted on train split only).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .data_pipeline import BuiltDataset, FeatureSpec


def _current_unit_watts(metadata: list[dict[str, Any]], unit_names: tuple[str, ...]) -> np.ndarray:
    n = len(metadata)
    u = len(unit_names)
    out = np.zeros((n, u), dtype=np.float32)
    for i, m in enumerate(metadata):
        for j, uname in enumerate(unit_names):
            w = m["unit_watts"].get(uname)
            out[i, j] = float(w) if (w is not None and w >= 0.0) else 0.0
    return out


class PersistenceBaseline:
    """Zero temperature change; constant-power cumulative Wh with train-fitted uncertainty."""

    name = "persistence"

    def __init__(self, spec: FeatureSpec):
        self.spec = spec
        self.temp_sigma = np.full((spec.n_zones, len(spec.horizons_minutes)), 0.5, dtype=np.float32)
        self.unit_wh_sigma = np.full((spec.n_units, len(spec.horizons_minutes)), 25.0, dtype=np.float32)
        self.total_wh_sigma = np.full((len(spec.horizons_minutes),), 50.0, dtype=np.float32)

    def fit(self, dataset: BuiltDataset, train_idx: np.ndarray) -> "PersistenceBaseline":
        if train_idx.size == 0:
            return self
        pred = self.predict(dataset, train_idx, use_actual_actions=False)
        for z in range(self.spec.n_zones):
            for k in range(len(self.spec.horizons_minutes)):
                mask = dataset.y_temp_valid[train_idx, z, k]
                if mask.sum() >= 5:
                    err = dataset.y_temp_delta[train_idx[mask], z, k] - pred["temp_delta"][mask, z, k]
                    self.temp_sigma[z, k] = max(0.05, float(np.std(err)))
        for u in range(self.spec.n_units):
            for k in range(len(self.spec.horizons_minutes)):
                mask = dataset.y_unit_valid[train_idx, u, k]
                if mask.sum() >= 5:
                    err = dataset.y_unit_wh[train_idx[mask], u, k] - pred["unit_wh"][mask, u, k]
                    self.unit_wh_sigma[u, k] = max(1.0, float(np.std(err)))
        for k in range(len(self.spec.horizons_minutes)):
            mask = dataset.y_total_valid[train_idx, k]
            if mask.sum() >= 5:
                err = dataset.y_total_wh[train_idx[mask], k] - pred["total_wh"][mask, k]
                self.total_wh_sigma[k] = max(2.0, float(np.std(err)))
        return self

    def predict(
        self,
        dataset: BuiltDataset,
        indices: np.ndarray,
        *,
        use_actual_actions: bool = False,
    ) -> dict[str, np.ndarray]:
        del use_actual_actions
        n = len(indices)
        n_z = self.spec.n_zones
        n_u = self.spec.n_units
        n_k = len(self.spec.horizons_minutes)
        temp_delta = np.zeros((n, n_z, n_k), dtype=np.float32)
        watts = _current_unit_watts([dataset.metadata[int(i)] for i in indices], self.spec.unit_names)
        unit_wh = np.zeros((n, n_u, n_k), dtype=np.float32)
        for k, h_min in enumerate(self.spec.horizons_minutes):
            unit_wh[:, :, k] = np.maximum(0.0, watts * (float(h_min) / 60.0))
        total_wh = unit_wh.sum(axis=1)
        return {
            "temp_delta": temp_delta,
            "temp_sigma": np.broadcast_to(self.temp_sigma[None, :, :], (n, n_z, n_k)).copy(),
            "unit_wh": unit_wh,
            "unit_wh_sigma": np.broadcast_to(self.unit_wh_sigma[None, :, :], (n, n_u, n_k)).copy(),
            "total_wh": total_wh,
            "total_wh_sigma": np.broadcast_to(self.total_wh_sigma[None, :], (n, n_k)).copy(),
        }


class ExistingTrendBaseline:
    """Reuses the existing repository's 15m robust median-slope trend prediction."""

    name = "existing_trend"

    def __init__(self, spec: FeatureSpec):
        self.spec = spec
        self.persistence = PersistenceBaseline(spec)
        self.temp_sigma = np.full((spec.n_zones, len(spec.horizons_minutes)), 0.5, dtype=np.float32)

    def fit(self, dataset: BuiltDataset, train_idx: np.ndarray) -> "ExistingTrendBaseline":
        self.persistence.fit(dataset, train_idx)
        if train_idx.size == 0:
            return self
        pred = self.predict(dataset, train_idx, use_actual_actions=False)
        for z in range(self.spec.n_zones):
            for k in range(len(self.spec.horizons_minutes)):
                mask = dataset.y_temp_valid[train_idx, z, k]
                if mask.sum() >= 5:
                    err = dataset.y_temp_delta[train_idx[mask], z, k] - pred["temp_delta"][mask, z, k]
                    self.temp_sigma[z, k] = max(0.05, float(np.std(err)))
        return self

    def predict(
        self,
        dataset: BuiltDataset,
        indices: np.ndarray,
        *,
        use_actual_actions: bool = False,
    ) -> dict[str, np.ndarray]:
        base = self.persistence.predict(dataset, indices, use_actual_actions=use_actual_actions)
        n = len(indices)
        n_z = self.spec.n_zones
        n_k = len(self.spec.horizons_minutes)
        temp_delta = np.zeros((n, n_z, n_k), dtype=np.float32)
        # Damp linear extrapolation gently beyond 15m to avoid runaway blowup at 60m.
        scale_map = {15: 1.0, 30: 1.6, 60: 2.2}
        for row_i, idx in enumerate(indices):
            meta = dataset.metadata[int(idx)]
            for z_idx, zname in enumerate(self.spec.zone_names):
                t15 = meta["zone_trend_15m"].get(zname)
                if t15 is None:
                    continue
                for k, h_min in enumerate(self.spec.horizons_minutes):
                    mult = scale_map.get(h_min, float(h_min) / 15.0)
                    temp_delta[row_i, z_idx, k] = float(np.clip(t15 * mult, -3.5, 3.5))
        base["temp_delta"] = temp_delta
        base["temp_sigma"] = np.broadcast_to(self.temp_sigma[None, :, :], (n, n_z, n_k)).copy()
        return base


class ThermalInertiaLinearBaseline:
    """Simple physical thermal-inertia & action-conditioned ridge regression baseline.

    Fitted strictly on training data only. Features:
    - latest history step (t0 state, indoor-outdoor gradient, inter-zone gradient, power, mode)
    - 30m temperature slope from history
    - mean future candidate action over each horizon (distinct setpoint, mode, duration)
    Enforces non-negative, monotonically non-decreasing cumulative Wh across horizons.
    """

    name = "thermal_inertia_linear"

    def __init__(self, spec: FeatureSpec, ridge_alpha: float = 10.0):
        self.spec = spec
        self.ridge_alpha = ridge_alpha
        self.fill_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.temp_weights: dict[tuple[int, int], np.ndarray] = {}
        self.temp_sigma = np.full((spec.n_zones, len(spec.horizons_minutes)), 0.5, dtype=np.float32)
        self.wh_weights: dict[tuple[int, int], np.ndarray] = {}
        self.unit_wh_sigma = np.full((spec.n_units, len(spec.horizons_minutes)), 25.0, dtype=np.float32)
        self.total_wh_sigma = np.full((len(spec.horizons_minutes),), 50.0, dtype=np.float32)

    def _design_raw(self, X_hist: np.ndarray, X_act: np.ndarray, horizon_step_idx: int) -> np.ndarray:
        """Build [N, D] tabular thermal-inertia features for a given horizon readout."""
        n = X_hist.shape[0]
        last_h = X_hist[:, -1, :]  # t0 causal features
        # 30-min (6 steps) causal history average for thermal inertia (warning-free on all-NaN slices)
        k_back = min(6, X_hist.shape[1])
        win = X_hist[:, -k_back:, :]
        valid_mask = np.isfinite(win)
        counts = valid_mask.sum(axis=1)
        sums = np.where(valid_mask, win, 0.0).sum(axis=1)
        recent_h = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        # Future candidate action summary up to this horizon (strictly candidate actions, no future measurements!)
        act_mean = X_act[:, : horizon_step_idx + 1, :].mean(axis=1)
        act_first = X_act[:, 0, :]
        return np.concatenate([last_h, recent_h, act_first, act_mean], axis=1).astype(np.float64)

    def _fit_scaler(self, raw: np.ndarray) -> np.ndarray:
        d = raw.shape[1]
        fill = np.zeros(d, dtype=np.float64)
        mean = np.zeros(d, dtype=np.float64)
        std = np.ones(d, dtype=np.float64)
        for j in range(d):
            col = raw[:, j]
            valid = col[np.isfinite(col)]
            if valid.size > 0:
                fill[j] = float(np.median(valid))
                c2 = np.where(np.isfinite(col), col, fill[j])
                mean[j] = float(np.mean(c2))
                s = float(np.std(c2))
                std[j] = s if s > 1e-4 else 1.0
        self.fill_, self.mean_, self.std_ = fill, mean, std
        return self._transform(raw)

    def _transform(self, raw: np.ndarray) -> np.ndarray:
        assert self.fill_ is not None and self.mean_ is not None and self.std_ is not None
        filled = np.where(np.isfinite(raw), raw, self.fill_)
        z = np.clip((filled - self.mean_) / self.std_, -10.0, 10.0)
        return np.concatenate([np.ones((z.shape[0], 1), dtype=np.float64), z], axis=1)

    def _solve_ridge(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        reg = np.eye(d, dtype=np.float64) * self.ridge_alpha
        reg[0, 0] = 1e-6  # do not regularize intercept
        a = X.T @ X + reg
        b = X.T @ y
        return np.linalg.solve(a, b)

    def fit(self, dataset: BuiltDataset, train_idx: np.ndarray) -> "ThermalInertiaLinearBaseline":
        if train_idx.size < 5:
            return self
        xh = dataset.X_hist[train_idx]
        xa = dataset.X_act_actual[train_idx]
        # Fit scaler on longest-horizon design matrix (train split only!)
        last_step = self.spec.horizon_step_indices[-1]
        self._fit_scaler(self._design_raw(xh, xa, last_step))

        for k, step_idx in enumerate(self.spec.horizon_step_indices):
            Xk = self._transform(self._design_raw(xh, xa, step_idx))
            for z in range(self.spec.n_zones):
                mask = dataset.y_temp_valid[train_idx, z, k]
                if mask.sum() >= 5:
                    y = dataset.y_temp_delta[train_idx[mask], z, k].astype(np.float64)
                    w = self._solve_ridge(Xk[mask], y)
                    self.temp_weights[(z, k)] = w
                    err = y - Xk[mask] @ w
                    self.temp_sigma[z, k] = max(0.05, float(np.std(err)))
            for u in range(self.spec.n_units):
                mask = dataset.y_unit_valid[train_idx, u, k]
                if mask.sum() >= 5:
                    y = np.maximum(0.0, dataset.y_unit_wh[train_idx[mask], u, k].astype(np.float64))
                    w = self._solve_ridge(Xk[mask], y)
                    self.wh_weights[(u, k)] = w
                    pred_u = np.maximum(0.0, Xk[mask] @ w)
                    self.unit_wh_sigma[u, k] = max(1.0, float(np.std(y - pred_u)))

        # Fit total Wh residual std on train split
        pred_tr = self.predict(dataset, train_idx, use_actual_actions=True)
        for k in range(len(self.spec.horizons_minutes)):
            mask = dataset.y_total_valid[train_idx, k]
            if mask.sum() >= 5:
                err = dataset.y_total_wh[train_idx[mask], k] - pred_tr["total_wh"][mask, k]
                self.total_wh_sigma[k] = max(2.0, float(np.std(err)))
        return self

    def predict(
        self,
        dataset: BuiltDataset,
        indices: np.ndarray,
        *,
        use_actual_actions: bool = False,
    ) -> dict[str, np.ndarray]:
        n = len(indices)
        n_z = self.spec.n_zones
        n_u = self.spec.n_units
        n_k = len(self.spec.horizons_minutes)
        temp_delta = np.zeros((n, n_z, n_k), dtype=np.float32)
        unit_wh = np.zeros((n, n_u, n_k), dtype=np.float32)

        if n > 0 and self.fill_ is not None:
            xh = dataset.X_hist[indices]
            xa = dataset.X_act_actual[indices] if use_actual_actions else dataset.X_act_hold[indices]
            for k, step_idx in enumerate(self.spec.horizon_step_indices):
                Xk = self._transform(self._design_raw(xh, xa, step_idx))
                for z in range(n_z):
                    if (z, k) in self.temp_weights:
                        temp_delta[:, z, k] = np.clip(Xk @ self.temp_weights[(z, k)], -6.0, 6.0).astype(np.float32)
                for u in range(n_u):
                    if (u, k) in self.wh_weights:
                        unit_wh[:, u, k] = np.maximum(0.0, Xk @ self.wh_weights[(u, k)]).astype(np.float32)
            # Enforce non-decreasing cumulative Wh with horizon
            unit_wh = np.maximum.accumulate(unit_wh, axis=-1)

        total_wh = unit_wh.sum(axis=1)
        return {
            "temp_delta": temp_delta,
            "temp_sigma": np.broadcast_to(self.temp_sigma[None, :, :], (n, n_z, n_k)).copy(),
            "unit_wh": unit_wh,
            "unit_wh_sigma": np.broadcast_to(self.unit_wh_sigma[None, :, :], (n, n_u, n_k)).copy(),
            "total_wh": total_wh,
            "total_wh_sigma": np.broadcast_to(self.total_wh_sigma[None, :], (n, n_k)).copy(),
        }
