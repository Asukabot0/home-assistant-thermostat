"""Small action-conditioned GRU neural network (v1) for thermostat shadow forecasting.

Architecture (~230k-255k parameters):
- History input projection + LayerNorm + 2-layer 128-d GRU encoder over 1-3h history.
- Step-by-step action-conditioned GRUCell decoder over future 5-min candidate steps (up to 60m).
- Distinct branches for:
  1. Per-unit incremental electrical Wh (>= 0 via softplus), summed monotonically across steps
     so CumWh(15m) <= CumWh(30m) <= CumWh(60m) and TotalWh == sum_u UnitWh by construction.
  2. Per-unit effective thermal output (signed, distinct from setpoint and electrical watts),
     propagated through a simple thermal-inertia and inter-zone coupling state update.
  3. Calibrated heteroscedastic uncertainty (sigma) for zone temperature deltas and Wh.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import FEATURE_VERSION, MODEL_ARCH_VERSION, ModelState
from .config import SiteConfig
from .data_pipeline import FeatureSpec


class ModelVersionMismatchError(ValueError):
    """Raised when a model checkpoint does not match the active site/feature/arch contract."""


@dataclass
class FeatureNormalizer:
    """Train-only missing-value imputer, standardizer, and training-coverage OOD envelope."""

    hist_fill: np.ndarray  # [D_hist] float32
    hist_mean: np.ndarray  # [D_hist] float32
    hist_std: np.ndarray  # [D_hist] float32
    act_mean: np.ndarray  # [D_act] float32
    act_std: np.ndarray  # [D_act] float32
    coverage_envelope: dict[str, Any]

    @classmethod
    def fit(
        cls,
        X_hist_train: np.ndarray,
        X_act_train: np.ndarray,
        train_metadata: list[dict[str, Any]],
        spec: FeatureSpec,
    ) -> "FeatureNormalizer":
        """Fit imputation, standardization, and OOD coverage bounds STRICTLY on training data."""
        if X_hist_train.shape[0] == 0:
            raise ValueError("Cannot fit FeatureNormalizer on empty training split")

        flat_h = X_hist_train.reshape(-1, X_hist_train.shape[-1]).astype(np.float64)
        d_h = flat_h.shape[1]
        h_fill = np.zeros(d_h, dtype=np.float32)
        h_mean = np.zeros(d_h, dtype=np.float32)
        h_std = np.ones(d_h, dtype=np.float32)

        for j in range(d_h):
            col = flat_h[:, j]
            valid = col[np.isfinite(col)]
            if valid.size == 0:
                h_fill[j] = 0.0
                h_mean[j] = 0.0
                h_std[j] = 1.0
            else:
                med = float(np.median(valid))
                h_fill[j] = med
                filled = np.where(np.isfinite(col), col, med)
                h_mean[j] = float(np.mean(filled))
                s = float(np.std(filled))
                h_std[j] = s if s > 1e-4 else 1.0

        flat_a = np.nan_to_num(X_act_train.reshape(-1, X_act_train.shape[-1]).astype(np.float64), nan=0.0)
        a_mean = flat_a.mean(axis=0).astype(np.float32)
        a_std_raw = flat_a.std(axis=0).astype(np.float32)
        a_std = np.where(a_std_raw > 1e-4, a_std_raw, 1.0).astype(np.float32)

        # Build training-data coverage envelope for runtime OOD detection.
        zone_ranges: dict[str, list[float] | None] = {}
        for z in spec.zone_names:
            vals = [m["zone_temps"][z] for m in train_metadata if m["zone_temps"].get(z) is not None]
            zone_ranges[z] = [float(min(vals)), float(max(vals))] if vals else None

        env_ranges: dict[str, list[float] | None] = {}
        for e in spec.env_sensor_names:
            vals = [m["env_temps"][e] for m in train_metadata if m["env_temps"].get(e) is not None]
            env_ranges[e] = [float(min(vals)), float(max(vals))] if vals else None

        seen_modes: dict[str, list[str]] = {}
        power_max: dict[str, float | None] = {}
        for u in spec.unit_names:
            modes = sorted({m["unit_modes"][u] for m in train_metadata if m["unit_modes"].get(u) is not None})
            seen_modes[u] = modes
            pvals = [m["unit_watts"][u] for m in train_metadata if m["unit_watts"].get(u) is not None]
            power_max[u] = float(max(pvals)) if pvals else None

        envelope = {
            "zone_temp_min_max_c": zone_ranges,
            "env_temp_min_max_c": env_ranges,
            "unit_seen_modes": seen_modes,
            "unit_power_max_w": power_max,
            "n_train_samples": len(train_metadata),
        }
        return cls(
            hist_fill=h_fill,
            hist_mean=h_mean,
            hist_std=h_std,
            act_mean=a_mean,
            act_std=a_std,
            coverage_envelope=envelope,
        )

    def transform_hist(self, X_hist: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(X_hist), X_hist, self.hist_fill)
        scaled = (filled - self.hist_mean) / self.hist_std
        return np.clip(scaled, -12.0, 12.0).astype(np.float32)

    def transform_act(self, X_act: np.ndarray) -> np.ndarray:
        filled = np.nan_to_num(X_act, nan=0.0)
        scaled = (filled - self.act_mean) / self.act_std
        return np.clip(scaled, -12.0, 12.0).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hist_fill": self.hist_fill.tolist(),
            "hist_mean": self.hist_mean.tolist(),
            "hist_std": self.hist_std.tolist(),
            "act_mean": self.act_mean.tolist(),
            "act_std": self.act_std.tolist(),
            "coverage_envelope": self.coverage_envelope,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeatureNormalizer":
        return cls(
            hist_fill=np.asarray(raw["hist_fill"], dtype=np.float32),
            hist_mean=np.asarray(raw["hist_mean"], dtype=np.float32),
            hist_std=np.asarray(raw["hist_std"], dtype=np.float32),
            act_mean=np.asarray(raw["act_mean"], dtype=np.float32),
            act_std=np.asarray(raw["act_std"], dtype=np.float32),
            coverage_envelope=dict(raw.get("coverage_envelope", {})),
        )


class ThermostatShadowGRU(nn.Module):
    """2-layer 128-d GRU encoder + action-conditioned thermal/electrical decoder.

    Parameter budget: ~240k parameters for the default 2-zone / 3-unit configuration.
    """

    def __init__(
        self,
        *,
        hist_dim: int,
        act_dim: int,
        n_zones: int,
        n_units: int,
        horizon_step_indices: tuple[int, ...] = (2, 5, 11),
        influence_matrix: list[list[float]] | None = None,
        coupling_matrix: list[list[float]] | None = None,
        encoder_input_dim: int = 96,
        encoder_hidden_dim: int = 128,
        encoder_layers: int = 2,
        decoder_hidden_dim: int = 64,
        action_embed_dim: int = 48,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hist_dim = hist_dim
        self.act_dim = act_dim
        self.n_zones = n_zones
        self.n_units = n_units
        self.horizon_step_indices = tuple(horizon_step_indices)
        self.n_horizons = len(self.horizon_step_indices)
        self.encoder_input_dim = encoder_input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_layers = encoder_layers
        self.decoder_hidden_dim = decoder_hidden_dim
        self.action_embed_dim = action_embed_dim
        self.dropout_rate = dropout

        # 1. History encoder: input projection -> 2-layer 128-d GRU
        self.hist_proj = nn.Sequential(
            nn.Linear(hist_dim, encoder_input_dim),
            nn.LayerNorm(encoder_input_dim),
            nn.GELU(),
        )
        self.encoder = nn.GRU(
            input_size=encoder_input_dim,
            hidden_size=encoder_hidden_dim,
            num_layers=encoder_layers,
            batch_first=True,
            dropout=dropout if encoder_layers > 1 else 0.0,
        )

        # 2. Action-conditioned step-by-step decoder
        self.dec_init = nn.Sequential(
            nn.Linear(encoder_hidden_dim, decoder_hidden_dim),
            nn.Tanh(),
        )
        self.act_proj = nn.Sequential(
            nn.Linear(act_dim, action_embed_dim),
            nn.LayerNorm(action_embed_dim),
            nn.GELU(),
        )
        self.decoder_cell = nn.GRUCell(action_embed_dim, decoder_hidden_dim)
        self.dec_drop = nn.Dropout(dropout)

        # 3a. Electrical Wh increment branch (strictly non-negative, summed monotonically)
        # Distinct from setpoint and distinct from effective heating/cooling thermal output.
        self.elec_head = nn.Sequential(
            nn.Linear(decoder_hidden_dim + action_embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_units),
        )

        # 3b. Effective thermal output per unit (signed: +heating, -cooling)
        self.thermal_head = nn.Sequential(
            nn.Linear(decoder_hidden_dim + action_embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_units),
        )

        # 3c. Ambient / unmodeled zone drift residual per step
        self.drift_head = nn.Sequential(
            nn.Linear(decoder_hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_zones),
        )

        # 3d. Simple thermal inertia & zone-coupling structure
        inf_mat = (
            np.asarray(influence_matrix, dtype=np.float32)
            if influence_matrix is not None
            else np.ones((n_zones, n_units), dtype=np.float32) / max(1, n_units)
        )
        cpl_mat = (
            np.asarray(coupling_matrix, dtype=np.float32)
            if coupling_matrix is not None
            else (np.ones((n_zones, n_zones), dtype=np.float32) - np.eye(n_zones, dtype=np.float32)) * 0.25
        )
        self.register_buffer("influence_prior", torch.tensor(inf_mat, dtype=torch.float32))
        self.register_buffer("coupling_mask", torch.tensor(cpl_mat, dtype=torch.float32))
        # Learnable bounded gain on actuator->zone influence, zone inertia, and inter-zone coupling
        self.influence_log_gain = nn.Parameter(torch.zeros((n_zones, n_units)))
        self.zone_inertia_logit = nn.Parameter(torch.full((n_zones,), 2.0))  # sigmoid ~ 0.88 per 5m
        self.coupling_logit = nn.Parameter(torch.tensor(-2.0))  # small positive exchange

        # 4. Heteroscedastic uncertainty heads (evaluated at readout horizons 15/30/60m)
        self.temp_sigma_head = nn.Sequential(
            nn.Linear(decoder_hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_zones),
        )
        self.unit_wh_sigma_head = nn.Sequential(
            nn.Linear(decoder_hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_units),
        )
        self.total_wh_sigma_head = nn.Sequential(
            nn.Linear(decoder_hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        # Post-hoc validation calibration multipliers (registered as buffers, updated only on val split)
        self.register_buffer("temp_cal_scale", torch.ones((n_zones, self.n_horizons), dtype=torch.float32))
        self.register_buffer("unit_wh_cal_scale", torch.ones((n_units, self.n_horizons), dtype=torch.float32))
        self.register_buffer("total_wh_cal_scale", torch.ones((self.n_horizons,), dtype=torch.float32))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict[str, Any]:
        return {
            "hist_dim": self.hist_dim,
            "act_dim": self.act_dim,
            "n_zones": self.n_zones,
            "n_units": self.n_units,
            "horizon_step_indices": list(self.horizon_step_indices),
            "influence_matrix": self.influence_prior.detach().cpu().tolist(),
            "coupling_matrix": self.coupling_mask.detach().cpu().tolist(),
            "encoder_input_dim": self.encoder_input_dim,
            "encoder_hidden_dim": self.encoder_hidden_dim,
            "encoder_layers": self.encoder_layers,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "action_embed_dim": self.action_embed_dim,
            "dropout": self.dropout_rate,
        }

    def forward(self, x_hist: torch.Tensor, x_act: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x_hist: [B, H, hist_dim] normalized causal history features.
            x_act:  [B, F, act_dim] normalized future candidate action sequence.

        Returns dict with:
            temp_delta:    [B, Z, K] predicted zone temperature delta (C) at horizons (15, 30, 60m)
            temp_sigma:    [B, Z, K] calibrated uncertainty (std, C) >= 0.05
            unit_wh:       [B, U, K] cumulative per-unit Wh (>= 0, non-decreasing in K)
            unit_wh_sigma: [B, U, K] calibrated per-unit Wh uncertainty (>= 1.0)
            total_wh:      [B, K]    cumulative total Wh == sum_u unit_wh (>= 0, non-decreasing in K)
            total_wh_sigma:[B, K]    calibrated total Wh uncertainty (>= 2.0)
            step_thermal_q:[B, F, U] latent effective thermal output per step (for diagnostics)
            step_elec_wh:  [B, F, U] incremental electrical Wh per step (for diagnostics)
        """
        bsz = x_hist.shape[0]
        n_steps = x_act.shape[1]
        enc_in = self.hist_proj(x_hist)
        _, h_n = self.encoder(enc_in)  # [num_layers, B, encoder_hidden_dim]
        d_state = self.dec_init(h_n[-1])  # [B, decoder_hidden_dim]

        # Effective actuator-to-zone influence matrix (non-negative scaled prior)
        eff_influence = self.influence_prior * F.softplus(self.influence_log_gain + 0.5413)  # softplus(0.5413) ~ 1.0
        inertia = torch.sigmoid(self.zone_inertia_logit)  # [Z] in (0, 1)
        couple_rate = 0.25 * torch.sigmoid(self.coupling_logit)  # bounded <= 0.25 per step

        delta_t = torch.zeros((bsz, self.n_zones), dtype=x_hist.dtype, device=x_hist.device)
        cum_unit_wh = torch.zeros((bsz, self.n_units), dtype=x_hist.dtype, device=x_hist.device)

        temp_deltas: list[torch.Tensor] = []
        temp_sigmas: list[torch.Tensor] = []
        unit_whs: list[torch.Tensor] = []
        unit_wh_sigmas: list[torch.Tensor] = []
        total_whs: list[torch.Tensor] = []
        total_wh_sigmas: list[torch.Tensor] = []
        step_q_list: list[torch.Tensor] = []
        step_e_list: list[torch.Tensor] = []

        horizon_set = {step_idx: h_pos for h_pos, step_idx in enumerate(self.horizon_step_indices)}

        for s in range(n_steps):
            a_emb = self.act_proj(x_act[:, s, :])  # [B, action_embed_dim]
            d_state = self.decoder_cell(a_emb, d_state)
            d_reg = self.dec_drop(d_state)
            joint = torch.cat([d_reg, a_emb], dim=-1)

            # Incremental electrical Wh: strictly non-negative!
            inc_wh = F.softplus(self.elec_head(joint)) * 12.0  # typical 5m scale ~ 0..100+ Wh
            cum_unit_wh = cum_unit_wh + inc_wh

            # Effective signed thermal output per unit (distinct from electrical Wh and setpoint)
            q_unit = torch.tanh(self.thermal_head(joint)) * 0.45  # bounded thermal effect per 5m step

            # Simple zone-coupling heat exchange proportional to delta_t difference
            if self.n_zones > 1:
                # diff_mat[b, i, j] = delta_t[b, j] - delta_t[b, i]
                diff = delta_t.unsqueeze(1) - delta_t.unsqueeze(2)  # [B, Z, Z]
                coupling_flux = couple_rate * (diff * self.coupling_mask.unsqueeze(0)).sum(dim=-1)
            else:
                coupling_flux = torch.zeros_like(delta_t)

            # Zone thermal inertia update: delta_t_{s} = inertia * delta_t_{s-1} + influence @ q + coupling + drift
            # Using mild damping (1 - 0.15*(1 - inertia)) so inertia preserves accumulated delta without collapse
            retention = 0.85 + 0.15 * inertia
            act_effect = torch.matmul(q_unit, eff_influence.transpose(0, 1))  # [B, Z]
            drift = self.drift_head(d_reg) * 0.20
            delta_t = retention * delta_t + act_effect + coupling_flux + drift

            step_q_list.append(q_unit)
            step_e_list.append(inc_wh)

            if s in horizon_set:
                h_pos = horizon_set[s]
                temp_deltas.append(delta_t)
                raw_tsig = F.softplus(self.temp_sigma_head(d_state)) + 0.05
                temp_sigmas.append(raw_tsig * self.temp_cal_scale[:, h_pos].unsqueeze(0))

                unit_whs.append(cum_unit_wh)
                raw_usig = F.softplus(self.unit_wh_sigma_head(d_state)) * 10.0 + 1.0
                unit_wh_sigmas.append(raw_usig * self.unit_wh_cal_scale[:, h_pos].unsqueeze(0))

                tot_wh = cum_unit_wh.sum(dim=-1)
                total_whs.append(tot_wh)
                raw_tot_sig = F.softplus(self.total_wh_sigma_head(d_state)).squeeze(-1) * 20.0 + 2.0
                total_wh_sigmas.append(raw_tot_sig * self.total_wh_cal_scale[h_pos])

        return {
            "temp_delta": torch.stack(temp_deltas, dim=-1),  # [B, Z, K]
            "temp_sigma": torch.stack(temp_sigmas, dim=-1),  # [B, Z, K]
            "unit_wh": torch.stack(unit_whs, dim=-1),  # [B, U, K]
            "unit_wh_sigma": torch.stack(unit_wh_sigmas, dim=-1),  # [B, U, K]
            "total_wh": torch.stack(total_whs, dim=-1),  # [B, K]
            "total_wh_sigma": torch.stack(total_wh_sigmas, dim=-1),  # [B, K]
            "step_thermal_q": torch.stack(step_q_list, dim=1),  # [B, F, U]
            "step_elec_wh": torch.stack(step_e_list, dim=1),  # [B, F, U]
        }


@dataclass
class ModelArtifact:
    """Loaded model bundle with strict provenance, lifecycle state, and version contract."""

    model: ThermostatShadowGRU
    normalizer: FeatureNormalizer | None
    model_state: ModelState
    metadata: dict[str, Any]

    @property
    def model_id(self) -> str:
        return str(self.metadata.get("model_id", "untrained"))

    @property
    def training_cutoff_ts(self) -> float | None:
        val = self.metadata.get("training_cutoff_ts")
        return float(val) if val is not None else None


def build_untrained_artifact(site: SiteConfig, spec: FeatureSpec, *, seed: int = 42) -> ModelArtifact:
    """Instantiate a randomly initialized model explicitly tagged as ModelState.UNTRAINED."""
    torch.manual_seed(seed)
    model = ThermostatShadowGRU(
        hist_dim=spec.hist_dim,
        act_dim=spec.act_dim,
        n_zones=spec.n_zones,
        n_units=spec.n_units,
        horizon_step_indices=spec.horizon_step_indices,
        influence_matrix=site.influence_matrix(),
        coupling_matrix=site.coupling_matrix(),
    )
    meta = {
        "model_id": f"untrained-{seed}",
        "model_state": ModelState.UNTRAINED.value,
        "model_arch_version": MODEL_ARCH_VERSION,
        "feature_version": spec.feature_version,
        "config_fingerprint": site.config_fingerprint,
        "observation_version": site.observation_version,
        "parameter_count": model.parameter_count(),
        "trainable_parameter_count": model.trainable_parameter_count(),
        "training_cutoff_ts": None,
        "data_origin": None,
        "seed": seed,
        "readiness_reasons": ["untrained_random_weights"],
        "feature_spec": spec.to_dict(),
    }
    return ModelArtifact(model=model, normalizer=None, model_state=ModelState.UNTRAINED, metadata=meta)


def save_checkpoint(artifact: ModelArtifact, path: Path) -> Path:
    """Persist model weights, normalizer, version contract, and lifecycle state."""
    if artifact.model_state == ModelState.READY_TO_USE and artifact.normalizer is None:
        raise ValueError("Refusing to save READY_TO_USE model without train-fitted normalizer")
    if artifact.model_state == ModelState.READY_TO_USE and artifact.metadata.get("training_cutoff_ts") is None:
        raise ValueError("Refusing to save READY_TO_USE model without training_cutoff_ts")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_arch_version": artifact.metadata.get("model_arch_version", MODEL_ARCH_VERSION),
        "feature_version": artifact.metadata.get("feature_version", FEATURE_VERSION),
        "model_state": artifact.model_state.value,
        "model_config": artifact.model.config_dict(),
        "state_dict": artifact.model.state_dict(),
        "normalizer": artifact.normalizer.to_dict() if artifact.normalizer is not None else None,
        "metadata": artifact.metadata,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)

    # Also write human-readable sidecar metadata JSON for inspection without loading PyTorch tensors.
    digest = sha256(path.read_bytes()).hexdigest()
    sidecar = dict(artifact.metadata, checkpoint_file=path.name, checkpoint_sha256=digest)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_checkpoint(
    path: Path,
    *,
    expected_site: SiteConfig | None = None,
    expected_spec: FeatureSpec | None = None,
    expected_arch_version: str = MODEL_ARCH_VERSION,
    expected_feature_version: str = FEATURE_VERSION,
    strict: bool = True,
) -> ModelArtifact:
    """Load a saved checkpoint and strictly validate version and site compatibility."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    arch_ver = payload.get("model_arch_version")
    feat_ver = payload.get("feature_version")
    meta = dict(payload.get("metadata") or {})
    mcfg = payload["model_config"]

    mismatches: list[str] = []
    if arch_ver != expected_arch_version:
        mismatches.append(f"arch_version_mismatch:{arch_ver}!={expected_arch_version}")
    if feat_ver != expected_feature_version:
        mismatches.append(f"feature_version_mismatch:{feat_ver}!={expected_feature_version}")
    if expected_site is not None and meta.get("config_fingerprint") != expected_site.config_fingerprint:
        mismatches.append(
            f"config_fingerprint_mismatch:{meta.get('config_fingerprint')}!={expected_site.config_fingerprint}"
        )
    if expected_spec is not None:
        if mcfg["hist_dim"] != expected_spec.hist_dim or mcfg["act_dim"] != expected_spec.act_dim:
            mismatches.append(
                f"feature_dim_mismatch:({mcfg['hist_dim']},{mcfg['act_dim']})!=({expected_spec.hist_dim},{expected_spec.act_dim})"
            )
        if mcfg["n_zones"] != expected_spec.n_zones or mcfg["n_units"] != expected_spec.n_units:
            mismatches.append(
                f"topology_dim_mismatch:({mcfg['n_zones']},{mcfg['n_units']})!=({expected_spec.n_zones},{expected_spec.n_units})"
            )
        if tuple(mcfg["horizon_step_indices"]) != expected_spec.horizon_step_indices:
            mismatches.append("horizon_mismatch")

    if mismatches and strict:
        raise ModelVersionMismatchError("; ".join(mismatches))

    model = ThermostatShadowGRU(**mcfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    norm = FeatureNormalizer.from_dict(payload["normalizer"]) if payload.get("normalizer") is not None else None
    state = ModelState(payload.get("model_state", ModelState.UNTRAINED.value))
    if norm is None and state == ModelState.READY_TO_USE:
        state = ModelState.UNTRAINED
    if mismatches:
        meta["version_mismatches"] = mismatches
    return ModelArtifact(model=model, normalizer=norm, model_state=state, metadata=meta)


@torch.no_grad()
def predict_numpy(
    artifact: ModelArtifact,
    X_hist: np.ndarray,
    X_act: np.ndarray,
    *,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Run batched CPU inference returning numpy arrays."""
    if artifact.normalizer is not None:
        xh = artifact.normalizer.transform_hist(X_hist)
        xa = artifact.normalizer.transform_act(X_act)
    else:
        xh = np.nan_to_num(X_hist, nan=0.0).astype(np.float32)
        xa = np.nan_to_num(X_act, nan=0.0).astype(np.float32)

    artifact.model.eval()
    n = xh.shape[0]
    if n == 0:
        n_z = artifact.model.n_zones
        n_u = artifact.model.n_units
        n_k = artifact.model.n_horizons
        return {
            "temp_delta": np.zeros((0, n_z, n_k), dtype=np.float32),
            "temp_sigma": np.zeros((0, n_z, n_k), dtype=np.float32),
            "unit_wh": np.zeros((0, n_u, n_k), dtype=np.float32),
            "unit_wh_sigma": np.zeros((0, n_u, n_k), dtype=np.float32),
            "total_wh": np.zeros((0, n_k), dtype=np.float32),
            "total_wh_sigma": np.zeros((0, n_k), dtype=np.float32),
        }

    buckets: dict[str, list[np.ndarray]] = {
        "temp_delta": [],
        "temp_sigma": [],
        "unit_wh": [],
        "unit_wh_sigma": [],
        "total_wh": [],
        "total_wh_sigma": [],
    }
    for i in range(0, n, batch_size):
        bh = torch.from_numpy(xh[i : i + batch_size])
        ba = torch.from_numpy(xa[i : i + batch_size])
        out = artifact.model(bh, ba)
        for key in buckets:
            buckets[key].append(out[key].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in buckets.items()}
