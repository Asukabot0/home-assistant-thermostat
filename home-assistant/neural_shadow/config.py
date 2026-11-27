"""Site topology, sensor/zone/actuator mapping, and neural shadow configuration.

Reuses home-assistant/shadow/topology.py and house_policy.json directly so
zone definitions, shared sensors, soft limits, and observation versions stay aligned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SHADOW_DIR = REPO_ROOT / "home-assistant" / "shadow"
TELEMETRY_DIR = REPO_ROOT / "home-assistant" / "telemetry"


def _load_shadow_module(name: str):
    if str(SHADOW_DIR) not in sys.path:
        sys.path.insert(0, str(SHADOW_DIR))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SHADOW_DIR / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to locate shadow module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_telemetry_module(name: str):
    if str(TELEMETRY_DIR) not in sys.path:
        sys.path.insert(0, str(TELEMETRY_DIR))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(f"telemetry_{name}", TELEMETRY_DIR / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to locate telemetry module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"telemetry_{name}"] = module
    spec.loader.exec_module(module)
    return module


shadow_model = _load_shadow_module("model")
shadow_topology = _load_shadow_module("topology")
shadow_policy = _load_shadow_module("house_policy")
telemetry_evidence = _load_telemetry_module("evidence")


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(raw).hexdigest()


@dataclass(frozen=True)
class ZoneMapping:
    """An independently measured temperature zone (label target)."""

    name: str
    primary_sensor: str
    primary_entity: str
    max_age_seconds: int = 1200
    target_actuator: str | None = None
    target_entity: str | None = None
    target_celsius: float | None = None
    auxiliary_sensors: tuple[str, ...] = ()
    constraint_sensors: tuple[str, ...] = ()
    pause_entity: str | None = None
    source_version: str = ""


@dataclass(frozen=True)
class ActuatorMapping:
    """A controllable indoor/outdoor AC unit."""

    name: str
    home_zone: str
    influences: dict[str, float]
    adapter_entity: str
    power_entity: str
    power_fallback: str | None
    temperature_entity: str
    soft_limits: dict[str, tuple[float, float]]
    lock_group: str
    priority: int = 0
    verified_off_ceiling_w: float | None = None


@dataclass(frozen=True)
class EnvironmentSensorMapping:
    """Outdoor or ambient reference sensor (e.g., weather reference, future balcony sensor)."""

    name: str
    entity: str | None
    from_actuator: str | None
    enabled: bool
    max_age_seconds: int = 7200
    attribute: str | None = None


@dataclass(frozen=True)
class SiteConfig:
    """Validated site mapping separating independent measurement zones from actuators."""

    observation_version: str
    topology_fingerprint: str
    config_fingerprint: str
    zones: tuple[ZoneMapping, ...]
    actuators: tuple[ActuatorMapping, ...]
    environment_sensors: tuple[EnvironmentSensorMapping, ...]
    connections: tuple[tuple[str, str, float], ...]
    opposite_operation: dict[str, Any]
    policy: dict[str, Any]
    raw_rooms: dict[str, Any] = field(repr=False)
    raw_topology: dict[str, Any] = field(repr=False)

    @property
    def zone_names(self) -> tuple[str, ...]:
        return tuple(z.name for z in self.zones)

    @property
    def unit_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.actuators)

    @property
    def active_environment_sensors(self) -> tuple[EnvironmentSensorMapping, ...]:
        return tuple(s for s in self.environment_sensors if s.enabled)

    def zone_by_name(self, name: str) -> ZoneMapping:
        for z in self.zones:
            if z.name == name:
                return z
        raise KeyError(f"Unknown zone: {name}")

    def actuator_by_name(self, name: str) -> ActuatorMapping:
        for a in self.actuators:
            if a.name == name:
                return a
        raise KeyError(f"Unknown actuator: {name}")

    def influence_matrix(self) -> list[list[float]]:
        """Return [n_zones, n_units] static influence weights from topology."""
        return [[float(a.influences.get(z.name, 0.0)) for a in self.actuators] for z in self.zones]

    def coupling_matrix(self) -> list[list[float]]:
        """Return symmetric [n_zones, n_zones] inter-zone thermal coupling matrix."""
        idx = {z.name: i for i, z in enumerate(self.zones)}
        n = len(self.zones)
        mat = [[0.0] * n for _ in range(n)]
        for a, b, w in self.connections:
            if a in idx and b in idx:
                mat[idx[a]][idx[b]] = float(w)
                mat[idx[b]][idx[a]] = float(w)
        return mat

    def summary_dict(self) -> dict[str, Any]:
        return {
            "observation_version": self.observation_version,
            "topology_fingerprint": self.topology_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "zones": [z.name for z in self.zones],
            "actuators": [a.name for a in self.actuators],
            "actuator_home_zones": {a.name: a.home_zone for a in self.actuators},
            "active_environment_sensors": [s.name for s in self.active_environment_sensors],
        }


def _parse_soft_limits(raw: dict[str, Any] | None) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for mode in ("heat", "cool"):
        pair = (raw or {}).get(mode)
        if pair and len(pair) == 2:
            out[mode] = (float(pair[0]), float(pair[1]))
        else:
            out[mode] = (20.0, 25.0) if mode == "heat" else (22.0, 26.0)
    return out


def build_site_config(
    rooms: dict[str, Any],
    topology_def: dict[str, Any],
    policy: dict[str, Any] | None = None,
) -> SiteConfig:
    """Build SiteConfig from rooms.json + topology.json + house_policy.json."""
    resolved_policy = dict(shadow_policy.POLICY)
    if policy:
        resolved_policy.update(policy)
    configured = shadow_topology.configure(topology_def, rooms)
    sensors = configured["sensors"]
    sources = configured["sources"]

    # Preserve deterministic ordering: sorted zones and units, or canonical bedroom/living/study order when present.
    zone_order = sorted(configured["zones"].keys())
    unit_order = sorted(configured["actuators"].keys())
    if set(unit_order) == {"bedroom", "living", "study"}:
        unit_order = ["bedroom", "living", "study"]
    if set(zone_order) == {"sleep", "lounge"}:
        zone_order = ["sleep", "lounge"]

    zones: list[ZoneMapping] = []
    seen_primary_entities: set[str] = set()
    for zname in zone_order:
        zspec = configured["zones"][zname]
        pkey = zspec["primary"]
        pentity = sources[pkey]
        if pentity in seen_primary_entities:
            raise ValueError(f"Duplicate independent primary sensor entity {pentity} across zones")
        seen_primary_entities.add(pentity)
        target_spec = zspec["target"]
        extras = zspec.get("sensors", ())
        aux = tuple(k for k in extras if k in sources and sensors[k]["role"] == "auxiliary")
        cons = tuple(k for k in extras if k in sources and sensors[k]["role"] == "constraint")
        zones.append(
            ZoneMapping(
                name=zname,
                primary_sensor=pkey,
                primary_entity=pentity,
                max_age_seconds=int(sensors[pkey].get("max_age_seconds", 1200)),
                target_actuator=target_spec.get("actuator"),
                target_entity=target_spec.get("entity"),
                target_celsius=float(target_spec["celsius"]) if "celsius" in target_spec else None,
                auxiliary_sensors=aux,
                constraint_sensors=cons,
                pause_entity=zspec.get("pause_entity"),
                source_version=configured["zone_versions"][zname],
            )
        )

    actuators: list[ActuatorMapping] = []
    for uname in unit_order:
        uspec = configured["actuators"][uname]
        rcfg = rooms[uname]
        off_ref = rcfg.get("verified_off_power") or {}
        off_entity = off_ref.get("entity") if isinstance(off_ref, dict) else None
        ceiling = (
            shadow_model.numeric(off_ref.get("ceiling_w"))
            if isinstance(off_ref, dict) and off_entity in {None, rcfg["power"]}
            else None
        )
        actuators.append(
            ActuatorMapping(
                name=uname,
                home_zone=uspec["zone"],
                influences={k: float(v) for k, v in uspec["influences"].items()},
                adapter_entity=rcfg["adapter"],
                power_entity=rcfg["power"],
                power_fallback=rcfg.get("power_fallback"),
                temperature_entity=rcfg["temperature"],
                soft_limits=_parse_soft_limits(rcfg.get("soft_limits")),
                lock_group=uspec["lock_group"],
                priority=int(uspec.get("priority", 0)),
                verified_off_ceiling_w=ceiling if ceiling is not None and ceiling > 0 else None,
            )
        )

    env_sensors: list[EnvironmentSensorMapping] = []
    for sname in sorted(sensors.keys()):
        sdef = sensors[sname]
        if sdef["role"] != "environment":
            continue
        enabled = bool(sdef.get("enabled", True))
        env_sensors.append(
            EnvironmentSensorMapping(
                name=sname,
                entity=sdef.get("entity"),
                from_actuator=sdef.get("from_actuator"),
                enabled=enabled,
                max_age_seconds=int(sdef.get("max_age_seconds", 7200)),
                attribute=sdef.get("attribute"),
            )
        )

    connections = tuple(
        (str(link["from"]), str(link["to"]), float(link["coupling"]))
        for link in configured.get("connections", ())
    )
    fingerprint_payload = {
        "observation_version": configured["observation_version"],
        "topology_fingerprint": configured["fingerprint"],
        "zones": [(z.name, z.primary_sensor, z.primary_entity) for z in zones],
        "actuators": [
            (
                a.name,
                a.home_zone,
                sorted(a.influences.items()),
                a.soft_limits,
                a.power_entity,
                a.power_fallback,
                a.verified_off_ceiling_w,
            )
            for a in actuators
        ],
        "env": [(e.name, e.enabled) for e in env_sensors],
    }
    return SiteConfig(
        observation_version=str(configured["observation_version"]),
        topology_fingerprint=str(configured["fingerprint"]),
        config_fingerprint=canonical_hash(fingerprint_payload),
        zones=tuple(zones),
        actuators=tuple(actuators),
        environment_sensors=tuple(env_sensors),
        connections=connections,
        opposite_operation=dict(configured.get("opposite_operation", {"enabled": False})),
        policy=resolved_policy,
        raw_rooms=rooms,
        raw_topology=topology_def,
    )


def load_site_config_from_files(
    rooms_path: Path,
    topology_path: Path,
    policy_path: Path | None = None,
) -> SiteConfig:
    rooms = json.loads(rooms_path.read_text(encoding="utf-8"))
    topology_def = json.loads(topology_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path and policy_path.is_file() else None
    return build_site_config(rooms, topology_def, policy)


def load_site_config_from_effective_telemetry(effective: dict[str, Any]) -> SiteConfig:
    """Reconstruct SiteConfig from an anonymized telemetry archive's effective-config.json."""
    bindings = effective.get("bindings", {})
    sensors = effective.get("sensors", {})
    zones_raw = effective.get("zones", {})
    actuators_raw = effective.get("actuators", {})

    # Reconstruct rooms + topology shapes compatible with shadow_topology.configure().
    rooms: dict[str, Any] = {}
    zone_primary_source: dict[str, str] = {}
    for zname, zspec in zones_raw.items():
        pkey = zspec["primary"]
        zone_primary_source[zname] = sensors[pkey].get("source") or f"sensor.{zname}"

    for uname, bind in bindings.items():
        uzone = actuators_raw[uname]["zone"]
        off_ref = bind.get("off_reference") or {}
        rooms[uname] = {
            "name": uname,
            "temperature": zone_primary_source.get(uzone, f"sensor.{uname}_temp"),
            "power": bind["power"],
            "power_fallback": bind.get("fallback"),
            "adapter": bind["adapter"],
            "reference": uzone,
            "soft_limits": bind.get("soft_limits", {"cool": [22, 26], "heat": [20, 25]}),
            "transition_off_seconds": bind.get("transition_off_seconds", 240),
            "overshoot_high_power_seconds": bind.get("overshoot_high_power_seconds", 180),
        }
        if off_ref.get("ceiling_w") is not None:
            rooms[uname]["verified_off_power"] = {
                "entity": bind["power"],
                "ceiling_w": off_ref["ceiling_w"],
            }

    topo_sensors: dict[str, Any] = {}
    for sname, sspec in sensors.items():
        entry: dict[str, Any] = {
            "role": sspec["role"],
            "enabled": bool(sspec.get("enabled", True)),
            "max_age_seconds": sspec.get("max_age_seconds", 1200),
            "weight": sspec.get("weight", 1),
        }
        if entry["enabled"]:
            src = sspec.get("source")
            if src:
                entry["entity"] = src
            else:
                # Environment reference derived from room samples.
                first_unit = sorted(rooms.keys())[0]
                entry["from_actuator"] = first_unit
        topo_sensors[sname] = entry

    topo_zones: dict[str, Any] = {}
    for zname, zspec in zones_raw.items():
        tgt = dict(zspec.get("target") or {})
        if not (set(tgt) & {"actuator", "entity", "celsius"}):
            tgt = {"celsius": 24.0}
        topo_zones[zname] = {
            "primary": zspec["primary"],
            "sensors": list(zspec.get("sensors", ())),
            "fusion": zspec.get("fusion", "primary"),
            "target": tgt,
        }
        if zspec.get("pause_input"):
            topo_zones[zname]["pause_entity"] = zspec["pause_input"]

    topology_def = {
        "schema": int(effective.get("schema", 1)),
        "observation_version": str(effective.get("observation_version", "telemetry-export")),
        "sensors": topo_sensors,
        "zones": topo_zones,
        "actuators": actuators_raw,
        "connections": effective.get("connections", []),
        "opposite_operation": effective.get("opposite_operation", {"enabled": False, "max_cross_influence": 0.1}),
    }
    return build_site_config(rooms, topology_def, effective.get("policy"))


def load_site_config_from_acceptance(redacted: dict[str, Any]) -> SiteConfig:
    """Reconstruct SiteConfig from an older acceptance ZIP's effective-config.redacted.json."""
    bindings = redacted.get("room_bindings", {})
    example_rooms = json.loads((SHADOW_DIR / "rooms.example.json").read_text(encoding="utf-8"))
    example_topo = json.loads((SHADOW_DIR / "topology.example.json").read_text(encoding="utf-8"))
    rooms: dict[str, Any] = {}
    for room in ("bedroom", "living", "study"):
        base = dict(example_rooms[room])
        if room in bindings:
            rb = bindings[room]
            if rb.get("soft_limits"):
                base["soft_limits"] = rb["soft_limits"]
            if rb.get("power_entity_hash"):
                base["power"] = rb["power_entity_hash"]
        rooms[room] = base
    return build_site_config(rooms, example_topo, redacted.get("policy"))


def load_default_example_config() -> SiteConfig:
    return load_site_config_from_files(
        SHADOW_DIR / "rooms.example.json",
        SHADOW_DIR / "topology.example.json",
        SHADOW_DIR / "house_policy.json",
    )
