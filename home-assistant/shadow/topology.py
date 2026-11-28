"""Site definitions for observations, control zones and equipment influence."""
from datetime import datetime
from hashlib import sha256
import json
from urllib.error import HTTPError

from model import numeric


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def report_time(state, *other):
    times = []
    for value in (state.get("last_reported"), *other):
        if value:
            times.append(datetime.fromisoformat(value).timestamp())
    return max(times, default=None)


def read_topology(path, rooms):
    return configure(json.loads(path.read_text()), rooms) if path.is_file() else None


def configure(definition, rooms):
    if definition["schema"] != 1 or not definition["observation_version"]:
        raise ValueError("Unsupported topology schema or missing observation version")
    sensors = definition["sensors"]
    zones = definition["zones"]
    actuators = definition["actuators"]
    if not zones or set(actuators) != set(rooms):
        raise ValueError("Topology and equipment bindings disagree")
    sources = {}
    for key, sensor in sensors.items():
        role = sensor["role"]
        if role not in {"primary", "auxiliary", "constraint", "environment"}:
            raise ValueError("Unknown sensor role")
        if sensor.get("enabled", True):
            selector = set(sensor) & {"from_actuator", "entity"}
            if len(selector) != 1 or ("from_actuator" in selector and sensor["from_actuator"] not in rooms):
                raise ValueError("Invalid enabled sensor source")
            sources[key] = sensor.get("entity") or rooms[sensor["from_actuator"]]["temperature"]
            if not 0 < sensor.get("max_age_seconds", 1200) <= 7200 or not 0 < sensor.get("weight", 1) <= 10:
                raise ValueError("Invalid sensor age or weight")
            offset = sensor.get("offset_c", 0)
            if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not -3 <= offset <= 3:
                raise ValueError("Invalid sensor calibration offset")
    primary_sources = set()
    for key, zone in zones.items():
        primary = zone["primary"]
        if primary not in sources or sensors[primary]["role"] != "primary" or sources[primary] in primary_sources:
            raise ValueError("Each zone needs an enabled, independent primary sensor")
        primary_sources.add(sources[primary])
        extras = zone.get("sensors", ())
        if (primary in extras or len(set(extras)) != len(extras)
                or any(sensors[item]["role"] not in {"auxiliary", "constraint"} for item in extras)
                or zone.get("fusion", "primary") not in {"primary", "weighted", "estimated"}):
            raise ValueError("Invalid zone sensor roles or fusion")
        target = zone["target"]
        if (len(set(target) & {"actuator", "entity", "celsius"}) != 1
                or ("actuator" in target and target["actuator"] not in rooms)
                or ("celsius" in target and not 16 <= target["celsius"] <= 30)):
            raise ValueError("Invalid zone target")
    for unit in actuators.values():
        impact = unit["influences"]
        if (unit["zone"] not in impact or any(key not in zones or not 0 < value <= 1 for key, value in impact.items())
                or not unit.get("lock_group")):
            raise ValueError("Invalid equipment zone, influence or lock group")
    links = {}
    for link in definition.get("connections", ()):
        a, b, coupling = link["from"], link["to"], link["coupling"]
        pair = tuple(sorted((a, b)))
        if a == b or a not in zones or b not in zones or pair in links or not 0 <= coupling <= 1:
            raise ValueError("Invalid zone connection")
        links[pair] = coupling
    zone_versions = {}
    for name, zone in zones.items():
        fusion = zone.get("fusion", "primary")
        selected = [item for item in zone.get("sensors", ()) if item in sources and
                    (sensors[item]["role"] == "constraint" or fusion in {"weighted", "estimated"})]
        sensor_definitions = {item: [sources[item], {k: v for k, v in sensors[item].items() if k != "enabled"}]
                              for item in [zone["primary"], *selected]}
        zone_versions[name] = fingerprint([name, zone["primary"], fusion, zone["target"], zone.get("pause_entity"),
                                           zone.get("observation_version"), sensor_definitions])
    action_versions = {}
    learning_versions = {}
    opposing = definition.get("opposite_operation", {})
    environment = {name: (sources[name], sensor) for name, sensor in sensors.items()
                   if name in sources and sensor["role"] == "environment"}
    for room, unit in actuators.items():
        affected = set(unit["influences"])
        relevant_links = [(pair, value) for pair, value in sorted(links.items()) if affected.intersection(pair)]
        relevant_zones = affected | {name for pair, _ in relevant_links for name in pair}
        binding = {k: v for k, v in rooms[room].items() if k not in {"name", "reference", "temperature"}}
        action_versions[room] = fingerprint([unit, binding, opposing, relevant_links,
                                             {key:zone_versions[key] for key in relevant_zones}])
        learning_versions[room] = fingerprint([unit, binding, zone_versions[unit["zone"]], environment])
    # The installation fingerprint is retained for audit and upgrade migration.
    return dict(definition, fingerprint=fingerprint([definition, sources]), sources=sources, links=links,
                zone_versions=zone_versions, action_versions=action_versions, learning_versions=learning_versions)


def safe_state(request, entity):
    try:
        return request("states/" + entity)
    except HTTPError as error:
        error.close()
        if error.code != 404:
            raise
        return {}


def extra_inputs(request, topology):
    if topology is None:
        return {}
    entities = {s["entity"] for s in topology["sensors"].values() if s.get("enabled", True) and "entity" in s}
    entities.update(z["target"]["entity"] for z in topology["zones"].values() if "entity" in z["target"])
    entities.update(z["pause_entity"] for z in topology["zones"].values() if z.get("pause_entity"))
    return {entity: safe_state(request, entity) for entity in sorted(entities)}


def sensor_reading(sensor, live, inputs, now):
    environment = sensor["role"] == "environment"
    if via := sensor.get("from_actuator"):
        row = live[via]
        if environment:
            value, at, source = row.get("outdoor"), row.get("outdoor_report_at"), row.get("outdoor_source")
        else:
            value, at, source = row["temperature"], row.get("temperature_report_at"), row.get("temperature_source")
    else:
        state = inputs.get(sensor["entity"], {})
        value = state.get("attributes", {}).get(sensor["attribute"]) if sensor.get("attribute") else state.get("state")
        at, source = report_time(state), [sensor["entity"], "ha_report"]
    value = numeric(value)
    if value is not None and sensor.get("offset_c"):
        value += sensor["offset_c"]
    valid_value = value is not None and (-50 <= value <= 65 if environment else 0 < value <= 50)
    valid = valid_value and source is not None and at is not None and 0 <= now-at <= sensor.get("max_age_seconds", 1200)
    return {"value": value, "at": at, "source": source, "quality": "fresh" if valid else
            "stale" if valid_value and at is not None else "unavailable"}


def zone_target(zone, live, inputs):
    info = zone["target"]
    if unit := info.get("actuator"):
        value = live[unit]["signature"][4]
    elif entity := info.get("entity"):
        row = inputs.get(entity, {})
        value = row.get("attributes", {}).get(info["attribute"]) if info.get("attribute") else row.get("state")
    else:
        value = info["celsius"]
    value = numeric(value)
    return value if value is not None and 16 <= value <= 30 else None
