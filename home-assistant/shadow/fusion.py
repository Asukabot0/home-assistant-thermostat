"""Fuse a zone's primary and auxiliary temperature sensors with learned offsets."""
from math import inf


FUSED = {"weighted", "estimated"}
FUSION_DEFAULTS = {"fusion_offset_min_pairs": 6, "fusion_hold_seconds": 7200, "fusion_failover_seconds": 3600}
OFFSET_ALPHA = 0.1
OFFSET_STEP_LIMIT_C = 1.0
PAIR_SECONDS = 900


def setting(policy, key):
    return policy.get(key, FUSION_DEFAULTS[key])


def calibrated(reading, sensor):
    """Apply a sensor's commissioned calibration offset."""
    offset = sensor.get("offset_c", 0)
    if not offset or reading["value"] is None:
        return reading
    return dict(reading, value=reading["value"] + offset)


def auxiliaries(zone, topology, by_sensor):
    return sorted(key for key in zone.get("sensors", ())
                  if key in by_sensor and topology["sensors"][key]["role"] == "auxiliary")


def learn_offsets(zone, topology, by_sensor, memory):
    """Track each auxiliary sensor's offset from the primary using reports taken close together.

    A pair is formed once per new report of either sensor, so cached values
    cannot be counted twice. Offsets are learned in every fusion mode so they
    are already established when an operator switches a zone to fusion.
    """
    sensors = topology["sensors"]
    primary = calibrated(by_sensor[zone["primary"]], sensors[zone["primary"]])
    for key in auxiliaries(zone, topology, by_sensor):
        reading = calibrated(by_sensor[key], sensors[key])
        definition = [topology["sources"][key], sensors[key].get("offset_c", 0),
                      topology["sources"][zone["primary"]], sensors[zone["primary"]].get("offset_c", 0)]
        row = memory.get(key)
        if row is None or row.get("definition") != definition:
            row = memory[key] = {"definition": definition, "pairs": 0}
        if primary["quality"] != "fresh" or reading["quality"] != "fresh":
            continue
        stamps = [reading["at"], primary["at"]]
        if abs(stamps[0]-stamps[1]) > PAIR_SECONDS or stamps == row.get("paired"):
            continue
        difference = reading["value"]-primary["value"]
        if row["pairs"]:
            alpha = max(OFFSET_ALPHA, 1/(row["pairs"]+1))
            step = max(-OFFSET_STEP_LIMIT_C, min(OFFSET_STEP_LIMIT_C, difference-row["offset"]))
            row["offset"] = round(row["offset"]+alpha*step, 4)
        else:
            row["offset"] = round(difference, 4)
        row.update(pairs=row["pairs"]+1, paired=stamps, at=max(stamps))


def fuse(zone, topology, by_sensor, memory, now, policy):
    """Return the zone reading used for control and the per-sensor detail kept in the trace.

    `primary` keeps the primary reading and identity unchanged. `weighted` and
    `estimated` average the primary with auxiliary sensors. A stale auxiliary
    with an established offset is held at primary + offset, so a dropout neither
    moves the value nor changes the zone identity. `estimated` additionally
    dates the zone by the latest contributing report and, for a bounded time,
    replaces a stale primary with calibrated auxiliaries.
    """
    fusion = zone.get("fusion", "primary")
    sensors = topology["sensors"]
    primary = calibrated(by_sensor[zone["primary"]], sensors[zone["primary"]])
    learn_offsets(zone, topology, by_sensor, memory)
    keys = auxiliaries(zone, topology, by_sensor)
    detail = {"mode": fusion, "failover": False,
              "sensors": {key: {"offset": memory[key].get("offset"), "pairs": memory[key]["pairs"],
                                "at": memory[key].get("at")} for key in keys}} if keys else None
    if fusion not in FUSED:
        return primary, detail
    if primary["source"] is not None:
        memory["primary_source"] = primary["source"]
    ref = dict(primary, source=[memory.get("primary_source", primary["source"]), keys])
    if detail is None:
        return ref, detail
    minimum, hold = setting(policy, "fusion_offset_min_pairs"), setting(policy, "fusion_hold_seconds")

    def learned(key):
        row = memory[key]
        return row["pairs"] >= minimum and now-row.get("at", -inf) <= hold

    fresh = {key: calibrated(by_sensor[key], sensors[key]) for key in keys if by_sensor[key]["quality"] == "fresh"}
    if primary["quality"] == "fresh":
        anchor = primary["value"]
    else:
        usable = {key: reading for key, reading in fresh.items() if learned(key)}
        if (fusion != "estimated" or not usable or primary["at"] is None
                or now-primary["at"] > setting(policy, "fusion_failover_seconds")):
            return ref, detail
        weights = {key: sensors[key].get("weight", 1) for key in usable}
        anchor = sum(weights[key]*(usable[key]["value"]-memory[key]["offset"]) for key in usable)/sum(weights.values())
        ref.update(quality="fresh", at=max(reading["at"] for reading in usable.values()))
        detail["failover"] = True
    parts = [(anchor, 1.0)]
    for key in keys:
        weight = sensors[key].get("weight", 1)
        if key in fresh:
            parts.append((fresh[key]["value"], weight))
            detail["sensors"][key]["estimated"] = False
        elif learned(key):
            parts.append((anchor+memory[key]["offset"], weight))
            detail["sensors"][key]["estimated"] = True
    ref["value"] = sum(value*weight for value, weight in parts)/sum(weight for _, weight in parts)
    if fusion == "estimated" and primary["quality"] == "fresh" and fresh:
        ref["at"] = max([primary["at"], *(reading["at"] for reading in fresh.values())])
    return ref, detail
