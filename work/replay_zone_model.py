"""Replay hourly evidence packs through the zone model and sensor fusion without any HA calls.

Usage:
    python3 -B work/replay_zone_model.py evidence-*.zip [--fusion estimated] [--json report.json]

Each control cycle in raw/live-audit.jsonl supplies the zone readings, the rule
trend the controller used, environment readings and each unit's mode and power.
Interval power comes from raw/power-reports.jsonl. The report compares settled
10-minute forecasts of the model with persistence and the logged rule trend,
lists learned influence, and shows how auxiliary sensors would be fused.
"""
import argparse
import bisect
from collections import defaultdict
import json
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "home-assistant/shadow"))
from fusion import fuse  # noqa: E402
from service import POWER_REPORT_GAP_SECONDS, metered_interval  # noqa: E402
from topology import fingerprint  # noqa: E402
import zone_model  # noqa: E402

POLICY = Path(__file__).resolve().parents[1] / "home-assistant/shadow/house_policy.json"


def read_packs(paths):
    cycles, reports, actions, config, versions = {}, {}, set(), None, {}
    for path in sorted(paths):
        with ZipFile(path) as pack:
            names = set(pack.namelist())
            if "effective-config.json" in names:
                config = json.loads(pack.read("effective-config.json"))
            if "configuration-versions.json" in names:
                versions.update(json.loads(pack.read("configuration-versions.json")))
            for line in pack.read("raw/live-audit.jsonl").splitlines():
                event = json.loads(line)
                if event.get("event") == "cycle" and (event.get("assessment") or {}).get("zones"):
                    cycles[event["at"]] = event
            if "raw/power-reports.jsonl" in names:
                for line in pack.read("raw/power-reports.jsonl").splitlines():
                    row = json.loads(line)
                    reports[(row["source"], row["observer_epoch"], row["report_at"])] = row["watts"]
            if "raw/commands.jsonl" in names:
                for line in pack.read("raw/commands.jsonl").splitlines():
                    row = json.loads(line)
                    if row.get("status") == "proxy_echo" and row.get("proxy_at") is not None:
                        actions.add(row["proxy_at"])
    if config is None:
        raise SystemExit("No effective-config.json in the evidence packs")
    return [cycles[at] for at in sorted(cycles)], reports, sorted(actions), config, versions


def site(config, fusion):
    zones = {key: dict(zone, fusion=fusion or zone.get("fusion", "primary")) for key, zone in config["zones"].items()}
    return {"sensors": config["sensors"], "zones": zones, "actuators": config["actuators"],
            "sources": {key: sensor["source"] for key, sensor in config["sensors"].items()},
            "zone_versions": {key: fingerprint([key, zone]) for key, zone in zones.items()}}


def meters(configs, reports):
    sources = {binding.get("power"): unit for config in configs for unit, binding in config["bindings"].items()}
    by_unit = defaultdict(dict)
    for (source, _, at), watts in reports.items():
        if source in sources:
            by_unit[sources[source]][at] = watts
    return {unit: sorted(points.items()) for unit, points in by_unit.items()}


def replay(cycles, config, power_reports, policy, fusion=None, actions=(), versions=None):
    """Run each cycle with the configuration recorded for it, as the controller did."""
    versions, sites = versions or {}, {}
    local, fusion_memory = {}, defaultdict(dict)
    context = topology = None
    modes, identities, failover, held = {}, defaultdict(set), defaultdict(int), defaultdict(int)
    tallies = defaultdict(lambda: {"n": 0, "model": 0.0, "persistence": 0.0, "rule": 0.0, "covered": 0})
    pending = defaultdict(list)
    first_ready = {}
    stamps = {unit: [at for at, _ in points] for unit, points in power_reports.items()}

    def power(unit, start, end):
        points = power_reports.get(unit, [])
        lo = max(0, bisect.bisect_left(stamps.get(unit, []), start)-1)
        hi = bisect.bisect_right(stamps.get(unit, []), end)+1
        kwh, covered = metered_interval(points[lo:hi], start, end)
        return kwh*3600/covered if covered >= 0.8*(end-start) else None

    for event in cycles:
        now, assessment = event["at"], event["assessment"]
        version = event.get("config_sha256") if event.get("config_sha256") in versions else None
        if version not in sites:
            sites[version] = site(versions[version] if version else config, fusion)
        topology = sites[version]
        context = zone_model.begin(local, topology, policy)
        units_now = {}
        for unit, room in (event.get("rooms") or {}).items():
            if unit not in topology["actuators"]:
                continue
            mode = room.get("mode_reported")
            if modes.get(unit, (None,))[0] != mode:
                modes[unit] = (mode, now)
            age, watts = room.get("power_age_s"), room.get("power_w")
            fresh = watts is not None and age is not None and 0 <= age <= POWER_REPORT_GAP_SECONDS
            units_now[unit] = {"mode": mode, "changed_at": modes[unit][1], "kw": watts/1000 if fresh else None}
        environment = assessment.get("environment") or {}
        done = bisect.bisect_right(actions, now)
        intervened = actions[done-1] if done else 0
        changes = [intervened, *(row["changed_at"] for row in units_now.values())]
        for key, observed in assessment["zones"].items():
            if key not in topology["zones"] or not observed.get("primary_reading"):
                continue
            zone = topology["zones"][key]
            readings = {zone["primary"]: observed["primary_reading"]}
            readings.update({name: reading for name, reading in (observed.get("inputs") or {}).items()
                             if name in topology["sensors"]})
            ref, detail = fuse(zone, topology, readings, fusion_memory[key], now, policy)
            identities[key].add(json.dumps(ref["source"]))
            if detail:
                failover[key] += detail["failover"]
                held[key] += any(row.get("estimated") for row in detail["sensors"].values())
            previous = (local.get("model", {}).get("zones", {}).get(key) or {}).get("last") or {}
            modeled = zone_model.observe_zone(context, key, ref, observed.get("trend"), environment, units_now, power,
                                              now, intervened)
            state = local["model"]["zones"][key]
            if state["ready"] and key not in first_ready:
                first_ready[key] = now
            if ref["quality"] == "fresh" and ref["at"] != previous.get("at") and state["last"] and state["last"]["at"] == ref["at"]:
                keep = []
                for item in pending[key]:
                    age = ref["at"]-item["at"]
                    if age < 450:
                        keep.append(item)
                    elif age <= 1200 and max(changes) < item["at"]:
                        scale, actual = age/policy["forecast_seconds"], ref["value"]-item["value"]
                        tally = tallies[key]
                        tally["n"] += 1
                        tally["model"] += abs(actual-scale*item["model"])
                        tally["persistence"] += abs(actual)
                        tally["rule"] += abs(actual-scale*item["rule"])
                        tally["covered"] += abs(actual-scale*item["model"]) <= item["sigma"]*max(1, scale)
                pending[key] = keep
                if modeled["forecast"] is not None:
                    rule = observed.get("trend")
                    pending[key].append({"at": ref["at"], "value": ref["value"], "model": modeled["forecast"]["delta"],
                                         "sigma": modeled["forecast"]["uncertainty"], "rule": rule["delta"] if rule else 0.0})
    learned = zone_model.influence(local.get("model"), topology, policy) if topology else {}
    return {"zones": {key: {
                "updates": state["updates"], "settled": tallies[key]["n"],
                "mae_model": round(tallies[key]["model"]/max(1, tallies[key]["n"]), 3),
                "mae_persistence": round(tallies[key]["persistence"]/max(1, tallies[key]["n"]), 3),
                "mae_rule": round(tallies[key]["rule"]/max(1, tallies[key]["n"]), 3),
                "coverage": round(tallies[key]["covered"]/max(1, tallies[key]["n"]), 2),
                "ready": state["ready"], "first_ready_at": first_ready.get(key),
                "noise_c": round(state["noise"]**0.5, 3),
                "coefficients": dict(zip(state["features"], [round(v, 3) for v in state["theta"]])),
                "excited": state["excited"], "identities": len(identities[key]),
                "failover_cycles": failover[key], "held_offset_cycles": held[key],
                "offsets": {name: {k: row.get(k) for k in ("offset", "pairs")} for name, row in fusion_memory[key].items()
                            if isinstance(row, dict) and "pairs" in row}}
            for key, state in local.get("model", {}).get("zones", {}).items()},
            "influence": [{"unit": unit, "mode": mode, "zone": zone, "ratio": round(entry["ratio"], 3),
                           "std": round(entry["std"], 3), "status": entry["status"],
                           "configured": topology["actuators"][unit]["influences"].get(zone)}
                          for (unit, mode, zone), entry in sorted(learned.items())],
            "cycles": len(cycles), "from": cycles[0]["at"] if cycles else None, "to": cycles[-1]["at"] if cycles else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("packs", nargs="+", type=Path)
    parser.add_argument("--fusion", choices=("primary", "weighted", "estimated"),
                        help="override every zone's fusion mode for the replay")
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    args = parser.parse_args()
    cycles, reports, actions, config, versions = read_packs(args.packs)
    policy = json.loads(args.policy.read_text())
    report = replay(cycles, config, meters([config, *versions.values()], reports), policy, args.fusion, actions, versions)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"cycles {report['cycles']}  window {report['from']} .. {report['to']}")
    print("zone                      updates settled  MAE model  persistence  rule   coverage  ready  noise")
    for key, row in sorted(report["zones"].items()):
        print(f"{key:<26}{row['updates']:>7}{row['settled']:>8}  {row['mae_model']:>9}  {row['mae_persistence']:>11}"
              f"  {row['mae_rule']:>5}  {row['coverage']:>8}  {str(row['ready']):>5}  {row['noise_c']}")
    for key, row in sorted(report["zones"].items()):
        if row["offsets"]:
            print(f"{key}: identities {row['identities']}, held-offset cycles {row['held_offset_cycles']}, "
                  f"failover cycles {row['failover_cycles']}, offsets {row['offsets']}")
    print("learned influence (ratio to the unit's home-zone response)")
    for row in report["influence"]:
        print(f"  {row['unit']} {row['mode']:<4} -> {row['zone']}: {row['ratio']:+.3f} ± {row['std']:.3f}  "
              f"{row['status']:<11} configured {row['configured']}")


if __name__ == "__main__":
    main()
