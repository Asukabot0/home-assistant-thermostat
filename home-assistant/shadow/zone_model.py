"""Learn each zone's thermal response online and report calibrated forecasts and equipment influence.

Between two temperature reports a zone is modeled as

    dT = dh * (sum_e a_e*(T_e - T) + b*(T_m - T) + sum_u sum_m g_um*P_um + c)

where T_e are environment readings, T_m is a slow lagged average of the zone
temperature standing in for walls and furniture, P_um is the measured mean
electric power (kW) of unit u while it reported mode m, dh is the interval in
hours and c is a slowly varying drift. A Kalman filter with random-walk
coefficients starts from the configured influence matrix. Forecasts assume the
current output continues; they are scored only over windows without a control
action or mode change, against persistence and the rule trend, before the
controller may use them.
"""
from math import exp, inf, isfinite, sqrt

from topology import fingerprint


MODEL_VERSION = 1
MODES = ("heat", "cool")
SIGN = {"heat": 1, "cool": -1}
MODEL_DEFAULTS = {"model_active": 0, "model_min_samples": 48, "model_ready_margin": 0.9, "model_prior_gain": 8,
                  "model_max_interval_seconds": 1800, "influence_negligible": 0.15, "influence_significant": 0.3,
                  "influence_min_excitation": 6}
MIN_INTERVAL_SECONDS = 120
SETTLE_SECONDS = (450, 1200)
ENVIRONMENT_HOLD_SECONDS = 7200
MASS_SECONDS = 2700
STATS_WINDOW = 48
PENDING_SPACING_SECONDS = 240
EXCITATION_KW = 0.15
HUBER = 3.0
INITIAL_NOISE_C = 0.08
NOISE_FLOOR_C, NOISE_CEILING_C = 0.03, 0.5
NOISE_ADAPTATION = 0.05
# Prior mean, standard deviation and hourly random-walk standard deviation of the non-equipment coefficients.
ENVELOPE = (0.05, 0.1, 0.005)
MASS = (0.5, 1.5, 0.03)
BIAS = (0.0, 1.0, 0.15)
GAIN_DRIFT_FRACTION = 0.0125


def setting(policy, key):
    return policy.get(key, MODEL_DEFAULTS[key])


def active(policy):
    return bool(setting(policy, "model_active"))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def matvec(m, v):
    return [dot(row, v) for row in m]


def compact(value):
    return float(f"{value:.6g}")


def features(environment, units):
    return ([f"env:{key}" for key in environment] + ["mass"]
            + [f"{unit}:{mode}" for unit in units for mode in MODES] + ["drift"])


def prior(topology, key, environment, units, policy):
    """Prior mean, variance and hourly drift variance of each coefficient."""
    gain = setting(policy, "model_prior_gain")
    rows = [(ENVELOPE[0]/len(environment), ENVELOPE[1], ENVELOPE[2]) for _ in environment] + [MASS]
    for unit in units:
        influence = topology["actuators"][unit]["influences"].get(key, 0)
        rows += [(SIGN[mode]*gain*influence, gain, GAIN_DRIFT_FRACTION*gain) for mode in MODES]
    rows.append(BIAS)
    return [row[0] for row in rows], [row[1]**2 for row in rows], [row[2]**2 for row in rows]


def structure(topology, key, environment, units):
    return fingerprint([MODEL_VERSION, topology["zone_versions"][key],
                        [[name, topology["sources"][name]] for name in environment], units])


def begin(local, topology, policy):
    """Return the learning context of one assessment."""
    environment = sorted(name for name, sensor in topology["sensors"].items()
                         if sensor["role"] == "environment" and sensor.get("enabled", True))
    state = local.get("model")
    if not isinstance(state, dict) or state.get("version") != MODEL_VERSION:
        state = local["model"] = {"version": MODEL_VERSION, "zones": {}}
    for removed in state["zones"].keys()-topology["zones"].keys():
        del state["zones"][removed]
    return {"state": state, "topology": topology, "policy": policy, "environment": environment,
            "units": sorted(topology["actuators"])}


def zone_state(context, key):
    topology, policy = context["topology"], context["policy"]
    environment, units = context["environment"], context["units"]
    shape = structure(topology, key, environment, units)
    zone = context["state"]["zones"].get(key)
    if zone is None or zone.get("structure") != shape:
        names = features(environment, units)
        mean, variance, _ = prior(topology, key, environment, units, policy)
        covariance = [[variance[i] if i == j else 0.0 for j in range(len(mean))] for i in range(len(mean))]
        kept = {}
        if zone is not None and zone.get("zone_version") == topology["zone_versions"][key]:
            # The zone measures the same thing; carry over the coefficients that remain.
            old = {name: i for i, name in enumerate(zone["features"])}
            kept = {i: old[name] for i, name in enumerate(names) if name in old}
            for i, a in kept.items():
                mean[i] = zone["theta"][a]
                for j, b in kept.items():
                    covariance[i][j] = zone["covariance"][a][b]
        zone = context["state"]["zones"][key] = {
            "structure": shape, "zone_version": topology["zone_versions"][key], "features": names, "theta": mean,
            "covariance": covariance, "noise": zone["noise"] if kept else INITIAL_NOISE_C**2,
            "updates": zone["updates"] if kept else 0,
            "excited": {unit: counts for unit, counts in zone["excited"].items() if unit in units} if kept else {},
            "pending": [], "stats": {"n": 0, "unscored": 0}, "ready": False, "last": None, "anchor": None}
    return zone


def kalman(zone, x, y, variance_cap, drift):
    """Random-walk prediction, then a Huber-weighted measurement update in Joseph form."""
    theta, p = zone["theta"], zone["covariance"]
    n = len(theta)
    for i in range(n):
        p[i][i] += drift[i]
    for i in range(n):
        if p[i][i] > variance_cap[i]:
            # Bound the growth of coefficients the data has not excited.
            scale = sqrt(variance_cap[i]/p[i][i])
            for j in range(n):
                p[i][j] *= scale
                p[j][i] *= scale
    px = matvec(p, x)
    spread = dot(x, px)
    innovation = y-dot(x, theta)
    limit = HUBER*sqrt(spread+zone["noise"])
    noise = zone["noise"]*max(1.0, abs(innovation)/limit)
    gain = [value/(spread+noise) for value in px]
    zone["theta"] = [t+k*innovation for t, k in zip(theta, gain)]
    a = [[(1.0 if i == j else 0.0)-gain[i]*x[j] for j in range(n)] for i in range(n)]
    columns = list(zip(*p))
    ap = [[dot(row, column) for column in columns] for row in a]
    updated = [[dot(ap[i], a[j])+noise*gain[i]*gain[j] for j in range(n)] for i in range(n)]
    zone["covariance"] = [[(updated[i][j]+updated[j][i])/2 for j in range(n)] for i in range(n)]
    clipped = max(-limit, min(limit, innovation))
    estimate = max(NOISE_FLOOR_C**2, min(NOISE_CEILING_C**2, clipped**2-spread))
    zone["noise"] += NOISE_ADAPTATION*(estimate-zone["noise"])


def regressor(last, environment, hours, powers):
    return ([hours*(value-last["value"]) for value in environment] + [hours*(last["mass"]-last["value"])]
            + [hours*kw for kw in powers] + [hours])


def split_power(mode, kw):
    return [kw if mode == wanted else 0.0 for wanted in MODES]


def learn(context, key, zone, ref, units_now, power):
    """Update the coefficients from the interval between the learning anchor and the new report.

    Reports closer than MIN_INTERVAL_SECONDS to the anchor return "accumulating"
    and leave the anchor in place, so frequent reports still form intervals.
    """
    anchor = zone["anchor"]
    if anchor is None:
        return "first_report"
    elapsed = ref["at"]-anchor["at"]
    if elapsed < MIN_INTERVAL_SECONDS:
        return "accumulating"
    if elapsed > setting(context["policy"], "model_max_interval_seconds"):
        return "interval"
    if any(value is None for value in anchor["environment"]):
        return "environment_missing"
    operation = {}
    for unit in context["units"]:
        condition = units_now.get(unit) or {}
        modes = ({condition.get("mode")} if (condition.get("changed_at") or inf) <= anchor["at"]
                 else {condition.get("mode"), anchor["modes"].get(unit)})
        running = modes & set(MODES)
        if len(running) > 1:
            return "mode_reversal"
        kw = power(unit, anchor["at"], ref["at"])
        if kw is None:
            if running:
                return "power_missing"
            kw = 0.0
        mode = next(iter(running), None)
        if mode is None and kw >= EXCITATION_KW:
            return "mode_unknown"
        operation[unit] = (mode, kw)
    hours = elapsed/3600
    powers = [value for unit in context["units"] for value in split_power(*operation[unit])]
    x = regressor(anchor, anchor["environment"], hours, powers)
    _, variance, drift = prior(context["topology"], key, context["environment"], context["units"], context["policy"])
    kalman(zone, x, ref["value"]-anchor["value"], variance, [hours*value for value in drift])
    zone["updates"] += 1
    drawing = [(unit, mode) for unit, (mode, kw) in operation.items() if mode and kw >= EXCITATION_KW]
    if len(drawing) == 1:
        # Only intervals where a single unit draws real power separate its influence from the others.
        unit, mode = drawing[0]
        counts = zone["excited"].setdefault(unit, {})
        counts[mode] = counts.get(mode, 0)+1
    return "updated"


def forecast(context, zone, environment, units_now):
    """Projected change over the controller's forecast horizon if the current output continues."""
    last = zone["last"]
    if last is None or any(value is None for value in environment):
        return None
    powers = []
    for unit in context["units"]:
        condition = units_now.get(unit) or {}
        mode, kw = condition.get("mode"), condition.get("kw")
        if mode in MODES and kw is None:
            return None
        powers += split_power(mode, kw)
    x = regressor(last, environment, context["policy"]["forecast_seconds"]/3600, powers)
    spread = dot(x, matvec(zone["covariance"], x))
    return {"delta": dot(x, zone["theta"]), "uncertainty": sqrt(max(0.0, spread)+zone["noise"]),
            "from": last["at"], "to": last["at"], "reports": zone["updates"], "source": "model"}


def settle(context, zone, ref, intervened):
    """Score forecasts against the first report 7.5 to 20 minutes later when nothing was changed meanwhile."""
    horizon = context["policy"]["forecast_seconds"]
    stats, keep = zone["stats"], []
    for item in zone["pending"]:
        age = ref["at"]-item["at"]
        if age < SETTLE_SECONDS[0]:
            keep.append(item)
            continue
        if age > SETTLE_SECONDS[1]:
            continue
        if intervened >= item["at"]:
            stats["unscored"] += 1
            continue
        scale = age/horizon
        actual = ref["value"]-item["value"]
        errors = {"model": abs(actual-scale*item["model"]), "persistence": abs(actual),
                  "rule": abs(actual-scale*item["rule"])}
        covered = float(errors["model"] <= item["sigma"]*max(1.0, scale))
        stats["n"] += 1
        alpha = max(1/stats["n"], 1/STATS_WINDOW)
        for name, error in [*errors.items(), ("coverage", covered)]:
            stats[name] = compact(stats.get(name, error)+alpha*(error-stats.get(name, error)))
    zone["pending"] = keep
    policy = context["policy"]
    if stats["n"] < setting(policy, "model_min_samples"):
        zone["ready"] = False
        return
    baseline = min(stats["persistence"], stats["rule"])
    if zone["ready"]:
        zone["ready"] = stats["model"] <= baseline and stats["coverage"] >= 0.4
    else:
        zone["ready"] = stats["model"] <= setting(policy, "model_ready_margin")*baseline and 0.5 <= stats["coverage"] <= 0.95


def observe_zone(context, key, ref, rule, environment, units_now, power, now, intervened=0):
    """Settle, learn from and forecast one zone.

    `ref` is the zone reading used for control and `rule` the rule trend of this
    assessment. `environment` holds environment readings, `units_now` maps each
    unit to its current mode, last mode change and kW, `power(unit, start, end)`
    returns the measured mean kW over an interval or None, and `intervened` is the
    latest time any control action took effect. State changes only when the zone
    has a new report.
    """
    zone = zone_state(context, key)
    values = []
    for name in context["environment"]:
        reading = environment.get(name) or {}
        recent = reading.get("at") is not None and 0 <= now-reading["at"] <= ENVIRONMENT_HOLD_SECONDS
        values.append(reading.get("value") if recent else None)
    last = zone["last"]
    fresh = (ref["quality"] == "fresh" and ref["value"] is not None and ref["at"] is not None
             and isfinite(ref["value"]) and ref["at"] > (last or {}).get("at", -inf))
    if fresh:
        changes = [intervened, *((units_now.get(unit) or {}).get("changed_at") or 0 for unit in context["units"])]
        settle(context, zone, ref, max(changes))
        zone["reason"] = learn(context, key, zone, ref, units_now, power)
        elapsed = ref["at"]-last["at"] if last else inf
        if elapsed <= setting(context["policy"], "model_max_interval_seconds"):
            weight = 1-exp(-elapsed/MASS_SECONDS)
            mass = last["mass"]+weight*((last["value"]+ref["value"])/2-last["mass"])
        else:
            mass = ref["value"]
        zone["last"] = {"at": ref["at"], "value": ref["value"], "environment": values, "mass": compact(mass),
                        "modes": {unit: (units_now.get(unit) or {}).get("mode") for unit in context["units"]}}
        if zone["reason"] != "accumulating":
            zone["anchor"] = zone["last"]
    projected = forecast(context, zone, values, units_now)
    if fresh and projected is not None:
        # Bound pending forecasts by age and space them out, so frequent reports
        # cannot push every forecast out before it can be settled.
        pending = [item for item in zone["pending"] if ref["at"]-item["at"] <= SETTLE_SECONDS[1]]
        if not pending or ref["at"]-pending[-1]["at"] >= PENDING_SPACING_SECONDS:
            pending.append({"at": ref["at"], "value": ref["value"], "model": compact(projected["delta"]),
                            "sigma": compact(projected["uncertainty"]), "rule": compact(rule["delta"]) if rule else 0.0})
        zone["pending"] = pending
    if fresh:
        zone["theta"] = [compact(value) for value in zone["theta"]]
        zone["covariance"] = [[compact(value) for value in row] for row in zone["covariance"]]
        zone["noise"] = compact(zone["noise"])
    return {"forecast": projected, "ready": zone["ready"],
            "use": active(context["policy"]) and zone["ready"] and projected is not None}


def influence(state, topology, policy):
    """Learned influence of each unit and mode on each other zone relative to its home zone.

    A ratio is reported only when the unit's home response has the expected
    sign with at least three standard deviations and enough excitation.
    """
    result = {}
    if not isinstance(state, dict) or state.get("version") != MODEL_VERSION:
        return result
    zones = state["zones"]
    minimum = setting(policy, "influence_min_excitation")
    negligible, significant = setting(policy, "influence_negligible"), setting(policy, "influence_significant")
    for unit, actuator in topology["actuators"].items():
        home = zones.get(actuator["zone"])
        for mode in MODES:
            name = f"{unit}:{mode}"
            if home is None or name not in home["features"]:
                continue
            index = home["features"].index(name)
            reference, spread = home["theta"][index], max(home["covariance"][index][index], 0.0)
            if home["excited"].get(unit, {}).get(mode, 0) < minimum or SIGN[mode]*reference <= 3*sqrt(spread):
                continue
            for key, zone in zones.items():
                if key == actuator["zone"] or name not in zone["features"]:
                    continue
                i = zone["features"].index(name)
                ratio = zone["theta"][i]/reference
                std = sqrt(max(zone["covariance"][i][i], 0.0)+ratio**2*spread)/abs(reference)
                excited = zone["excited"].get(unit, {}).get(mode, 0) >= minimum
                status = ("negligible" if excited and abs(ratio)+2*std < negligible else
                          "significant" if excited and ratio-2*std > significant else "uncertain")
                result[(unit, mode, key)] = {"ratio": ratio, "std": std, "status": status}
    return result


def negligible(learned, unit, mode, zone):
    return (learned.get((unit, mode, zone)) or {}).get("status") == "negligible"


def risk_influences(room, mode, topology, learned):
    """Zones whose comfort a unit's output can affect, corrected by confident learned influence."""
    unit = topology["actuators"][room]
    influences = dict(unit["influences"])
    for (name, learned_mode, zone), entry in learned.items():
        if name != room or learned_mode != mode or zone == unit["zone"]:
            continue
        if entry["status"] == "negligible":
            influences.pop(zone, None)
        elif entry["status"] == "significant":
            influences.setdefault(zone, round(entry["ratio"], 3))
    return influences


def summary(state, zones, learned, policy):
    """Compact trace of model readiness, forecast quality and confident learned influence."""
    if not isinstance(state, dict):
        return None
    rows = {}
    for key, zone in state.get("zones", {}).items():
        stats = zone["stats"]
        rows[key] = {"ready": zone["ready"], "used": ((zones.get(key) or {}).get("trend") or {}).get("source") == "model",
                     "updates": zone["updates"], "settled": stats["n"], "unscored": stats.get("unscored", 0),
                     "reason": zone.get("reason"), "sigma": round(sqrt(zone["noise"]), 3),
                     "mae": [round(stats[name], 3) if name in stats else None for name in ("model", "persistence", "rule")],
                     "coverage": round(stats["coverage"], 2) if "coverage" in stats else None}
    confident = [{"unit": unit, "mode": mode, "zone": zone, "ratio": round(entry["ratio"], 3),
                  "std": round(entry["std"], 3), "status": entry["status"]}
                 for (unit, mode, zone), entry in sorted(learned.items()) if entry["status"] != "uncertain"]
    return {"active": active(policy), "zones": rows, "influence": confident}
