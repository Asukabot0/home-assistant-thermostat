"""Reconstruct the first decision in historical operating episodes with fake HA."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from test_unified_control import UnifiedCycleTests


def at(clock):
    return datetime.fromisoformat("2026-11-27T"+clock+"+09:00").timestamp()


def replay(path):
    with ZipFile(path) as archive:
        rows=[json.loads(line) for line in archive.read("raw/live-audit.jsonl").splitlines()]
        report=[json.loads(line) for line in archive.read("analysis/same-band-1w-to-2w.jsonl").splitlines()]
    assert any(r["room"]=="bedroom" and round(r["from"]["seconds"])==240 and round(r["to"]["seconds"])==300 for r in report)
    for label, begin, end in (("gentle_heat", "04:12:00", "04:41:00"),
                              ("low_power_exit_wait", "06:42:00", "07:20:00"),
                              ("opposed_independent_readings", "08:00:00", "08:37:00")):
        source=[r for r in rows if at(begin)<=r["at"]<at(end) and r.get("event")=="cycle" and len(r.get("rooms",{}))==3]
        if not source:
            raise ValueError("Replay window missing real cycle records")
        harness=UnifiedCycleTests()
        harness.setUp()
        try:
            harness.h.now=9000
            first=source[0]
            room_data=lambda key:first["rooms"][key]
            harness.scene(*((room_data(key)["mode_reported"],room_data(key)["setpoint_reported"],room_data(key)["room_c"],
                              room_data(key)["power_w"]) for key in ("bedroom","living","study")))
            h=harness.h
            h.update_state(manual_holds={"living":"historical pause"})
            origin=9000
            translate=lambda t:origin+t-first["at"]
            reasons=Counter()
            selected=None
            for r in source:
                h.now=int(translate(r["at"]))
                for key,data in r["rooms"].items():
                    row=h.live[key]
                    row["temperature"]=data["room_c"]
                    if data["temperature_report_at"] is not None:
                        row["temperature_report_at"]=translate(data["temperature_report_at"])
                    row["adapter"]["power"]=data["power_w"]
                    h.observe(key,new_report=False)
                state=h.snapshot()
                trace=harness.trace()
                reasons.update(v["reason"] for v in trace["zones"].values())
                if h.calls:
                    selected={k:state["commands"][-1][k] for k in ("room","kind","value","purpose")}
                    break
            print(json.dumps({"episode":label,"cycles_available":len(source),"first_simulated_action":selected,
                              "observed_reasons":dict(reasons),"classification":"offline; historical measurements held exogenous"}))
        finally:
            harness.doCleanups()


if __name__=="__main__":
    replay(Path(sys.argv[1]))
