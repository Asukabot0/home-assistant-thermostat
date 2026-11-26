"""Reconstruct selected decisions from a local, sanitized acceptance ZIP.

Original observations are exogenous. This does not model the effect of actions.
Historical live assessment reasons are unknown. Only the first action per
counterfactual segment is compared.
"""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from test_p0_p1_dispatch import DispatchTests, NOW, pilot


class Replay:
    def __init__(self, archive):
        with ZipFile(archive) as data:
            def load(name):
                path = next(path for path in data.namelist() if path.endswith(f"/new_latest_24h/{name}.jsonl"))
                return [json.loads(line) for line in data.read(path).splitlines()]
            self.observations = load("observations")
            self.decisions = load("shadow_decisions")
            self.commands = load("commands")
        self.day = self.observations[0]["timestamp"][:10]

    def at(self, clock):
        return datetime.fromisoformat(f"{self.day}T{clock}+09:00").timestamp()

    def run(self, clock):
        at = self.at(clock)
        anchor = at-NOW
        translate = lambda t: t-anchor if t is not None else None
        prior = [deepcopy(a) for a in self.commands if a.get("proxy_at", at) < at and a["status"]=="proxy_echo"]
        since = at-3900
        obs = {r["room"]: r for r in self.observations if at-120 < r["tick"] and r["tick"]+59 <= at}
        fixture = DispatchTests()
        fixture.setUp()
        try:
            fixture.whole_scene(("heat",24,24,20), ("cool",24,24,20), ("heat",24,24,20))
            decisions = defaultdict(list)
            for row in self.decisions:
                if since <= row["tick"] and row["tick"]+59 <= at:
                    decisions[row["room"]].append(row)
            for room,v in fixture.live.items():
                latest = obs[room]
                mode, sp, temp, watts = (latest[k] for k in ("adapter_mode","reported","measured","watts"))
                last = next((a for a in reversed(prior) if a["room"]==room), None)
                fan = last["expected"][2] if last and last["kind"]=="temperature" else "auto"
                v["signature"] = (mode,sp,fan,"auto",latest["desired"],"标准",False)
                # The most recent real temperature report can precede the next
                # minute snapshot. Only import its temperature and timestamp.
                live_temp = max((r for r in self.observations if r["room"]==room
                                 and r["measured"] is not None and r["temp_report"] is not None
                                 and r["tick"] <= at and r["temp_report"] <= at),
                                key=lambda r:r["temp_report"], default=latest)
                v["temperature"] = live_temp["measured"]
                v["temperature_report_at"] = translate(live_temp["temp_report"])
                v["temperature_age"] = at-live_temp["temp_report"]
                recent = decisions[room][-1]["value"]
                mode_since = next((a["proxy_at"] for a in reversed(prior) if a["room"]==room and a["kind"]=="mode"),
                                  next((a["before_mode_at"] for a in prior if a["room"]==room), at-7200))
                v["adapter"].update(mode=mode,setpoint=sp,power=watts,from_power_s=at-recent["power_report_at"],
                                    last_mode_change=translate(mode_since), power_state_updated_at=translate(recent["power_state_updated_at"]))
                v["temperature_source"] = ["fixture_living" if room=="study" else f"fixture_{room}","boot"]
                with fixture.db:
                    fixture.db.execute("DELETE FROM decisions WHERE room=?", (room,))
                    for item in decisions[room]:
                        d = deepcopy(item["value"])
                        for k in ("observed_at","temperature_report_at","power_report_at","power_state_updated_at"):
                            if k in d:
                                d[k] = translate(d[k])
                        d["temperature_source"] = v["temperature_source"]
                        proof = d.get("power_evidence")
                        if proof:
                            proof["context"] = v["power_context"]
                            proof["last_report"] = translate(proof["last_report"])
                            for band in proof["bands"].values():
                                band["since"] = translate(band["since"])
                        fixture.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?)",
                                           (room,translate(item["tick"]),"fixture",0,json.dumps(d)))
            parked = {}
            for a in prior:
                for k in ("sent_at","proxy_at","relay_at","before_mode_at"):
                    if k in a:
                        a[k] = translate(a[k])
                a["followup"] = {k:dict(value, at=translate(value["at"])) for k,value in a["followup"].items()
                                  if value["at"] <= at}
                if a["kind"]=="mode":
                    if a["value"]=="off":
                        parked[a["room"]] = {"off_at":a["proxy_at"],"started_at":a["before_mode_at"],"mode":a["before"][0]}
                    else:
                        parked.pop(a["room"],None)
            transition = {}
            if clock=="14:54:10":
                a=next(a for a in reversed(prior) if a["purpose"]=="energy_start")
                transition["living_study"]={"id":a["sent_at"],"mode":"cool","ready_at":a["proxy_at"]+5,
                                            "members":{"living":{"phase":"initial","target":24}}}
            fixture.update_state(commands=prior, parked=parked, transitions=transition)
            state = fixture.snapshot()
            chosen = state["commands"][-1] if fixture.calls else None
            details = pilot.publish.call_args.args[-1]
            print(clock, (chosen["room"],chosen["purpose"],chosen["value"]) if chosen else "idle",
                  {r:details[r]["status"] for r in fixture.live})
            return (chosen["room"],chosen["purpose"],chosen["value"]) if chosen else None
        finally:
            fixture.doCleanups()


if __name__ == "__main__":
    replay = Replay(Path(sys.argv[1]))
    expected = {
        "14:34:10": ("study", "energy_off", "off"),
        "14:46:10": None,
        "14:48:10": ("living", "energy_start", "cool"),
        "14:54:10": None,
        "15:07:19": ("living", "trend_easing", 25),
        "15:18:10": ("living", "energy_off", "off"),
        "15:27:10": None,
    }
    for instant, action in expected.items():
        assert replay.run(instant) == action, instant
