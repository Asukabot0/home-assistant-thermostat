"""Regression tests for persisted steady episodes and graduated reversal setpoints."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "home-assistant/shadow"))
from service import continuous_entity, init, low_power_span, power_span, stable_setpoints
from steady_learning import EPISODE_SECONDS, LEARNING_WINDOW, initial_setpoint, learned_offset, learning_active, learning_session


TICK = 2_000_040
SESSION = 1
COOL_BOUNDS = [22, 26]
HEAT_BOUNDS = [20, 25]


def qualifying_episode(db, state, room, end, *, mode="cool", setpoint=28, edit=None, duration=1200):
    ticks = range(end-EPISODE_SECONDS, end+1, 60)
    rows = [{"tick": t, "adapter_mode": mode, "reported": setpoint, "measured": 24, "temp_report": t} for t in ticks]
    if edit:
        rows[-1].update(edit)
    db.executemany("INSERT OR IGNORE INTO decisions VALUES (?,?,?,?,?)",
                   [(room, t, "fixture", 0, json.dumps({"learning_session": SESSION})) for t in ticks if t < end])
    return stable_setpoints(db, room, end, rows, state, SESSION, duration)[mode]


def run(path):
    session = {"start": SESSION, "holds": {}, "commands": []}
    with closing(sqlite3.connect(path)) as db, db:
        init(db, TICK)
        assert initial_setpoint(db, "bedroom", "cool", TICK, 24, COOL_BOUNDS) == 24
        assert qualifying_episode(db, session, "bedroom", TICK)["episodes"] == 1
        assert initial_setpoint(db, "bedroom", "cool", TICK, 24, COOL_BOUNDS) == 25
        assert qualifying_episode(db, session, "bedroom", TICK+60)["episodes"] == 1
        assert qualifying_episode(db, session, "bedroom", TICK+1200)["episodes"] == 2
        assert initial_setpoint(db, "bedroom", "cool", TICK+1200, 24, COOL_BOUNDS) == 25
        assert qualifying_episode(db, session, "bedroom", TICK+2400)["episodes"] == 3
        assert initial_setpoint(db, "bedroom", "cool", TICK+2400, 24, COOL_BOUNDS) == 26
        assert initial_setpoint(db, "bedroom", "heat", TICK+2400, 24, HEAT_BOUNDS) == 24
        assert qualifying_episode(db, session, "study", TICK, setpoint=21)["episodes"] == 1
        assert initial_setpoint(db, "study", "cool", TICK, 24, COOL_BOUNDS) == 24

        cmd = {"room": "living", "kind": "temperature", "value": 25, "before": ["cool", 24, "auto"],
               "before_watts": 68, "sent_at": TICK-1200, "status": "unconfirmed"}
        session["commands"].append(cmd)
        assert qualifying_episode(db, session, "living", TICK) is None
        cmd.update(status="proxy_echo", followup={
            "5": {"at": TICK-900, "power_age_s": 1, "watts": 68, "reported": ["cool", 25]},
            "15": {"at": TICK-300, "power_age_s": 1, "watts": 70, "reported": ["cool", 25]},
        })
        assert qualifying_episode(db, session, "living", TICK) is None
        cmd["followup"]["15"]["watts"] = 50
        assert qualifying_episode(db, session, "living", TICK, edit={"measured": 25.2}) is None
        assert qualifying_episode(db, session, "living", TICK, edit={"temp_report": TICK-1201}) is None
        assert qualifying_episode(db, session, "living", TICK, duration=1199) is None
        assert qualifying_episode(db, session, "living", TICK)["episodes"] == 1

        end = TICK + LEARNING_WINDOW + 2401
        assert learned_offset(db, "bedroom", "cool", end) is None
        for i in range(22):
            db.execute("INSERT INTO stable_episodes VALUES (?,?,?,?,?)",
                       ("bedroom", "cool", end+i*1200-1200, end+i*1200, 3 if i >= 2 else -4))
        assert learned_offset(db, "bedroom", "cool", end+21*1200)["episodes"] == 20
        assert initial_setpoint(db, "bedroom", "cool", end+21*1200, 24, COOL_BOUNDS) == 26
    with closing(sqlite3.connect(path)) as db:
        assert initial_setpoint(db, "bedroom", "cool", TICK+2400, 24, COOL_BOUNDS) == 26


with TemporaryDirectory() as folder:
    folder = Path(folder)
    run(folder / "shadow.sqlite3")
    (folder / "live-permit.json").write_text(json.dumps({"start": SESSION, "end": None, "enabled": True}))
    (folder / "continuous-1.json").write_text(json.dumps({"start": SESSION, "holds": {}, "commands": []}))
    active = learning_session(folder, TICK)
    auto = {"state": "auto", "shared_attrs": json.dumps({"preset_mode": "标准"})}
    assert learning_active(active, "study", auto) == SESSION
    assert learning_active(active, "study", dict(auto, state="off")) is None
    active["holds"]["living_study"] = "manual"
    assert learning_active(active, "study", auto) is None

model = {"centers_w": [18.4, 68, 310], "edges_w": [40, 190]}
threshold, elapsed_low = low_power_span(TICK, [(TICK-1800, 68)], model)
assert threshold == 23 and elapsed_low == 0
assert power_span(TICK, [(TICK-1800, 68)], lambda power: power > threshold) == 1800
assert power_span(TICK, [(TICK-1800, 68), (TICK-400, None), (TICK-200, 68)],
                  lambda power: power > threshold) == 200
with closing(sqlite3.connect(":memory:")) as recorder:
    recorder.row_factory = sqlite3.Row
    recorder.executescript("""CREATE TABLE states_meta(metadata_id,entity_id);
    CREATE TABLE states(state_id,state,last_reported_ts,last_updated_ts,metadata_id,attributes_id);
    CREATE TABLE state_attributes(attributes_id, shared_attrs);
    INSERT INTO states_meta VALUES (1,'climate.fixture');
    INSERT INTO states VALUES (1,'cool',100,100,1,NULL),(2,'unavailable',139,139,1,NULL),
                              (3,'cool',141,141,1,NULL);""")
    assert not continuous_entity(recorder, "climate.fixture", 120, 160, lambda row: row["state"] == "cool")
    assert continuous_entity(recorder, "climate.fixture", 142, 160, lambda row: row["state"] == "cool")
print("Verified steady episodes, confidence filtering, window expiry, cold-start tiers, and study isolation.")
