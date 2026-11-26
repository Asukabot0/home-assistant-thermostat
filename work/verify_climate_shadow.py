"""Regression checks for evidence labels, low-frequency advice, and the trial log."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "home-assistant/shadow"))
from cadence import BEEP_INTERVAL, TRIALS_SCHEMA, latest_beeps, record_beep, save_receipt
from model import Activity, advise, calibrate, identify
from pilot_gate import eligible
from record_trial import collect
from service import coordinate, low_power_span, metrics, power_state_window, powers, recent_power_reports


def row(tick, watts, mode="cool", outside=31.):
    return {"tick": tick, "watts": watts, "adapter_mode": mode, "reported": 25., "outdoor": outside,
            "measured": 26., "desired": 24., "temp_report": tick}


clusters = [12,150,480]
points = [(120*i, clusters[i%3] + i%4) for i in range(72)]
room_a = calibrate(points)
room_b = calibrate([(t, v*1.7) for t,v in points])
assert len(room_a["centers_w"]) == 3 and len(room_b["centers_w"]) == 3
assert room_a["edges_w"] != room_b["edges_w"]
assert calibrate([(120*i, 90) for i in range(72)]) is None
assert calibrate(points[:5]) is None

tick = 12000
assert identify(tick, row(tick,12,"off"), [(tick,12)], room_a, []).state == "待机迹象"
assert identify(tick, row(tick,12,"fan_only"), [(tick,12)], room_a, []).state == "送风迹象"
assert identify(tick, row(tick,150), [(tick,150)], room_a, []).state == "低功率变频迹象"
assert identify(tick, row(tick,480), [(tick,480)], room_a, []).state.startswith("高功率运行")
assert identify(tick, row(tick,480), [(tick-100,12),(tick-70,12),(tick,480)], room_a, []).state == "压缩机启动迹象"
assert identify(tick, row(tick,480), [(tick-650,480)], room_a, []).confidence == "未知"
heat = [row(tick-(5-i)*60,w,"heat",3) for i,w in enumerate([480,480,480,12,12,480])]
assert identify(tick, heat[-1], [(tick-100,12),(tick-70,12),(tick,480)], room_a, heat).state == "疑似除霜后恢复"

profile = {"target":24,"quiet":False,"offsets":{"heat":1,"cool":1}}
history = [row(tick-i*60,480) for i in range(20,-1,-1)]
active = Activity("高功率运行", "推测·中", "fixture")
result = advise(tick, history[-1], profile, history, active, None, {})
assert result["hypothetical_send"] and result["setpoint"] == 23 and result["send_after"] == tick
unchanged = [dict(r, temp_report=tick-1200) for r in history]
assert advise(tick, unchanged[-1], profile, unchanged, active, None, {})["hypothetical_send"]
temp_expired = [dict(r, temp_report=tick-1260) for r in history]
assert not advise(tick, temp_expired[-1], profile, temp_expired, active, None, {})["hypothetical_send"]
assert not advise(tick, history[-1], dict(profile,quiet=True), history, active, None, {})["hypothetical_send"]
assert advise(tick, history[-1], profile, history, Activity("未知","未知","fixture"), None, {})["hypothetical_send"]
assert not advise(tick, history[-1], profile, history[:-4]+history[-1:], active, None, {})["hypothetical_send"]
within_minute = dict(history[-1], temp_report=tick+42)
assert advise(tick, within_minute, profile, history[:-1]+[within_minute], active, None, {})["hypothetical_send"]
assert not advise(tick, history[-1], profile, history, active,
                  {"mode":"cool","setpoint":23,"from_setpoint":25,"tick":tick-2000}, {})["hypothetical_send"]
assert advise(tick, history[-1], profile, history, active,
              {"mode":"cool","setpoint":24,"from_setpoint":25,"tick":tick-60}, {})["send_after"] == tick+1740
warmer = [row(tick-i*60,480, "heat") for i in range(60,-1,-1)]
switch = advise(tick, warmer[-1], profile, warmer, active, None, {})
assert switch["hypothetical_send"] and switch["mode"] == "cool" and switch["command_kind"] == "mode"
assert advise(tick, warmer[-1], profile, warmer[-31:], active, None, {})["send_after"] == tick+1800
assert not advise(tick, warmer[-1], profile, warmer, active,
                  dict(switch, reported_mode="heat", tick=tick-120), {})["hypothetical_send"]
cool_night = [dict(row(tick-i*60,480,"cool",15.8), measured=23.0) for i in range(60,-1,-1)]
with_weather = advise(tick, cool_night[-1], profile, cool_night, active, None, {})
assert with_weather["weather_assisted_reversal"] and with_weather["reversal_half_band"] == 0.8
assert with_weather["hypothetical_send"] and with_weather["mode"] == "heat"
for outside in (None, 21.0, 31.0):
    without_support = [dict(r, outdoor=outside) for r in cool_night]
    hold = advise(tick, without_support[-1], profile, without_support, active, None, {})
    assert not hold["hypothetical_send"] and hold["reversal_half_band"] == 1.1
warm_day = [dict(r, measured=25.0, outdoor=31.0, adapter_mode="heat") for r in cool_night]
assert advise(tick, warm_day[-1], profile, warm_day, active, None, {})["weather_assisted_reversal"]
edge = [dict(r, measured=23.2) for r in cool_night]
assert not advise(tick, edge[-1], profile, edge, active, None, {})["hypothetical_send"]
assert not advise(tick, cool_night[-1], profile, cool_night[-30:], active, None, {})["hypothetical_send"]
assert not advise(tick, cool_night[-1], profile, cool_night, active, None, {},
                  last_beep_at=tick-60)["hypothetical_send"]
assert not advise(tick, cool_night[-1], dict(profile,quiet=True), cool_night, active, None, {})["hypothetical_send"]
assert not advise(tick, cool_night[-1], profile, cool_night, Activity("未知","未知","fixture"), None, {})["hypothetical_send"]
fixed = advise(tick, warmer[-1], profile, warmer, active, None, {}, fixed_mode="heat")
assert fixed["hypothetical_send"] and fixed["setpoint"] == 23 and fixed["command_kind"] == "temperature"
colder = [dict(r, measured=22.5) for r in warmer]
assert advise(tick, colder[-1], profile, colder, active, None, {}, fixed_mode="heat")["setpoint"] == 27
assert not advise(tick, history[-1], profile, history, active, None, {}, fixed_mode="heat")["hypothetical_send"]
assert not advise(tick, warmer[-1], dict(profile, quiet=True), warmer, active, None, {}, fixed_mode="heat")["hypothetical_send"]
manual_change = warmer[:-1]+[dict(warmer[-1], reported=24)]
assert not advise(tick, manual_change[-1], profile, manual_change, active, None, {}, fixed_mode="heat")["hypothetical_send"]
stale = dict(warmer[-1], temp_report=tick-1201)
assert not advise(tick, stale, profile, warmer, active, None, {}, fixed_mode="heat")["hypothetical_send"]
assert not advise(tick, warmer[-1], profile, warmer, Activity("未知","未知","fixture"), None, {}, fixed_mode="heat")["hypothetical_send"]
assert metrics([])["projected_beeps_24h"] is None

last_beep = tick - 25 * 60
paused = advise(tick, history[-1], profile, history, active, None, {}, last_beep_at=last_beep)
assert not paused["hypothetical_send"] and paused["send_after"] == last_beep + BEEP_INTERVAL
assert advise(tick, history[-1], profile, history, active, None, {},
              last_beep_at=tick-BEEP_INTERVAL)["hypothetical_send"]
rising = [dict(r, measured=25.0 if r["tick"] < tick-240 else 26.2) for r in history]
rapid = advise(tick, rising[-1], profile, rising, active, None, {}, last_beep_at=last_beep)
assert rapid["hypothetical_send"] and rapid["rapid_override"] and rapid["setpoint"] == 23
assert not advise(tick, rising[-1], profile, rising, active, None, {}, last_beep_at=tick-300)["hypothetical_send"]
assert not advise(tick, rising[-1], profile, rising[2:], active, None, {}, last_beep_at=last_beep)["hypothetical_send"]
assert not advise(tick, rising[-1], dict(profile, quiet=True), rising, active, None, {},
                  last_beep_at=last_beep)["hypothetical_send"]
falling = [dict(r, measured=23.0 if r["tick"] < tick-240 else 21.8, adapter_mode="heat") for r in history]
assert advise(tick, falling[-1], profile, falling, active, None, {}, last_beep_at=last_beep,
              fixed_mode="heat")["rapid_override"]
reversing = warmer[:-len(rising)]+[dict(r, adapter_mode="heat", measured=r["measured"]+0.2) for r in rising]
switch = advise(tick, reversing[-1], profile, reversing, active, None, {}, last_beep_at=last_beep)
assert switch["command_kind"] == "mode" and not switch["hypothetical_send"] and not switch["rapid_override"]
old_report = [dict(r, temp_report=min(r["tick"], tick-360)) for r in rising]
assert not advise(tick, old_report[-1], profile, old_report, active, None, {},
                  last_beep_at=last_beep)["rapid_override"]
recovering = rising[:-1]+[dict(rising[-1], measured=26.1)]
assert not advise(tick, recovering[-1], profile, recovering, active, None, {},
                  last_beep_at=last_beep)["rapid_override"]

shared = {"living": dict(result, source_fresh=True), "study": dict(result, source_fresh=True)}
coordinate(shared)
assert sum(r["hypothetical_send"] for r in shared.values()) == 1
shared = {"living": dict(paused, source_fresh=True), "study": dict(rapid, source_fresh=True)}
coordinate(shared)
assert shared["study"]["hypothetical_send"] and shared["study"]["rapid_override"]
assert shared["living"]["send_after"] is None and not shared["living"]["hypothetical_send"]

live = {"signature": ("stable",), "temperature": rising[-1]["measured"], "temperature_age": 10,
        "adapter": {"mode": "cool", "setpoint": 25, "power": 480, "from_power_s": 10, "last_change": tick}}
gate = {"witness": "matched", "followup_complete": True, "sample": rising[-1], "decision": rapid, "edges": [230]}
assert eligible(tick+1, ("stable",), live, gate, mode="cool", setpoints=(23,))[1] == 23
assert eligible(tick+1, ("changed",), live, gate, mode="cool", setpoints=(24,))[1] is None
assert eligible(tick+1, ("stable",), dict(live, temperature_age=301), gate, mode="cool", setpoints=(24,))[1] is None
assert eligible(tick+1, ("stable",), dict(live, temperature_age=1201), gate, mode="cool", setpoints=(24,))[1] is None

receipts = sqlite3.connect(":memory:")
receipts.execute(TRIALS_SCHEMA)
accepted = {"room": "study", "sent_at": last_beep, "relay": "accepted", "relay_delay_s": 2}
assert record_beep(accepted).at == last_beep + 2
assert record_beep(dict(accepted, relay="rejected", physical="unknown")) is None
assert record_beep(dict(accepted, relay="timeout")).at == last_beep
with TemporaryDirectory() as folder:
    path = Path(folder)
    save_receipt(path, "study", last_beep, "accepted", last_beep+2)
    save_receipt(path, "bedroom", tick-120, "accepted", tick-119)
    saved = latest_beeps(receipts, tick, path)
    assert saved["study"] == saved["living"] and saved["living"].at == last_beep+2
    assert saved["bedroom"].at == tick-119
    heard = dict(accepted, beep_at=last_beep+1, beeps_observed=1)
    receipts.execute("INSERT INTO trials VALUES (?,?,?,?)", ("study-trial","study",last_beep,json.dumps(heard)))
    assert latest_beeps(receipts, tick, path)["living"] == (last_beep+1, "现场听到")
    save_receipt(path, "study", tick-60, "rejected", tick-59)
    assert latest_beeps(receipts, tick, path)["living"].at == last_beep+1
    receipts.execute("UPDATE trials SET value=?", (json.dumps(dict(accepted, beeps_observed=0)),))
    assert set(latest_beeps(receipts, tick, path)) == {"bedroom"}

recorder = sqlite3.connect(":memory:")
recorder.row_factory = sqlite3.Row
recorder.executescript("""CREATE TABLE states_meta(metadata_id,entity_id);
CREATE TABLE states(state_id,state,last_reported_ts,last_updated_ts,metadata_id,attributes_id);
CREATE TABLE state_attributes(attributes_id, shared_attrs);
INSERT INTO states_meta VALUES (1,'sensor.synthetic_power'),(2,'climate.synthetic_ac');
INSERT INTO state_attributes VALUES (1,'{"temperature":24}');
INSERT INTO states VALUES (1,'100',11940,11940,1,NULL),(2,'500',12005,12005,1,NULL),
(3,'cool',12002,12002,2,1);""")
assert powers(recorder, "sensor.synthetic_power", 0, end=12000) == [(11940, 100)]
recorder.execute("INSERT INTO states_meta VALUES (3,'climate.synthetic_observer')")
for key, observed in ((2, 12500), (3, 12820)):
    attrs = json.dumps({"power_reported_at": datetime.fromtimestamp(observed, timezone.utc).isoformat()})
    recorder.execute("INSERT INTO state_attributes VALUES (?,?)", (key, attrs))
    recorder.execute("INSERT INTO states VALUES (?,?,?,?,?,?)", (key+10,"cool",observed+10,observed+10,3,key))
reports = recent_power_reports(recorder,"sensor.synthetic_power","climate.synthetic_observer",12400,12900)
assert reports == [(12500,500), (12820,500)]
assert recent_power_reports(recorder,"sensor.synthetic_power","climate.synthetic_observer",12850,12900) == []
model = {"edges_w":[800],"centers_w":[450]}
assert low_power_span(12840,[(12500,500)],model) == (562.5,340)
assert low_power_span(12840,[(12500,500),(12700,None),(12820,500)],model) == (562.5,20)
assert low_power_span(12840,[(12500,500),(12800,900)],model) == (562.5,0)
states, current = power_state_window(recorder,"sensor.synthetic_power",12400,12900)
assert states == [(12400,500)] and current["updated"] == 12005
recorder.executescript("INSERT INTO states VALUES (20,'unavailable',12600,12600,1,NULL),"
                       "(21,'500',12720,12720,1,NULL);")
states, current = power_state_window(recorder,"sensor.synthetic_power",12400,12900)
assert low_power_span(12840,states,model) == (562.5,120)
observer = sqlite3.connect(":memory:")
observer.row_factory = sqlite3.Row
observer.executescript("""CREATE TABLE samples(room,temp_report,measured);
INSERT INTO samples VALUES ('bedroom',11940,25.8),('bedroom',12120,25.6);""")
args = SimpleNamespace(room="bedroom", action="temperature", request="24 C", sent_at=tick, followup_minutes=15,
                       physical="unknown", physical_note=None, relay="accepted", relay_at=tick+1, observed_at=None, beeps=None)
trial = collect(args, recorder, observer, {"power":"sensor.synthetic_power", "adapter":"climate.synthetic_ac"})
assert trial["relay_delay_s"] == 1 and trial["power_after"]["median"] == 500
assert trial["physical"] == "unknown" and trial["observed_delay_s"] is None and len(trial["temperature_timeline"]) == 2
assert trial["proxy_timeline"][-1]["setpoint"] == 24
print("Verified power evidence, beep timing and shared anchors, rapid exceptions, and unchanged live gates.")
