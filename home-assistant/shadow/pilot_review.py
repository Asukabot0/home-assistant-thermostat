"""Summarize the live trial from its journal and the offline room measurements."""
from datetime import datetime
import json
from math import ceil, floor


def followup(action, readings, minute):
    sent = action["sent_at"]
    due = sent + minute*60
    saved = action.get("followup", {}).get(str(minute))
    if saved and due <= saved["at"] <= due+180:
        temperature = saved["room_c"] if 0 <= saved["room_age_s"] <= min(1200, saved["at"]-sent) else None
        power = saved["watts"] if 0 <= saved["power_age_s"] <= min(600, saved["at"]-sent) else None
        if temperature is not None and power is not None:
            return temperature, power
    else:
        temperature, power = None, None
    for tick, row in readings:
        if due <= tick <= due+180:
            temperature_at = row["temperature_report_at"]
            power_at = row.get("power_report_at")
            if temperature is None and row["source_fresh"] and temperature_at is not None and temperature_at >= sent:
                temperature = row["measured"]
            if power is None and power_at is not None and power_at >= sent and -59 <= tick-power_at <= 600:
                power = row.get("power_report_w", row["watts"])
            if temperature is not None and power is not None:
                break
    return temperature, power


def physical_result(db, action):
    if action.get("physical_verified") is True:
        return "matched"
    record = db.execute("""SELECT value FROM trials WHERE room=? AND sent_at BETWEEN ? AND ?
                           ORDER BY sent_at DESC LIMIT 1""",
                        (action["room"], action["sent_at"]-0.001, action["sent_at"]+0.001)).fetchone()
    return json.loads(record[0]).get("physical", "unknown") if record else "unknown"


def value_at(temperature, power):
    temp = f"{temperature:.1f}℃" if temperature is not None else "温度待更新"
    watts = f"{power:g} W" if power is not None else "功率待更新"
    return f"{temp} / {watts}"


def trial_review(db, config, journal, now, tick, zone):
    try:
        permit = json.loads((journal / "continuous-permit.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    start, end = permit["start"], permit["end"]
    try:
        state = json.loads((journal / f"continuous-{int(start)}.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        state = {"commands": [], "holds": {}}
    actions = [action for action in state["commands"] if action["status"] != "canceled"]
    last_followup = max((c["sent_at"] + 900 for c in actions if c["status"] != "rejected"), default=0)
    active = permit["enabled"] and now < end
    finishing = not active and tick < last_followup and now < last_followup + 180
    phase = "实机试点进行中" if active else "补采发令后随访" if finishing else "本轮汇总"
    through = min(tick, end)
    expected = max(0, floor(through/60) - ceil(start/60) + 1)
    readings = {room: [(stamp, json.loads(value)) for stamp, value in db.execute(
        "SELECT tick,value FROM decisions WHERE room=? AND tick BETWEEN ? AND ? ORDER BY tick",
        (room, start, min(tick, max(end, last_followup+180))))] for room in config}

    def local(stamp):
        return datetime.fromtimestamp(stamp, zone).strftime("%m-%d %H:%M")

    working = sum(action["status"] in {"accepted", "proxy_echo"} for action in actions)
    echoed = sum(action["status"] == "proxy_echo" for action in actions)
    physical = [physical_result(db, action) for action in actions]
    faults = sum(action["status"] in {"rejected", "unconfirmed", "prepared"} or evidence == "not_matched"
                 or (action["status"] == "accepted" and now-action["sent_at"] >= 180)
                 for action, evidence in zip(actions, physical))
    lines = ["# 三房持续恒温实机试点复盘", "",
             f"{phase}。发令窗口 {local(start)} – {local(end)}（{zone.key}）。温度观测截至 {local(through)}。",
             f"发令记录 {len(actions)} 条，米家调用已受理 {working} 条，伴侣已回报 {echoed} 条，"
             f"室内机现场确认 {physical.count('matched')} 条，异常或结果不明 {faults} 条。", "",
             "| 房间 | 有效温度分钟 | 温度范围 | 目标 ±0.5℃ 覆盖 |", "|---|---:|---:|---:|"]
    for room, rows in readings.items():
        valid = [r for stamp, r in rows if stamp <= through and r["source_fresh"] and r["measured"] is not None]
        band = sum(abs(r["measured"] - r["target"]) <= 0.5 for r in valid)
        span = f"{min(r['measured'] for r in valid):.1f}–{max(r['measured'] for r in valid):.1f}℃" if valid else "—"
        rate = f"{100*band/len(valid):.0f}%" if valid else "—"
        lines.append(f"| {config[room]['name']} | {len(valid)}/{expected} | {span} | {rate} |")
    if "living" in config and "study" in config and config["living"].get("temperature") == config["study"].get("temperature"):
        lines.append("\n书房使用客厅参考温度。")
    for group, reason in state["holds"].items():
        name = "客厅与书房" if group == "living_study" else config.get(group, {}).get("name", group)
        lines.append(f"- {name}暂停原因为{reason}。")

    if actions:
        lines.extend(["", "## 指令及随访", "",
                      "| 发送 | 房间与指令 | 回报 · 室内机 | 发送前 | 5 分钟 | 15 分钟 |",
                      "|---|---|---|---|---|---|"])
        statuses = {"proxy_echo": "伴侣已回报", "accepted": "已受理", "rejected": "调用拒绝",
                    "unconfirmed": "结果不明", "prepared": "发令结果不明"}
        physical_labels = {"matched": "现场确认", "not_matched": "现场不符", "unknown": "现场待核对"}
        for action, evidence in zip(actions, physical):
            room, at = action["room"], action["sent_at"]
            transition = (f"{action['before'][1]:g}→{action['value']:g}℃" if action["kind"] == "temperature" else
                          f"{'制冷' if action['before'][0] == 'cool' else '制热'}→{'制冷' if action['value'] == 'cool' else '制热'}")
            before = value_at(action["before_room_c"], action["before_watts"])
            after = [value_at(*followup(action, readings[room], minute)) for minute in (5, 15)]
            relay = (f"（{action['relay_at']-at:.2f} 秒）"
                     if action["status"] in {"accepted", "proxy_echo"} and action.get("relay_at") is not None else "")
            lines.append(f"| {datetime.fromtimestamp(at,zone):%H:%M:%S} | {config[room]['name']} {transition} | "
                         f"{statuses[action['status']]}{relay} · {physical_labels[evidence]} | {before} | {after[0]} | {after[1]} |")

    if faults:
        next_step = "先复核结果不明或不符的指令及对应伴侣数据，再决定该组能否恢复。"
    elif not actions:
        next_step = "本轮发令数为零。下一次有持续温差且数据新鲜时，继续验证第一条同模式设温。"
    elif any(result != "matched" for result in physical):
        next_step = "先核对表中待确认的室内机显示和蜂鸣，再根据 5 分钟及 15 分钟随访评估下一轮。"
    else:
        next_step = "核对温度变化和发令频率。结果稳定后可延长同模式试点，发生冷热切换时单独验收。"
    lines.extend(["", "## 下一步", "", next_step, ""])
    return "\n".join(lines)
