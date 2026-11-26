"""Run the room controller against an isolated Home Assistant service registry."""
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from homeassistant.core import HomeAssistant

sys.path.insert(0, "/probe")
from custom_components.room_climate_control.climate import Room, restore_command
from custom_components.room_climate_control.policy import Request
from custom_components.room_climate_control.telemetry import Telemetry


ADAPTER = "climate.synthetic_xiaomi"
ROOM = "sensor.synthetic_temperature"
POWER = "sensor.synthetic_power"
START = datetime(2026, 11, 25, 8, tzinfo=timezone.utc)


class SimulationRoom(Room):
    def async_write_ha_state(self):
        pass


class RegulatorFixture:
    def set_target_temp(self, target):
        pass

    def calculate_regulated_temperature(self, current, external, step):
        return 22


async def apply(relay, model, time):
    await model.observe(time)
    assert model.result.status == "ready" and model.result.command.value == 24
    await model.send(time)
    assert relay[-1] == {"entity_id": ADAPTER, "temperature": 24}


async def run():
    with TemporaryDirectory() as config_dir:
        ha = HomeAssistant(config_dir)
        entries = {}
        relay = []
        telemetry = Telemetry(Path(config_dir) / "room.sqlite3")
        telemetry.initialize()
        config = {"name": "Synthetic", "adapter": ADAPTER, "temperature": ROOM, "power": POWER,
                  "actuation_approved": True, "absolute_mode_verified": False, "ir_setpoint_validated": True,
                  "heating_offset": 0, "cooling_offset": 0}
        facade = SimpleNamespace(states=SimpleNamespace(get=entries.get),
                                 services=ha.services, async_add_executor_job=ha.async_add_executor_job)
        current_room = None

        async def set_temperature(call):
            assert telemetry.last_command(current_room.key)["status"] == "prepared"
            relay.append(dict(call.data))

        async def unsupported(call):
            raise AssertionError("The Xiaomi-only control path cannot change power or mode")

        ha.services.async_register("climate", "set_temperature", set_temperature)
        ha.services.async_register("climate", "set_hvac_mode", unsupported)

        def report(at, *, value=25, watts=400, fan="自动", update_power=True):
            entries[ADAPTER] = SimpleNamespace(state="cool", attributes={"temperature":value, "fan_mode":fan}, last_reported=at)
            entries[ROOM] = SimpleNamespace(domain="sensor", state="26", last_reported=at)
            if update_power:
                entries[POWER] = SimpleNamespace(domain="sensor", state=str(watts), last_reported=at)

        async def make(key, started):
            room = SimulationRoom(key, config, telemetry, lambda _: RegulatorFixture())
            room.hass = facade
            room.armed = True
            room.request = Request("cool", 24)
            room.started_at = started - timedelta(hours=2)
            room.ready_at = started - timedelta(minutes=10)
            report(started)
            await room.observe(started)
            return room

        current_room = await make("confirmation", START)
        sent = START + timedelta(minutes=1)
        report(sent)
        await apply(relay, current_room, sent)
        await current_room.send(sent)
        assert len(relay) == 1, "A pending command must suppress repeat dispatch"

        at = sent + timedelta(minutes=1, seconds=5)
        report(at, value=24)
        await current_room.observe(at)
        assert current_room.history.pending and current_room.command_result == "sent"

        first = at + timedelta(seconds=10)
        report(first, value=24, watts=560)
        await current_room.observe(first)
        assert current_room.history.pending, "An isolated power spike cannot settle a command"
        same = first + timedelta(seconds=35)
        report(same, value=24, update_power=False)
        await current_room.observe(same)
        assert current_room.history.pending, "The same power report cannot be counted twice"

        second = same + timedelta(seconds=12)
        report(second, value=24, watts=550)
        await current_room.observe(second)
        assert current_room.history.pending is None and current_room.command_result == "inferred_response"
        assert current_room.result.confidence == "inferred" and len(relay) == 1
        assert telemetry.last_command(current_room.key)["status"] == "inferred_response"
        print("Echo plus two separated power reports remains inferred. Re-reading a report produced no confirmation.")

        started = START + timedelta(hours=1)
        current_room = await make("timeout", started)
        sent = started + timedelta(minutes=1)
        report(sent)
        await apply(relay, current_room, sent)
        deadline = sent + timedelta(minutes=3)
        report(deadline, value=24)
        await current_room.observe(deadline)
        assert current_room.history.pending is None and current_room.command_result == "unconfirmed"
        assert current_room.history.manual_latched and current_room.result.command is None
        await current_room.send(deadline)
        assert len(relay) == 2
        restored, status = restore_command(telemetry.last_command(current_room.key), False)
        assert restored.manual_latched and status == "unconfirmed"
        print("An echoed setpoint with no physical evidence timed out, latched and remained paused after journal recovery.")

        started = START + timedelta(hours=2)
        current_room = await make("external", started)
        sent = started + timedelta(minutes=1)
        report(sent)
        await apply(relay, current_room, sent)
        at = sent + timedelta(minutes=1)
        report(at, value=24, fan="高")
        await current_room.observe(at)
        assert current_room.history.manual_latched, "A simultaneous external fan change must be treated as manual control"
        assert current_room.result.command is None and len(relay) == 3
        print("A competing Xiaomi state change triggered the manual hold without a mode, power or follow-up command.")

        started = START + timedelta(hours=3)
        current_room = await make("reconnect", started)
        outage = started + timedelta(minutes=1)
        entries[ADAPTER] = SimpleNamespace(state="unavailable", attributes={}, last_reported=outage)
        await current_room.observe(outage)
        assert current_room.transport_lost and current_room.result.command is None
        resumed = outage + timedelta(minutes=2)
        report(resumed)
        await current_room.observe(resumed)
        assert current_room.ready_at == resumed + timedelta(minutes=5)
        assert current_room.result.command is None and len(relay) == 3
        second_outage = resumed + timedelta(minutes=1)
        entries[ADAPTER] = SimpleNamespace(state="unavailable", attributes={}, last_reported=second_outage)
        await current_room.observe(second_outage)
        changed = second_outage + timedelta(minutes=1)
        report(changed, value=26)
        await current_room.observe(changed)
        assert current_room.history.manual_latched and current_room.result.command is None
        print("Reconnection waited for resynchronization and recognized an external setpoint changed during the outage.")


asyncio.run(run())
