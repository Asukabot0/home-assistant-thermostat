"""Configure locally scheduled room climate control."""
from asyncio import Lock
from dataclasses import dataclass
from pathlib import Path

from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.discovery import async_load_platform
import voluptuous as vol

from .telemetry import Telemetry

DOMAIN = "room_climate_control"
ROOM = vol.Schema({
    vol.Required("name"): cv.string,
    vol.Required("adapter"): cv.entity_id,
    vol.Required("temperature"): cv.entity_id,
    vol.Required("power"): cv.entity_id,
    vol.Optional("humidity"): cv.entity_id,
    vol.Optional("outdoor"): cv.entity_id,
    vol.Optional("verified_mode"): cv.entity_id,
    vol.Optional("verified_setpoint"): cv.entity_id,
    vol.Optional("group"): cv.string,
    vol.Optional("temperature_reference"): cv.string,
    vol.Optional("actuation_approved", default=False): cv.boolean,
    vol.Optional("absolute_mode_verified", default=False): cv.boolean,
    vol.Optional("ir_setpoint_validated", default=False): cv.boolean,
    vol.Optional("cooling_offset", default=0.0): vol.All(vol.Coerce(float),vol.Range(min=-4,max=4)),
    vol.Optional("heating_offset", default=0.0): vol.All(vol.Coerce(float),vol.Range(min=-4,max=4)),
})
CONFIG_SCHEMA=vol.Schema({vol.Optional(DOMAIN):{"rooms":{cv.slug:ROOM}}},extra=vol.ALLOW_EXTRA)

@dataclass
class RoomClimateData:
    telemetry: Telemetry
    update_lock: Lock
    backup_locked: bool = False

async def async_setup(hass, config):
    telemetry=Telemetry(Path(hass.config.path(".storage","room_climate_control.sqlite3")))
    await hass.async_add_executor_job(telemetry.initialize)
    hass.data[DOMAIN]=RoomClimateData(telemetry, Lock())
    hass.async_create_task(async_load_platform(hass,"climate",DOMAIN,config[DOMAIN],config))
    return True
