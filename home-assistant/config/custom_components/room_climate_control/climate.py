"""Virtual thermostats and a single coordinator for shared sensing."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import logging
from math import isfinite

from homeassistant.components.climate import ClimateEntity, ClimateEntityFeature, HVACAction, HVACMode
from homeassistant.const import UnitOfTemperature
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.service import async_register_admin_service
import voluptuous as vol
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt

from .policy import (
    ACTIVE, CONFIRM_TIMEOUT, MIN_DAY, MIN_QUIET, Command, Evidence, History, Limits, Pending, Reading,
    Request, choose_group, decide, demand, has_transport_evidence,
)
from . import DOMAIN

_LOGGER=logging.getLogger(__name__)

def number(value):
    try:
        parsed=float(value)
        return parsed if isfinite(parsed) else None
    except (ValueError,TypeError):
        return None

def reading(hass, entity):
    state=hass.states.get(entity) if entity else None
    if not state:
        return Reading()
    value=state.attributes.get("temperature") if state.domain=="weather" else state.state
    return Reading(number(value),state.last_reported)

def adapter_state(hass, entity):
    state = hass.states.get(entity)
    if state is None or state.state in {"unknown", "unavailable"}:
        return None
    return (state.state, number(state.attributes.get("temperature")), state.attributes.get("fan_mode"))

def restore_command(record, manual_latched):
    """An interrupted send is ambiguous across restarts and remains on manual hold."""
    if not record:
        return History(manual_latched=manual_latched), "尚无"
    timestamp = datetime.fromtimestamp(record["sent"], timezone.utc)
    value = number(record["value"]) if record["kind"] == "setpoint" else record["value"]
    pending = Pending(Command(record["kind"], value), timestamp)
    status = record["status"]
    uncertain = status in {"prepared", "sent", "unconfirmed", "transport_failed"}
    return History(last_command=pending, manual_latched=manual_latched or uncertain), (
        "interrupted" if status in {"prepared", "sent"} else status
    )

class Room(ClimateEntity,RestoreEntity):
    _attr_should_poll=False
    _attr_native_temperature_unit=UnitOfTemperature.CELSIUS
    _attr_min_temp=18
    _attr_max_temp=30
    _attr_target_temperature_step=0.5
    _attr_supported_features=ClimateEntityFeature.TARGET_TEMPERATURE|ClimateEntityFeature.PRESET_MODE
    _attr_hvac_modes=[HVACMode.OFF,HVACMode.HEAT,HVACMode.COOL,HVACMode.AUTO]
    _attr_preset_modes=["标准","睡眠静音"]

    def __init__(self,key,config,telemetry,make_regulator):
        self.key=key
        self.config=config
        self.telemetry=telemetry
        self.make_regulator=make_regulator
        self.regulator=None
        self.regulator_mode=None
        self.last_temperature_report=None
        self.regulated_setpoint=None
        self._attr_name=config["name"]+"恒温器"
        self._attr_unique_id="room_climate_control_"+key
        self.entity_id="climate."+key+"_room_thermostat"
        self.request=Request()
        self.evidence=Evidence()
        self.history=History()
        self.result=None
        self.last_proxy=None
        self.transport_lost=False
        self.last_power_band=None
        self.command_result="尚无"
        self.command_baseline=None
        self.power_confirmation_at=None
        self.coordination_group=config.get("group",key)
        self.default_offsets={"heat":config["heating_offset"],"cool":config["cooling_offset"]}
        self.offsets=self.default_offsets.copy()
        self.model_generations={"heat":None,"cool":None}
        self.limits=Limits(authorized=config["actuation_approved"],absolute_mode_verified=config["absolute_mode_verified"],
                           ir_setpoint_validated=config["ir_setpoint_validated"])
        self.daily={}
        self.armed=False
        self.ready_at=None
        self.started_at=None
        self.verified_target=Reading()

    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        manual_latched=False
        if (old:=await self.async_get_last_state()) and old.state in {m.value for m in self.hvac_modes}:
            target=number(old.attributes.get("temperature"))
            if target is not None and 18<=target<=30:
                self.request=Request(old.state,target,old.attributes.get("preset_mode")=="睡眠静音")
            self.armed=bool(self.limits.authorized and old.attributes.get("automatic_control_enabled"))
            manual_latched=old.attributes.get("control_hold_latched") is True
        record = await self.hass.async_add_executor_job(self.telemetry.last_command, self.key)
        self.history, self.command_result = restore_command(record, manual_latched)
        self.started_at=dt.now()
        self.ready_at=self.started_at+timedelta(minutes=5)
        await self.async_refresh_models()

    async def async_refresh_models(self):
        models=await self.hass.async_add_executor_job(self.telemetry.current_models,self.key,self.default_offsets)
        self.offsets={mode:model["offset"] for mode,model in models.items()}
        self.model_generations={mode:model["generation"] for mode,model in models.items()}
        self.async_write_ha_state()

    @property
    def hvac_mode(self):
        return HVACMode(self.request.mode)

    @property
    def hvac_action(self):
        observed=self.result.activity if self.result else None
        return {"heat":HVACAction.HEATING,"cool":HVACAction.COOLING,
                "off":HVACAction.OFF}.get(observed,HVACAction.IDLE)

    @property
    def native_current_temperature(self):
        room=self.evidence.room
        return room.value if room.fresh(dt.now(),self.limits.room_freshness,0,50) else None

    @property
    def native_target_temperature(self):
        return self.request.temperature

    @property
    def preset_mode(self):
        return "睡眠静音" if self.request.quiet else "标准"

    @property
    def extra_state_attributes(self):
        last=self.history.last_command
        return {
            "temperature_reference":self.config.get("temperature_reference","本区"),
            "automatic_control_enabled":self.armed,
            "configured_for_actuation":self.limits.authorized,
            "control_hold_latched":self.history.manual_latched,
            "reported_mode":self.evidence.adapter_mode,
            "reported_setpoint":self.evidence.adapter_setpoint,
            "observed_activity":self.result.activity if self.result else "unknown",
            "confidence":self.result.confidence if self.result else "unknown",
            "control_status":self.result.status if self.result else "waiting_for_observations",
            "temperature_reported_at":self.evidence.room.time.isoformat() if self.evidence.room.time else None,
            "temperature_entity_id":self.config["temperature"],
            "power_reported_at":self.evidence.power.time.isoformat() if self.evidence.power.time else None,
            "power_entity_id":self.config["power"],
            "power_observer_started_at":self.started_at.isoformat() if self.started_at else None,
            "recent_transport_evidence":self.evidence.adapter_available,
            "cooling_offset":self.offsets["cool"],"heating_offset":self.offsets["heat"],
            "cooling_model_generation":self.model_generations["cool"],
            "heating_model_generation":self.model_generations["heat"],
            "calibration_status":("版本已记录" if any(self.model_generations.values()) else
                "等待独立反馈与有效样本"),
            "last_command":f"{last.command.kind} {last.command.value}" if last else None,
            "last_command_at":last.sent.isoformat() if last else None,
            "last_command_result":self.command_result,"prior_day":self.daily,
        }

    async def async_set_temperature(self,**kwargs):
        value=number(kwargs.get("temperature"))
        if value is not None and 18<=value<=30:
            self.request=replace(self.request,temperature=value)
        if (mode:=kwargs.get("hvac_mode")) in self.hvac_modes:
            self.request=replace(self.request,mode=str(mode))
        self.async_write_ha_state()

    async def async_set_hvac_mode(self,hvac_mode):
        self.request=replace(self.request,mode=str(hvac_mode))
        self.async_write_ha_state()

    async def async_set_preset_mode(self,preset_mode):
        self.request=replace(self.request,quiet=preset_mode=="睡眠静音")
        self.async_write_ha_state()

    async def observe(self,now):
        state=self.hass.states.get(self.config["adapter"])
        proxy=adapter_state(self.hass,self.config["adapter"])
        mode,target,_=proxy or (None,None,None)
        power=reading(self.hass,self.config["power"])
        room=reading(self.hass,self.config["temperature"])
        available=has_transport_evidence(now,state.last_reported if proxy else None,power,self.limits)
        proxy_fresh=proxy is not None and timedelta(0)<=now-state.last_reported<=timedelta(minutes=30)
        if self.armed and self.last_proxy and not available:
            self.transport_lost=True
        elif available and self.transport_lost:
            self.transport_lost=False
            if self.armed:
                self.ready_at=now+timedelta(minutes=5)
        direct=reading(self.hass,self.config.get("verified_setpoint"))
        self.verified_target=direct
        direct_current=direct.fresh(now,self.limits.power_freshness,self.limits.min_setpoint,self.limits.max_setpoint)
        band=("active" if power.value is not None and power.value>=self.limits.active_watts else
              "low" if power.value is not None and power.value<=self.limits.low_watts else "unknown")
        band_since=self.evidence.power_stable_since if band==self.last_power_band else now
        self.last_power_band=band
        physical=self.hass.states.get(self.config["verified_mode"]) if "verified_mode" in self.config else None
        independent=physical.state if physical and physical.state in ACTIVE|{"off"} and timedelta(0)<=now-physical.last_reported<=self.limits.power_freshness else None
        phase=self.evidence.phase_since if independent==self.evidence.independent_mode else now
        active_start=self.evidence.last_active_start
        active_mode=self.evidence.last_active_mode
        if independent in ACTIVE and independent!=self.evidence.independent_mode:
            active_start,active_mode=now,independent
        self.evidence=Evidence(room,power,mode,target,independent,phase,active_mode,active_start,band_since,available,
                               self.started_at,direct.value if direct_current else None,proxy_fresh)
        pending=self.history.pending
        if proxy and self.last_proxy and proxy!=self.last_proxy:
            own=bool(pending and proxy[2]==self.last_proxy[2] and (
                    (pending.command.kind=="setpoint" and proxy[1]==pending.command.value and proxy[0]==self.last_proxy[0])
                    or (pending.command.kind=="mode" and proxy[0]==pending.command.value)))
            if not own and self.armed:
                self.history=replace(self.history,manual_latched=True)
        if proxy:
            self.last_proxy=proxy
        if pending:
            verified=(pending.command.kind=="mode" and independent==pending.command.value
                      and physical.last_reported>pending.sent) or (
                      pending.command.kind=="setpoint" and direct.value==pending.command.value
                      and direct_current
                      and direct.time>pending.sent)
            # An IR proxy echo never confirms execution. A sustained change in
            # independent measured power may support a separately marked inference.
            inferred=False
            qualifying_power=False
            if (not verified and not self.history.manual_latched and self.limits.ir_setpoint_validated
                and pending.command.kind=="setpoint"
                and self.command_baseline and target==pending.command.value
                and power.fresh(now,self.limits.power_freshness,0,10_000)
                and power.time>pending.sent+timedelta(minutes=1)):
                old_mode,old_target,old_watts=self.command_baseline
                if mode==old_mode and old_mode in ACTIVE and old_target is not None and old_watts is not None:
                    direction=(pending.command.value-old_target)*(1 if mode=="heat" else -1)
                    qualifying_power=(power.value-old_watts)*direction>=max(50,0.2*old_watts)
            if qualifying_power:
                if self.power_confirmation_at is None:
                    self.power_confirmation_at=power.time
                else:
                    inferred=power.time-self.power_confirmation_at>=timedelta(seconds=30)
            else:
                self.power_confirmation_at=None
            timeout=now-pending.sent>=CONFIRM_TIMEOUT
            if verified or inferred or timeout:
                self.command_result="verified" if verified else "inferred_response" if inferred else "unconfirmed"
                await self.hass.async_add_executor_job(self.telemetry.command,self.key,int(pending.sent.timestamp()),
                    pending.command.kind,pending.command.value,self.command_result,int(now.timestamp()))
                self.history=replace(self.history,pending=None,
                                     manual_latched=not (verified or inferred) or self.history.manual_latched)
        selected=demand(self.request,room.value,independent or mode) if room.value is not None else None
        ready=self.armed and self.ready_at is not None and now>=self.ready_at
        healthy=(ready and not self.history.manual_latched and not self.history.pending
                 and available
                 and selected in ACTIVE
                 and ((independent==selected and direct_current) or (not independent and proxy_fresh
                      and self.limits.ir_setpoint_validated and mode==selected
                      and band=="active" and power.fresh(now,self.limits.power_freshness,0,10_000)))
                 and room.fresh(now,self.limits.room_freshness,0,50))
        if healthy:
            if self.regulator is None or self.regulator_mode!=selected:
                self.regulator=self.make_regulator(self.request.temperature)
                self.regulator_mode=selected
                self.last_temperature_report=None
            self.regulator.set_target_temp(self.request.temperature)
            if self.last_temperature_report!=room.time:
                self.regulated_setpoint=self.regulator.calculate_regulated_temperature(room.value,None,1.0)
                self.last_temperature_report=room.time
        else:
            self.regulator=None
            self.regulator_mode=None
            self.regulated_setpoint=None
        self.result=decide(now,self.request,self.evidence,replace(self.limits,authorized=ready),
                         self.history,self.regulated_setpoint,self.offsets.get(selected,0))
        self.async_write_ha_state()

    async def send(self,now):
        cmd=self.result.command if self.result else None
        if (not cmd or not self.armed or not self.limits.authorized or self.history.pending
            or self.history.manual_latched):
            return
        request=self.request
        proxy=self.last_proxy
        pending=Pending(cmd,now)
        timestamp=int(now.timestamp())
        await self.hass.async_add_executor_job(self.telemetry.command,self.key,timestamp,
            cmd.kind,cmd.value,"prepared")
        live_proxy=adapter_state(self.hass,self.config["adapter"])
        if not self.armed or request!=self.request or proxy!=live_proxy:
            if proxy!=live_proxy and self.armed:
                self.history=replace(self.history,manual_latched=True)
            self.command_result="canceled"
            await self.hass.async_add_executor_job(self.telemetry.command,self.key,timestamp,
                cmd.kind,cmd.value,self.command_result,int(dt.now().timestamp()))
            self.async_write_ha_state()
            return
        self.history=replace(self.history,last_command=pending,pending=pending)
        self.command_result="sent"
        self.command_baseline=(self.evidence.adapter_mode,self.evidence.adapter_setpoint,self.evidence.power.value)
        self.power_confirmation_at=None
        action,key=("set_temperature","temperature") if cmd.kind=="setpoint" else ("set_hvac_mode","hvac_mode")
        try:
            async with asyncio.timeout(35):
                await self.hass.services.async_call("climate",action,{"entity_id":self.config["adapter"],key:cmd.value},blocking=True)
            await self.hass.async_add_executor_job(self.telemetry.command,self.key,timestamp,
                cmd.kind,cmd.value,"sent")
        except Exception as error:
            self.command_result="transport_failed"
            _LOGGER.warning("Room %s climate service failed with %s",self.key,type(error).__name__)
            self.history=replace(self.history,pending=None,manual_latched=True)
            await self.hass.async_add_executor_job(self.telemetry.command,self.key,timestamp,
                cmd.kind,cmd.value,self.command_result,int(dt.now().timestamp()))
        self.async_write_ha_state()

    def sample(self,now):
        ev=self.evidence
        actual=ev.room.value if ev.room.fresh(now,self.limits.room_freshness,0,50) else None
        power=ev.power.value if ev.power.fresh(now,self.limits.power_freshness,0,10000) else None
        humidity=reading(self.hass,self.config.get("humidity"))
        outdoor=reading(self.hass,self.config.get("outdoor"))
        return self.key,int(now.timestamp())//60*60,now.date().isoformat(),actual,(
          humidity.value if humidity.fresh(now,self.limits.room_freshness,0,100) else None
        ),self.request.temperature,ev.adapter_setpoint,ev.adapter_mode,self.result.activity,self.result.confidence,power,(
          outdoor.value if outdoor.fresh(now,timedelta(hours=2),-50,65) else None
        ),int(ev.room.time.timestamp()) if ev.room.time else None,self.result.status,(
          self.verified_target.value if self.verified_target.fresh(now,self.limits.power_freshness,16,30) else None
        )

async def async_setup_platform(hass,config,async_add_entities,discovery_info=None):
    if not discovery_info:
        return
    from custom_components.versatile_thermostat.const import RegulationParamLight as Params
    from custom_components.versatile_thermostat.pi_algorithm import PITemperatureRegulator
    data=hass.data[DOMAIN]
    telemetry=data.telemetry
    def regulator(target):
        return PITemperatureRegulator(target,Params.kp,Params.ki,0,Params.offset_max,
                                      Params.accumulated_error_threshold,Params.overheat_protection)
    rooms={key:Room(key,setting,telemetry,regulator) for key,setting in discovery_info["rooms"].items()}
    async_add_entities(list(rooms.values()))

    async def set_room_armed(call):
        room=rooms[call.data["room"]]
        confirmed_path=("verified_mode" in room.config and "verified_setpoint" in room.config)
        if call.data["enabled"] and not (room.limits.authorized and
                                         (confirmed_path or room.config["ir_setpoint_validated"])):
            raise HomeAssistantError("This appliance is limited to observation.")
        room.armed=call.data["enabled"]
        if room.armed:
            room.history=replace(room.history,manual_latched=False)
            room.ready_at=dt.now()+timedelta(minutes=5)
        if getattr(room,"hass",None):
            room.async_write_ha_state()

    async_register_admin_service(hass,"room_climate_control","set_automatic_control",set_room_armed,
        schema=vol.Schema({vol.Required("room"):vol.In(rooms),vol.Required("enabled"):bool}))

    async def set_offsets(call):
        room=rooms[call.data["room"]]
        offsets={mode:round(call.data[field],2) for mode,field in (
            ("cool","cooling_offset"),("heat","heating_offset")) if field in call.data}
        if not offsets or not all(map(isfinite,offsets.values())):
            raise HomeAssistantError("Provide at least one finite temperature offset.")
        async with data.update_lock:
            current=await hass.async_add_executor_job(telemetry.current_models,room.key,room.default_offsets)
            await hass.async_add_executor_job(telemetry.save_manual_offsets,room.key,offsets,current)
            await room.async_refresh_models()

    offset=vol.All(vol.Coerce(float),vol.Range(min=-4,max=4))
    async_register_admin_service(hass,"room_climate_control","set_offsets",set_offsets,
        schema=vol.Schema({vol.Required("room"):vol.In(rooms),vol.Optional("cooling_offset"):offset,
                           vol.Optional("heating_offset"):offset}))
    last_day=None
    async def tick(_):
        if data.update_lock.locked():
            return
        async with data.update_lock:
            await coordinate()

    async def coordinate():
        nonlocal last_day
        now=dt.now()
        active=[r for r in rooms.values() if getattr(r,"hass",None)]
        for r in active:
            await r.observe(now)
        await hass.async_add_executor_job(telemetry.record,[r.sample(now) for r in active])
        grouped={}
        for r in active: grouped.setdefault(r.coordination_group,[]).append(r)
        for group in grouped.values():
            interval=MIN_QUIET if any(r.request.quiet for r in group) else MIN_DAY
            if any(r.history.last_command and now-r.history.last_command.sent<interval for r in group):
                continue
            key=choose_group([(r.key,r.request,r.evidence,r.result) for r in group])
            if key:
                await rooms[key].send(now)
        if now.date()!=last_day and active:
            setting={r.key:r.default_offsets for r in active}
            await hass.async_add_executor_job(telemetry.maintenance,now,setting)
            for r in active:
                r.daily=await hass.async_add_executor_job(telemetry.current_summary,r.key)
                await r.async_refresh_models()
            last_day=now.date()
    async_track_time_interval(hass,tick,timedelta(minutes=1),cancel_on_shutdown=True)
