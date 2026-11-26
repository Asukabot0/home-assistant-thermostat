"""Include the controller's SQLite state consistently in HA backups."""

from homeassistant.helpers.restore_state import RestoreStateData

from . import DOMAIN


async def async_pre_backup(hass):
    """Pause writes while Home Assistant archives the checkpointed database."""
    data = hass.data[DOMAIN]
    await data.update_lock.acquire()
    data.backup_locked = True
    try:
        await RestoreStateData.async_save_persistent_states(hass)
        await hass.async_add_executor_job(data.telemetry.checkpoint)
    except BaseException:
        await async_post_backup(hass)
        raise


async def async_post_backup(hass):
    """Resume control and sampling after the archive is written."""
    data = hass.data[DOMAIN]
    if data.backup_locked:
        data.backup_locked = False
        data.update_lock.release()
