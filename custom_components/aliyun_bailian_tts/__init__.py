from homeassistant.components.tts import DOMAIN as TTS_DOMAIN
from homeassistant.helpers.discovery import async_load_platform
from .const import DOMAIN

async def async_setup(hass, config):
    """Set up the Aliyun BaiLian TTS component from configuration.yaml."""
    if config.get(DOMAIN):
        hass.data[DOMAIN] = config[DOMAIN]  # 存储配置到 hass.data
        hass.async_create_task(
            async_load_platform(hass, TTS_DOMAIN, DOMAIN, config[DOMAIN], config)
        )
    return True

async def async_setup_entry(hass, entry):
    """Set up the Aliyun BaiLian TTS component from a config entry."""
    hass.data[DOMAIN] = entry.data  # 存储配置到 hass.data
    hass.async_create_task(
        async_load_platform(hass, TTS_DOMAIN, DOMAIN, entry.data, {})
    )
    return True

async def async_unload_entry(hass, entry):
    """Unload the Aliyun BaiLian TTS component."""
    hass.data.pop(DOMAIN, None)
    return True