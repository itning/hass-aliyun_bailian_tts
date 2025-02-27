from homeassistant.components.tts import DOMAIN as TTS_DOMAIN
from homeassistant.helpers.discovery import async_load_platform
from .const import DOMAIN

async def async_setup(hass, config):
    """Set up the Aliyun BaiLian TTS component from configuration.yaml."""
    if config.get(DOMAIN):
        # 如果配置来自 configuration.yaml，建议将其迁移到 config_entry
        hass.async_create_task(
            hass.config_entries.flow.async_init(
                DOMAIN, context={"source": "import"}, data=config[DOMAIN]
            )
        )
    return True

async def async_setup_entry(hass, entry):
    """Set up the Aliyun BaiLian TTS component from a config entry."""
    # 将配置存储到 hass.data
    hass.data[DOMAIN] = entry.data
    hass.async_create_task(
        async_load_platform(hass, TTS_DOMAIN, DOMAIN, entry.data, {})
    )
    return True

async def async_unload_entry(hass, entry):
    """Unload the Aliyun BaiLian TTS component."""
    hass.data.pop(DOMAIN, None)
    return True