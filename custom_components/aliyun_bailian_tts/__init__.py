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
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry

    # 加载 TTS 平台
    await async_load_platform(hass, TTS_DOMAIN, DOMAIN, entry.data, {})

    # 监听选项更新事件
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(hass, entry):
    """Unload the Aliyun BaiLian TTS component."""
    # 清理存储的数据
    if DOMAIN in hass.data:
        hass.data[DOMAIN].pop(entry.entry_id, None)
        if not hass.data[DOMAIN]:
            hass.data.pop(DOMAIN, None)
    return True


async def async_update_options(hass, config_entry):
    """Update options when they are changed."""
    # 当选项更新时，重新加载集成
    await hass.config_entries.async_reload(config_entry.entry_id)
