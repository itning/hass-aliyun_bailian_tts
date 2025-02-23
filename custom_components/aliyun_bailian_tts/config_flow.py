import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from .const import DOMAIN, CONF_TOKEN, CONF_MODEL, CONF_VOICE

class AliyunBaiLianTTSConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Aliyun BaiLian TTS config flow."""

    async def async_step_user(self, user_input=None):
        """Handle a flow initiated by the user."""
        # 检查是否已经有配置条目
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        # 创建一个空的配置条目
        return self.async_create_entry(title="Aliyun BaiLian TTS", data={})

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return AliyunBaiLianTTSOptionsFlowHandler(config_entry)

class AliyunBaiLianTTSOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle an options flow for Aliyun BaiLian TTS."""

    def __init__(self, config_entry):
        """Initialize options flow."""
        self._config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if user_input is not None:
            # 更新配置到 hass.data
            self.hass.data[DOMAIN] = user_input
            return self.async_create_entry(title="Aliyun BaiLian TTS Options", data=user_input)

        # 默认值从现有配置中读取，如果不存在则使用默认值
        default_token = self._config_entry.options.get(CONF_TOKEN, "")
        default_model = self._config_entry.options.get(CONF_MODEL, "cosyvoice-v1")
        default_voice = self._config_entry.options.get(CONF_VOICE, "longxiaochun")

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required(CONF_TOKEN, default=default_token): str,
                vol.Optional(CONF_MODEL, default=default_model): str,
                vol.Optional(CONF_VOICE, default=default_voice): str,
            })
        )
