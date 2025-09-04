import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback

from .const import DOMAIN, CONF_TOKEN, CONF_MODEL, CONF_VOICE

OPTIONS_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_TOKEN): str,
        vol.Optional(CONF_MODEL, default="cosyvoice-v1"): str,
        vol.Optional(CONF_VOICE, default="longxiaochun"): str,
    }
)

class AliyunBaiLianTTSConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Aliyun BaiLian TTS config flow."""

    async def async_step_user(self, user_input=None):
        """Handle a flow initiated by the user."""
        # 检查是否已经有配置条目
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        # 如果用户提供了输入，创建配置条目
        if user_input is not None:
            return self.async_create_entry(title="Aliyun BaiLian TTS", data=user_input)

        # 显示表单让用户输入配置
        return self.async_show_form(
            step_id="user",
            data_schema=OPTIONS_SCHEMA
        )

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
            # 更新配置到 config_entry
            return self.async_create_entry(title="Aliyun BaiLian TTS Options", data=user_input)

        # 获取当前的配置值，优先从 options 读取，如果没有则从 data 读取
        current_config = {}
        if self.config_entry.options:
            # 如果已经有 options，使用它们
            current_config = self.config_entry.options
        else:
            # 如果没有 options，从 data 中读取（第一次打开设置页面的情况）
            current_config = self.config_entry.data

        # 显示表单，用当前配置作为默认值
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                OPTIONS_SCHEMA, current_config
            ),
        )
