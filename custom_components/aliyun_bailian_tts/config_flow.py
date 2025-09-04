from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult

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

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle a flow initiated by the user."""
        # 检查是否已经有配置条目
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        errors = {}

        # 如果用户提供了输入，验证并创建配置条目
        if user_input is not None:
            # 基本验证
            if not user_input.get(CONF_TOKEN):
                errors[CONF_TOKEN] = "invalid_token"

            if not errors:
                # 创建配置条目，同时将数据保存到 options 中
                return self.async_create_entry(
                    title="Aliyun BaiLian TTS",
                    data=user_input,
                    options=user_input  # 同时保存到 options 中
                )

        # 显示表单让用户输入配置
        return self.async_show_form(
            step_id="user",
            data_schema=OPTIONS_SCHEMA,
            errors=errors
        )

    async def async_step_import(self, import_config: dict[str, Any]) -> FlowResult:
        """Handle import from configuration.yaml."""
        # 检查是否已经有配置条目
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        # 从 configuration.yaml 导入时也同时保存到 options
        return self.async_create_entry(
            title="Aliyun BaiLian TTS",
            data=import_config,
            options=import_config
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return AliyunBaiLianTTSOptionsFlowHandler(config_entry)


class AliyunBaiLianTTSOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle an options flow for Aliyun BaiLian TTS."""

    def __init__(self, config_entry: config_entries.ConfigEntry):
        """Initialize options flow."""
        self._config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage the options."""
        errors = {}

        if user_input is not None:
            # 验证输入
            if not user_input.get(CONF_TOKEN):
                errors[CONF_TOKEN] = "invalid_token"

            if not errors:
                return self.async_create_entry(title="", data=user_input)

        # 获取当前的配置值，优先从 options 读取，如果没有则从 data 读取
        current_config = {}
        if self.config_entry.options:
            current_config = self.config_entry.options
        else:
            # 第一次打开设置页面，从 data 中读取
            current_config = self.config_entry.data

        # 显示表单，用当前配置作为默认值
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                OPTIONS_SCHEMA, current_config
            ),
            errors=errors
        )
