from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult

from .const import DOMAIN, CONF_TOKEN, CONF_MODEL, CONF_VOICE, CONF_NAME

OPTIONS_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_NAME): str,
        vol.Required(CONF_TOKEN): str,
        vol.Optional(CONF_MODEL, default="qwen3-tts-flash"): str,
        vol.Optional(CONF_VOICE, default="Cherry"): str,
    }
)


class AliyunBaiLianTTSConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Aliyun BaiLian TTS config flow."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle a flow initiated by the user."""
        errors = {}

        # 如果用户提供了输入，验证并创建配置条目
        if user_input is not None:
            # 基本验证
            if not user_input.get(CONF_TOKEN):
                errors[CONF_TOKEN] = "invalid_token"

            if not user_input.get(CONF_NAME):
                errors[CONF_NAME] = "invalid_name"

            # 检查名称是否已存在
            existing_entries = self._async_current_entries()
            for entry in existing_entries:
                entry_name = entry.options.get(CONF_NAME) or entry.data.get(CONF_NAME)
                if entry_name == user_input.get(CONF_NAME):
                    errors[CONF_NAME] = "name_exists"
                    break

            if not errors:
                # 创建配置条目，使用自定义名称作为标题
                return self.async_create_entry(
                    title=f"Aliyun BaiLian TTS - {user_input[CONF_NAME]}",
                    data=user_input,
                    options=user_input  # 同时保存到 options 中
                )

        # 显示表单让用户输入配置
        return self.async_show_form(
            step_id="user",
            data_schema=OPTIONS_SCHEMA,
            errors=errors,
            description_placeholders={
                "name": "为这个TTS实例起一个唯一的名称，例如：女声助手、男声助手等"
            }
        )

    async def async_step_import(self, import_config: dict[str, Any]) -> FlowResult:
        """Handle import from configuration.yaml."""
        # 为导入的配置生成默认名称
        if CONF_NAME not in import_config:
            import_config[CONF_NAME] = "Default"

        # 检查名称是否已存在
        existing_entries = self._async_current_entries()
        for entry in existing_entries:
            entry_name = entry.options.get(CONF_NAME) or entry.data.get(CONF_NAME)
            if entry_name == import_config[CONF_NAME]:
                return self.async_abort(reason="name_exists")

        # 从 configuration.yaml 导入时也同时保存到 options
        return self.async_create_entry(
            title=f"Aliyun BaiLian TTS - {import_config[CONF_NAME]}",
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

            if not user_input.get(CONF_NAME):
                errors[CONF_NAME] = "invalid_name"

            # 检查名称是否与其他实例冲突（排除当前实例）
            existing_entries = self.hass.config_entries.async_entries(DOMAIN)
            for entry in existing_entries:
                if entry.entry_id != self._config_entry.entry_id:
                    entry_name = entry.options.get(CONF_NAME) or entry.data.get(CONF_NAME)
                    if entry_name == user_input.get(CONF_NAME):
                        errors[CONF_NAME] = "name_exists"
                        break

            if not errors:
                # 更新配置条目的标题
                self.hass.config_entries.async_update_entry(
                    self._config_entry,
                    title=f"Aliyun BaiLian TTS - {user_input[CONF_NAME]}"
                )
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