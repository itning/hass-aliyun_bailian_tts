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
            if not errors:
                await self.async_set_unique_id(user_input[CONF_NAME])
                self._abort_if_unique_id_configured()

            if not errors:
                # 创建配置条目，同时将数据保存到 options 中
                return self.async_create_entry(
                    title=user_input[CONF_NAME],
                    data=user_input,
                    options=user_input  # 同时保存到 options 中
                )

        # 显示表单让用户输入配置
        return self.async_show_form(
            step_id="user",
            data_schema=OPTIONS_SCHEMA,
            errors=errors,
            description_placeholders={
                "name_help": "给这个TTS配置起一个唯一的名称，例如：阿里云TTS-中文、阿里云TTS-英文"
            }
        )

    async def async_step_import(self, import_config: dict[str, Any]) -> FlowResult:
        """Handle import from configuration.yaml."""
        # 如果没有名称，使用默认名称
        if CONF_NAME not in import_config:
            import_config[CONF_NAME] = "Aliyun BaiLian TTS"

        # 检查名称是否已存在
        await self.async_set_unique_id(import_config[CONF_NAME])
        self._abort_if_unique_id_configured()

        # 从 configuration.yaml 导入时也同时保存到 options
        return self.async_create_entry(
            title=import_config[CONF_NAME],
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

            # 如果名称改变了，检查新名称是否已存在
            current_name = self.config_entry.data.get(CONF_NAME, self.config_entry.title)
            if user_input.get(CONF_NAME) != current_name:
                # 检查新名称是否与其他条目冲突
                for entry in self.hass.config_entries.async_entries(DOMAIN):
                    if entry.entry_id != self.config_entry.entry_id:
                        other_name = entry.data.get(CONF_NAME, entry.title)
                        if other_name == user_input[CONF_NAME]:
                            errors[CONF_NAME] = "name_exists"
                            break

            if not errors:
                # 更新 unique_id
                if user_input.get(CONF_NAME) != current_name:
                    self.hass.config_entries.async_update_entry(
                        self.config_entry,
                        unique_id=user_input[CONF_NAME],
                        title=user_input[CONF_NAME]
                    )

                return self.async_create_entry(title="", data=user_input)

        # 获取当前的配置值，优先从 options 读取，如果没有则从 data 读取
        current_config = {}
        if self.config_entry.options:
            current_config = self.config_entry.options
        else:
            # 第一次打开设置页面，从 data 中读取
            current_config = self.config_entry.data.copy()
            # 如果旧配置没有名称字段，使用 title
            if CONF_NAME not in current_config:
                current_config[CONF_NAME] = self.config_entry.title

        # 显示表单，用当前配置作为默认值
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                OPTIONS_SCHEMA, current_config
            ),
            errors=errors
        )