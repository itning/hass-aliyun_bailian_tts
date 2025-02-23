from .const import CONF_TOKEN, CONF_MODEL, CONF_VOICE, DOMAIN
import time
from homeassistant.components.tts import Provider
from homeassistant.exceptions import HomeAssistantError
import logging
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer

_LOGGER = logging.getLogger(__name__)


class AliyunBaiLianTTSProvider(Provider):
    def __init__(self, hass, config):
        self.hass = hass
        self.name = "Aliyun Bailian TTS"

    @property
    def default_language(self):
        return "zh"

    @property
    def supported_languages(self):
        return ["en", "zh"]

    async def async_get_tts_audio(self, message, language, options=None):
        """Generate TTS audio."""
        try:
            # 动态读取最新配置
            config = self.hass.data.get(DOMAIN, {})
            dashscope.api_key = config.get(CONF_TOKEN)
            model = config.get(CONF_MODEL, "cosyvoice-v1")
            voice = config.get(CONF_VOICE, "longxiaochun")

            synthesizer = SpeechSynthesizer(model=model, voice=voice)

            start_time = time.perf_counter()
            audio = await self.hass.async_add_executor_job(synthesizer.call, message)
            end_time = time.perf_counter()

            elapsed_time = (end_time - start_time) * 1000
            _LOGGER.info(
                '[Metric] requestId: %s, first package delay ms: %s, elapsed_time: %sms',
                synthesizer.get_last_request_id(),
                synthesizer.get_first_package_delay(),
                elapsed_time
            )

            return "mp3", audio
        except Exception as e:
            _LOGGER.error("Error generating TTS audio: %s", e)
            raise HomeAssistantError(f"Error generating TTS audio: {e}")


async def async_get_engine(hass, config, discovery_info=None):
    """Set up the Aliyun BaiLian TTS platform."""
    return AliyunBaiLianTTSProvider(hass, config)
