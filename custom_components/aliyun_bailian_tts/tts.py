import base64
import logging
import time

import dashscope
from homeassistant.components.tts import Provider, TtsAudioType
from homeassistant.exceptions import HomeAssistantError

from .const import CONF_TOKEN, CONF_MODEL, CONF_VOICE, DOMAIN

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

    @staticmethod
    def _process_qwen_tts(model: str, voice: str, message: str) -> bytes:
        audio: bytes = bytes()
        responses = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
            model=model,
            text=message,
            voice=voice,
            stream=True
        )

        if responses is None:
            _LOGGER.error("SpeechSynthesizer returned None or empty audio for message: %s", message)
            raise HomeAssistantError("SpeechSynthesizer returned None or empty audio")

        input_tokens: int = 0
        output_tokens: int = 0
        for chunk in responses:
            if (chunk.get("output") and
                    chunk["output"].get("audio") and
                    chunk["output"]["audio"].get("data")):
                audio_string = chunk["output"]["audio"]["data"]
                wav_bytes: bytes = base64.b64decode(audio_string)
                audio = audio + wav_bytes

            if chunk.get("usage"):
                input_tokens += chunk["usage"].get("input_tokens", 0)
                output_tokens += chunk["usage"].get("output_tokens", 0)
        _LOGGER.info("input_tokens: %d, output_tokens: %d", input_tokens, output_tokens)

        return audio

    @staticmethod
    def _process_cosyvoice_tts(model: str, voice: str, message: str) -> bytes:
        start_time = time.perf_counter()
        synthesizer = dashscope.audio.tts_v2.SpeechSynthesizer(model=model, voice=voice)
        audio_data = synthesizer.call(message)
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000
        _LOGGER.info(
            '[Metric] requestId: %s, first package delay ms: %s, elapsed_time: %sms',
            synthesizer.get_last_request_id(),
            synthesizer.get_first_package_delay(),
            elapsed_time
        )
        return audio_data

    async def async_get_tts_audio(self, message: str, language: str, options=None) -> TtsAudioType:
        """Generate TTS audio."""
        try:
            # 动态读取最新配置
            config = self.hass.data.get(DOMAIN, {})
            dashscope.api_key = config.get(CONF_TOKEN)
            if not dashscope.api_key:
                raise ValueError("API Key is not set in configuration")

            model = config.get(CONF_MODEL, "cosyvoice-v1")
            voice = config.get(CONF_VOICE, "longxiaochun")

            if model.startswith("qwen"):
                audio_data = await self.hass.async_add_executor_job(self._process_qwen_tts, model, voice, message)
                audio_format = "wav"
            elif model.startswith("cosyvoice"):
                audio_data = await self.hass.async_add_executor_job(self._process_cosyvoice_tts, model, voice, message)
                audio_format = "mp3"
            else:
                _LOGGER.error("not supported model: %s", model)
                raise HomeAssistantError("not supported model: " + model)
            if audio_data is None or len(audio_data) == 0:
                _LOGGER.error("SpeechSynthesizer returned None or empty audio for message: %s", message)
                raise HomeAssistantError("SpeechSynthesizer returned None or empty audio")

            return audio_format, audio_data

        except Exception as e:
            _LOGGER.error("Error generating TTS audio: %s", e, exc_info=True)
            raise HomeAssistantError(f"Error generating TTS audio: {e}")


async def async_get_engine(hass, config, discovery_info=None):
    """Set up the Aliyun BaiLian TTS platform."""
    return AliyunBaiLianTTSProvider(hass, config)
