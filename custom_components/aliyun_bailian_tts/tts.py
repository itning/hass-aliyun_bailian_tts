import base64
import logging
import struct
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

    def _get_current_config(self):
        """获取当前的配置，优先使用 options"""
        config_entries = self.hass.config_entries.async_entries(DOMAIN)
        if config_entries:
            config_entry = config_entries[0]
            # 优先使用 options，如果没有则使用 data
            if config_entry.options:
                return config_entry.options
            else:
                return config_entry.data

        # 回退到旧的存储方式
        return self.hass.data.get(DOMAIN, {})

    @staticmethod
    def _process_qwen_tts(model: str, voice: str, message: str) -> bytes:
        def create_wav_header(data_size, channels=1, sample_rate=24000, bits_per_sample=16):
            """创建WAV文件头"""
            riff = b'RIFF'
            filesize = 36 + data_size
            wave_fmt = b'WAVE'
            fmt = b'fmt '
            subchunk1_size = 16
            audio_format = 1  # PCM
            byte_rate = sample_rate * channels * bits_per_sample // 8
            block_align = channels * bits_per_sample // 8
            subchunk2_id = b'data'

            header = riff + struct.pack('<I', filesize) + wave_fmt + fmt
            header += struct.pack('<IHHIIHH', subchunk1_size, audio_format, channels,
                                  sample_rate, byte_rate, block_align, bits_per_sample)
            header += subchunk2_id + struct.pack('<I', data_size)

            return header

        audio: bytes = bytes()
        responses = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
            model=model,
            text=message,
            voice=voice,
            stream=True
        )

        if responses is None:
            _LOGGER.error("qwen SpeechSynthesizer returned None or empty audio for message: %s model: %s voice: %s",
                          message, model, voice)
            raise HomeAssistantError("SpeechSynthesizer returned None or empty audio")

        input_tokens: int = 0
        output_tokens: int = 0
        for chunk in responses:
            if chunk.status_code != 200:
                _LOGGER.error(
                    "qwen SpeechSynthesizer returned error. status_code: %s request_id: %s code: %s message: %s model: %s voice: %s message: %s",
                    chunk.status_code, chunk.request_id, chunk.code, chunk.message, model, voice, message)
                raise HomeAssistantError("SpeechSynthesizer returned error: " + chunk.message)
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
        wav_header = create_wav_header(len(audio))
        audio = wav_header + audio
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
            # 获取当前配置
            config = self._get_current_config()

            dashscope.api_key = config.get(CONF_TOKEN)
            if not dashscope.api_key:
                raise ValueError("API Key is not set in configuration")

            model = config.get(CONF_MODEL, "qwen3-tts-flash")
            voice = config.get(CONF_VOICE, "Cherry")

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
                _LOGGER.error("SpeechSynthesizer returned None or empty audio for message: %s model: %s voice: %s",
                              message, model, voice)
                raise HomeAssistantError("SpeechSynthesizer returned None or empty audio")

            return audio_format, audio_data

        except Exception as e:
            _LOGGER.error("Error generating TTS audio: %s", e, exc_info=True)
            raise HomeAssistantError(f"Error generating TTS audio: {e}")


async def async_get_engine(hass, config, discovery_info=None):
    """Set up the Aliyun BaiLian TTS platform."""
    return AliyunBaiLianTTSProvider(hass, config)
