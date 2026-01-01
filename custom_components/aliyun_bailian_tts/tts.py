"""Aliyun BaiLian TTS platform using TextToSpeechEntity."""
import asyncio
import base64
import logging
import queue
import struct
import threading
import time
from typing import Any, AsyncGenerator

import dashscope
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat
from homeassistant.components.tts import (
    TextToSpeechEntity,
    TtsAudioType,
    ATTR_VOICE, TTSAudioRequest, TTSAudioResponse,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_TOKEN, CONF_MODEL, CONF_VOICE, DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Aliyun BaiLian TTS entity."""
    async_add_entities([AliyunBaiLianTTSEntity(hass, config_entry)])


class StreamingCallback(QwenTtsRealtimeCallback):
    """Callback handler for streaming TTS (Thread-Safe)."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.audio_queue = asyncio.Queue()
        self.complete_event = threading.Event()
        self.error = None

    def on_open(self) -> None:
        _LOGGER.debug("Streaming connection opened")

    def on_close(self, close_status_code, close_msg) -> None:
        _LOGGER.debug("Streaming connection closed: %s - %s", close_status_code, close_msg)
        self.complete_event.set()

    def on_event(self, response: dict) -> None:
        """Handle streaming events from SDK thread."""
        try:
            event_type = response.get("type")

            if event_type == "response.audio.delta":
                audio_chunk = base64.b64decode(response["delta"])
                _LOGGER.debug("Received audio chunk: %d bytes", len(audio_chunk))
                self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, audio_chunk)
            elif event_type == "session.created":
                _LOGGER.debug("Session created: %s", response.get('session', {}).get('id'))
            elif event_type == "response.done":
                _LOGGER.debug("Synthesis response complete")
                self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, None)
            elif event_type == "session.finished":
                _LOGGER.debug("Session finished")
                self.complete_event.set()

        except Exception as e:
            _LOGGER.error("Error in callback processing: %s", e)
            self.error = e
            self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, None)
            self.complete_event.set()

    async def get_audio_chunk(self) -> bytes | None:
        return await self.audio_queue.get()

    def wait_for_finished(self):
        if not self.complete_event.wait(timeout=10):
            _LOGGER.warning("Wait for session finish timed out")


def create_wav_header(
        data_size: int,
        channels: int = 1,
        sample_rate: int = 24000,
        bits_per_sample: int = 16,
) -> bytes:
    """Create WAV file header."""
    riff = b"RIFF"
    filesize = 36 + data_size
    wave_fmt = b"WAVE"
    fmt = b"fmt "
    subchunk1_size = 16
    audio_format = 1  # PCM
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    subchunk2_id = b"data"

    header = riff + struct.pack("<I", filesize) + wave_fmt + fmt
    header += struct.pack(
        "<IHHIIHH",
        subchunk1_size,
        audio_format,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    header += subchunk2_id + struct.pack("<I", data_size)

    return header


def _sync_tts_worker(model, voice, api_key, callback, text_queue):
    """Runs in executor thread."""
    dashscope.api_key = api_key
    qwen_tts = QwenTtsRealtime(
        model=model,
        callback=callback,
        url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
    )
    try:
        qwen_tts.connect()
        qwen_tts.update_session(
            voice=voice,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode="server_commit",
        )
        while True:
            text_chunk = text_queue.get()
            if text_chunk is None: break
            _LOGGER.debug("append_text %s", text_chunk)
            qwen_tts.append_text(text_chunk)
        qwen_tts.finish()
    except Exception as e:
        _LOGGER.error("Worker Error: %s", e)
        callback.error = e
        callback.loop.call_soon_threadsafe(callback.audio_queue.put_nowait, None)
    finally:
        callback.complete_event.set()


class AliyunBaiLianTTSEntity(TextToSpeechEntity):
    """Aliyun BaiLian TTS entity."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize TTS entity."""
        self.hass = hass
        self._config_entry = config_entry
        self._attr_name = "Aliyun BaiLian TTS"
        self._attr_unique_id = f"{DOMAIN}_{config_entry.entry_id}"

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return "zh"

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        return ["en", "zh"]

    @property
    def supported_options(self) -> list[str]:
        """Return list of supported options."""
        return [ATTR_VOICE]

    @property
    def default_options(self) -> dict[str, Any]:
        """Return default options."""
        config = self._get_current_config()
        return {
            ATTR_VOICE: config.get(CONF_VOICE, "Cherry"),
        }

    def _get_current_config(self) -> dict[str, Any]:
        """Get current configuration, preferring options over data."""
        if self._config_entry.options:
            return self._config_entry.options
        return self._config_entry.data

    @staticmethod
    def _process_qwen_tts(model: str, voice: str, message: str, api_key: str) -> bytes:
        """Process Qwen TTS synthesis."""
        audio: bytes = bytes()

        responses = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model=model,
            text=message,
            voice=voice,
            language_type="Chinese" if voice in ["Cherry", "Zhichu", "Zhixiang"] else "English",
            stream=True,
        )

        if responses is None:
            _LOGGER.error(
                "Qwen TTS returned None for message: %s model: %s voice: %s",
                message,
                model,
                voice,
            )
            raise HomeAssistantError("TTS synthesis returned None")


        for chunk in responses:
            if chunk.status_code != 200:
                _LOGGER.error(
                    "Qwen TTS error. status_code: %s request_id: %s code: %s message: %s",
                    chunk.status_code,
                    chunk.request_id,
                    chunk.code,
                    chunk.message,
                )
                raise HomeAssistantError(f"TTS synthesis error: {chunk.message}")

            if chunk.output and chunk.output.audio and chunk.output.audio.data:
                audio_string = chunk.output.audio.data
                wav_bytes: bytes = base64.b64decode(audio_string)
                audio += wav_bytes

        wav_header = create_wav_header(len(audio))
        return wav_header + audio

    @staticmethod
    def _process_cosyvoice_tts(
            model: str, voice: str, message: str, api_key: str
    ) -> bytes:
        """Process CosyVoice TTS synthesis."""
        start_time = time.perf_counter()

        dashscope.api_key = api_key
        synthesizer = dashscope.audio.tts_v2.SpeechSynthesizer(
            model=model, voice=voice
        )
        audio_data = synthesizer.call(message)

        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000

        _LOGGER.info(
            "[Metric] requestId: %s, first package delay: %s ms, elapsed: %.2f ms",
            synthesizer.get_last_request_id(),
            synthesizer.get_first_package_delay(),
            elapsed_time,
        )

        return audio_data

    async def async_get_tts_audio(
            self,
            message: str,
            language: str,
            options: dict[str, Any] | None = None,
    ) -> TtsAudioType:
        """Load TTS audio (non-streaming)."""
        try:
            config = self._get_current_config()
            api_key = config.get(CONF_TOKEN)

            if not api_key:
                raise HomeAssistantError("API Key is not configured")

            model = config.get(CONF_MODEL, "qwen3-tts-flash")
            voice = config.get(CONF_VOICE, "Cherry")

            if options and ATTR_VOICE in options:
                voice = options[ATTR_VOICE]

            _LOGGER.debug(
                "Generating TTS: model=%s, voice=%s, language=%s",
                model,
                voice,
                language,
            )

            dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

            if model.startswith("qwen"):
                audio_data = await self.hass.async_add_executor_job(
                    self._process_qwen_tts, model, voice, message, api_key
                )
                audio_format = "wav"
            elif model.startswith("cosyvoice"):
                audio_data = await self.hass.async_add_executor_job(
                    self._process_cosyvoice_tts, model, voice, message, api_key
                )
                audio_format = "mp3"
            else:
                _LOGGER.error("Unsupported model: %s", model)
                raise HomeAssistantError(f"Unsupported model: {model}")

            if not audio_data or len(audio_data) == 0:
                _LOGGER.error(
                    "TTS returned empty audio for message: %s model: %s voice: %s",
                    message,
                    model,
                    voice,
                )
                raise HomeAssistantError("TTS synthesis returned empty audio")

            return audio_format, audio_data

        except Exception as e:
            _LOGGER.error("Error generating TTS audio: %s", e, exc_info=True)
            raise HomeAssistantError(f"Error generating TTS audio: {e}")

    async def async_stream_tts_audio(self, request: TTSAudioRequest) -> TTSAudioResponse:
        config = self._get_current_config()
        api_key = config.get(CONF_TOKEN)
        model = config.get(CONF_MODEL, "qwen3-tts-flash")
        voice = request.options.get(ATTR_VOICE, config.get(CONF_VOICE, "Cherry"))

        if 'realtime' not in model:
            _LOGGER.info("Model not supported realtime stream, so sync get: %s", model)
            full_text = "".join([c async for c in request.message_gen])
            fmt, data = await self.async_get_tts_audio(full_text, request.language, request.options)

            async def gen(): yield data

            return TTSAudioResponse(fmt, gen())

        text_queue = queue.Queue()
        callback = StreamingCallback(asyncio.get_running_loop())

        worker_task = self.hass.async_add_executor_job(
            _sync_tts_worker, model, voice, api_key, callback, text_queue
        )

        async def get_async_generator() -> AsyncGenerator[bytes, None]:
            fill_task = None
            header_sent = False
            try:
                async def fill_queue():
                    try:
                        async for msg in request.message_gen:
                            _LOGGER.debug("request: %s", msg)
                            text_queue.put(msg)
                    finally:
                        _LOGGER.debug("finished message_gen")
                        text_queue.put(None)

                fill_task = asyncio.create_task(fill_queue())

                while True:
                    chunk = await callback.get_audio_chunk()
                    if chunk is None:
                        _LOGGER.debug("finished callback")
                        break
                    if callback.error:
                        raise callback.error
                    if not header_sent:
                        header = create_wav_header(data_size=0)
                        yield header
                        header_sent = True
                    yield chunk

                await fill_task
                await worker_task
            except Exception as e:
                _LOGGER.error("Stream generator exception: %s", e)
                text_queue.put(None)
            finally:
                if fill_task: fill_task.cancel()
                await self.hass.async_add_executor_job(callback.wait_for_finished)
                _LOGGER.debug("Stream generator finished")

        return TTSAudioResponse("wav", get_async_generator())
