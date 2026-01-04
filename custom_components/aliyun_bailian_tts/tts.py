"""Aliyun BaiLian TTS platform using TextToSpeechEntity."""
import asyncio
import base64
import functools
import logging
import queue
import struct
import threading
import time
from types import MappingProxyType
from typing import Any, AsyncGenerator

import dashscope
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat
from dashscope.audio.tts import ResultCallback as v1CallBack
from dashscope.audio.tts import SpeechSynthesisResult
from dashscope.audio.tts_v2 import ResultCallback as v2CallBack
from homeassistant.components.tts import (
    TextToSpeechEntity,
    TtsAudioType,
    ATTR_VOICE, TTSAudioRequest, TTSAudioResponse,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_TOKEN, CONF_MODEL, CONF_VOICE, DOMAIN, CONF_NAME

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Aliyun BaiLian TTS entity."""
    async_add_entities([AliyunBaiLianTTSEntity(hass, config_entry)])


class CosyVoiceStreamingCallback(v2CallBack):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.audio_queue = asyncio.Queue()
        self.complete_event = threading.Event()
        self.error = None

    def on_open(self) -> None:
        _LOGGER.debug("CosyVoice Streaming connection opened")

    def on_complete(self) -> None:
        _LOGGER.debug("CosyVoice Streaming complete")
        self.complete_event.set()

    def on_error(self, message) -> None:
        _LOGGER.error("CosyVoice Streaming error: %s", message)
        self.error = HomeAssistantError(message)
        self.complete_event.set()

    def on_close(self) -> None:
        _LOGGER.debug("CosyVoice Streaming closed")
        self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, None)
        self.complete_event.set()

    def on_event(self, message: str) -> None:
        _LOGGER.debug("CosyVoice Streaming event: %s", message)

    def on_data(self, data: bytes) -> None:
        _LOGGER.debug("CosyVoice Streaming received data chunk: %d bytes", len(data))
        self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, data)

    async def get_audio_chunk(self) -> bytes | None:
        return await self.audio_queue.get()

    def wait_for_finished(self):
        if not self.complete_event.wait(timeout=120):
            _LOGGER.warning("CosyVoice Wait for session finish timed out")


class SambertStreamingCallback(v1CallBack):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.audio_queue = asyncio.Queue()
        self.complete_event = threading.Event()
        self.error = None

    def on_open(self) -> None:
        _LOGGER.debug("Sambert Streaming connection opened")

    def on_complete(self) -> None:
        _LOGGER.debug("Sambert Streaming complete")
        self.complete_event.set()

    def on_error(self, response: SpeechSynthesisResponse) -> None:
        _LOGGER.error("Sambert Streaming error response: %s", response)
        self.error = HomeAssistantError(response)
        self.complete_event.set()

    def on_close(self) -> None:
        _LOGGER.debug("Sambert Streaming closed")
        self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, None)
        self.complete_event.set()

    def on_event(self, result: SpeechSynthesisResult) -> None:
        _LOGGER.debug("Sambert Streaming event received")
        self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, result.get_audio_frame())

    async def get_audio_chunk(self) -> bytes | None:
        return await self.audio_queue.get()

    def wait_for_finished(self):
        if not self.complete_event.wait(timeout=120):
            _LOGGER.warning("Sambert Wait for session finish timed out")


class QwenStreamingCallback(QwenTtsRealtimeCallback):
    """Callback handler for streaming TTS (Thread-Safe)."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.audio_queue = asyncio.Queue()
        self.complete_event = threading.Event()
        self.error = None

    def on_open(self) -> None:
        _LOGGER.debug("Qwen Streaming connection opened")

    def on_close(self, close_status_code, close_msg) -> None:
        _LOGGER.debug("Qwen Streaming connection closed: code=%s, msg=%s", close_status_code, close_msg)
        self.complete_event.set()

    def on_event(self, response: dict) -> None:
        """Handle streaming events from SDK thread."""
        try:
            event_type = response.get("type")

            if event_type == "response.audio.delta":
                audio_chunk = base64.b64decode(response["delta"])
                _LOGGER.debug("Qwen Streaming received audio chunk: %d bytes", len(audio_chunk))
                self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, audio_chunk)
            elif event_type == "session.created":
                _LOGGER.debug("Qwen Session created: %s", response.get('session', {}).get('id'))
            elif event_type == "response.done":
                _LOGGER.debug("Qwen Synthesis response done")
                self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, None)
            elif event_type == "session.finished":
                _LOGGER.debug("Qwen Session finished")
                self.complete_event.set()
            else:
                _LOGGER.debug("Qwen Streaming unhandled event: %s", event_type)

        except Exception as e:
            _LOGGER.error("Error in Qwen callback processing: %s", e, exc_info=True)
            self.error = e
            self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, None)
            self.complete_event.set()

    async def get_audio_chunk(self) -> bytes | None:
        return await self.audio_queue.get()

    def wait_for_finished(self):
        if not self.complete_event.wait(timeout=120):
            _LOGGER.warning("Qwen Wait for session finish timed out")


class AliyunBaiLianTTSEntity(TextToSpeechEntity):
    """Aliyun BaiLian TTS entity."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize TTS entity."""
        self.hass = hass
        self._config_entry = config_entry

        # 获取配置的名称
        config = self._get_current_config()
        entity_name = config.get(CONF_NAME, config_entry.title)

        self._attr_name = entity_name
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

    def _get_current_config(self) -> MappingProxyType[str, Any]:
        """Get current configuration, preferring options over data."""
        if self._config_entry.options:
            return self._config_entry.options
        return self._config_entry.data

    @staticmethod
    def _create_wav_header(
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

    @staticmethod
    def _sync_tts_worker(model: str,
                         voice: str,
                         api_key: str,
                         callback: QwenStreamingCallback,
                         text_queue: queue.Queue):
        """Runs in executor thread."""
        _LOGGER.debug("Starting _sync_tts_worker. Model: %s, Voice: %s", model, voice)
        if 'realtime' not in model:
            message = ''
            while True:
                text_chunk = text_queue.get()
                if text_chunk is None: break
                message += text_chunk

            _LOGGER.debug("Processing non-realtime Qwen TTS. Message: %s", message)

            responses = dashscope.MultiModalConversation.call(
                api_key=api_key,
                model=model,
                text=message,
                voice=voice,
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

            chunk_count = 0
            for chunk in responses:
                chunk_count += 1
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
                    callback.loop.call_soon_threadsafe(callback.audio_queue.put_nowait, wav_bytes)

            _LOGGER.debug("Non-realtime Qwen TTS complete. Processed %d chunks.", chunk_count)
            callback.loop.call_soon_threadsafe(callback.audio_queue.put_nowait, None)
            callback.complete_event.set()

        else:
            _LOGGER.debug("Initializing Qwen Realtime TTS connection")
            dashscope.api_key = api_key
            qwen_tts = QwenTtsRealtime(
                model=model,
                callback=callback,
                url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
            )
            try:
                qwen_tts.connect()
                _LOGGER.debug("Qwen Realtime TTS connected. Updating session...")
                qwen_tts.update_session(
                    voice=voice,
                    response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                    mode="server_commit",
                )
                _LOGGER.debug("Session updated. Waiting for text chunks...")
                while True:
                    text_chunk = text_queue.get()
                    if text_chunk is None:
                        _LOGGER.debug("Received None chunk, stopping text input.")
                        break
                    _LOGGER.debug("Appending text chunk to Qwen Realtime: %s...",
                                  text_chunk[:50] if len(text_chunk) > 50 else text_chunk)
                    qwen_tts.append_text(text_chunk)

                _LOGGER.debug("Finishing Qwen Realtime TTS session")
                qwen_tts.finish()
            except Exception as e:
                _LOGGER.error("Worker Error in Qwen Realtime TTS: %s", e, exc_info=True)
                callback.error = e
                callback.loop.call_soon_threadsafe(callback.audio_queue.put_nowait, None)
            finally:
                callback.wait_for_finished()
                qwen_tts.close()
        _LOGGER.debug("_sync_tts_worker finished")

    @staticmethod
    def _process_qwen_tts(model: str, voice: str, message: str, api_key: str) -> bytes:
        """Process Qwen TTS synthesis."""
        _LOGGER.debug("Starting Qwen TTS processing. Model: %s, Voice: %s, Message: %s", model, voice,
                      message)
        audio: bytes = bytes()

        responses = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model=model,
            text=message,
            voice=voice,
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

        chunk_count = 0
        for chunk in responses:
            chunk_count += 1
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

        _LOGGER.debug("Qwen TTS processing complete. Received %d chunks. Total audio size: %d bytes", chunk_count,
                      len(audio))
        wav_header = AliyunBaiLianTTSEntity._create_wav_header(len(audio))
        return wav_header + audio

    @staticmethod
    def _process_cosyvoice_tts(
            model: str, voice: str, message: str, api_key: str
    ) -> bytes:
        """Process CosyVoice TTS synthesis."""
        _LOGGER.debug("Starting CosyVoice TTS processing. Model: %s, Voice: %s, Message: %s", model, voice,
                      message)
        start_time = time.perf_counter()

        dashscope.api_key = api_key
        synthesizer = dashscope.audio.tts_v2.SpeechSynthesizer(
            model=model, voice=voice
        )
        audio_data = synthesizer.call(message)

        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000

        _LOGGER.info(
            "[Metric] requestId: %s, first package delay: %s ms, elapsed: %.2f ms, audio size: %d bytes",
            synthesizer.get_last_request_id(),
            synthesizer.get_first_package_delay(),
            elapsed_time,
            len(audio_data) if audio_data else 0
        )

        return audio_data

    @staticmethod
    def _process_sambert_tts(
            model: str, voice: str, message: str, api_key: str
    ) -> bytes:
        """Process sambert TTS synthesis."""
        _LOGGER.debug("Starting Sambert TTS processing. Model: %s, Voice: %s, Message: %s", model, voice,
                      message)
        start_time = time.perf_counter()
        dashscope.api_key = api_key
        synthesizer = dashscope.audio.tts.SpeechSynthesizer()
        result: SpeechSynthesisResult = synthesizer.call(model=model, text=message, format="mp3")
        if result is None:
            _LOGGER.error("Sambert TTS call returned None result")
            raise HomeAssistantError("call not return result")
        audio_data = result.get_audio_data()

        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000

        _LOGGER.debug("response: %s", result.get_response())

        _LOGGER.info(
            "[Metric] requestId: %s, first package delay: %s ms, elapsed: %.2f ms, audio size: %d bytes",
            synthesizer.get_last_request_id(),
            synthesizer.get_first_package_delay(),
            elapsed_time,
            len(audio_data) if audio_data else 0
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
                _LOGGER.error("API Key is not configured")
                raise HomeAssistantError("API Key is not configured")

            model = config.get(CONF_MODEL, "qwen3-tts-flash")
            voice = config.get(CONF_VOICE, "Cherry")

            if options and ATTR_VOICE in options:
                voice = options[ATTR_VOICE]

            _LOGGER.debug(
                "Generating TTS audio. Model: %s, Voice: %s, Language: %s, Message: %s",
                model,
                voice,
                language,
                message
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
            elif model.startswith("sambert"):
                audio_data = await self.hass.async_add_executor_job(
                    self._process_sambert_tts, model, voice, message, api_key
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

            _LOGGER.debug("TTS audio generated successfully. Format: %s, Size: %d bytes", audio_format, len(audio_data))

            return audio_format, audio_data

        except Exception as e:
            _LOGGER.error("Error generating TTS audio: %s", e, exc_info=True)
            raise HomeAssistantError(f"Error generating TTS audio: {e}")

    async def async_stream_tts_audio(self, request: TTSAudioRequest) -> TTSAudioResponse:
        config = self._get_current_config()
        api_key = config.get(CONF_TOKEN)
        model = config.get(CONF_MODEL, "qwen3-tts-flash")
        voice = request.options.get(ATTR_VOICE, config.get(CONF_VOICE, "Cherry"))

        _LOGGER.debug("Starting async_stream_tts_audio. Model: %s, Voice: %s", model, voice)

        if model.startswith("cosyvoice"):
            full_text = "".join([c async for c in request.message_gen])
            _LOGGER.debug("CosyVoice streaming request text: %s", full_text)

            dashscope.api_key = api_key
            cosy_voice_streaming_callback = CosyVoiceStreamingCallback(asyncio.get_running_loop())

            def cosy_voice_sync():
                synthesizer = dashscope.audio.tts_v2.SpeechSynthesizer(model=model, voice=voice,
                                                                       callback=cosy_voice_streaming_callback)
                synthesizer.call(full_text)

            worker_task = self.hass.async_add_executor_job(cosy_voice_sync)

            async def gen():
                try:
                    _LOGGER.debug("CosyVoice generator started")
                    while True:
                        chunk = await cosy_voice_streaming_callback.get_audio_chunk()
                        if chunk is None:
                            _LOGGER.debug("CosyVoice generator received None (finished)")
                            break
                        if cosy_voice_streaming_callback.error:
                            _LOGGER.error("CosyVoice generator encountered error: %s",
                                          cosy_voice_streaming_callback.error)
                            raise cosy_voice_streaming_callback.error
                        yield chunk
                    await worker_task
                except Exception as e:
                    _LOGGER.error("CosyVoice Stream generator exception: %s", e, exc_info=True)
                finally:
                    await self.hass.async_add_executor_job(cosy_voice_streaming_callback.wait_for_finished)
                    _LOGGER.debug("CosyVoice Stream generator finished")

            return TTSAudioResponse("mp3", gen())
        elif model.startswith("sambert"):
            full_text = "".join([c async for c in request.message_gen])
            _LOGGER.debug("Sambert streaming request text: %s", full_text)

            sambert_streaming_callback = SambertStreamingCallback(asyncio.get_running_loop())
            dashscope.api_key = api_key

            worker_task = self.hass.async_add_executor_job(functools.partial(
                dashscope.audio.tts.SpeechSynthesizer.call,
                model,
                full_text,
                sambert_streaming_callback,
                format='mp3'
            ))

            async def gen():
                try:
                    _LOGGER.debug("Sambert generator started")
                    while True:
                        chunk = await sambert_streaming_callback.get_audio_chunk()
                        if chunk is None:
                            _LOGGER.debug("Sambert generator received None (finished)")
                            break
                        if sambert_streaming_callback.error:
                            _LOGGER.error("Sambert generator encountered error: %s", sambert_streaming_callback.error)
                            raise sambert_streaming_callback.error
                        yield chunk
                    await worker_task
                except Exception as e:
                    _LOGGER.error("Sambert Stream generator exception: %s", e, exc_info=True)
                finally:
                    await self.hass.async_add_executor_job(sambert_streaming_callback.wait_for_finished)
                    _LOGGER.debug("Sambert Stream generator finished")

            return TTSAudioResponse("mp3", gen())
        elif model.startswith("qwen"):
            text_queue = queue.Queue()
            callback = QwenStreamingCallback(asyncio.get_running_loop())

            worker_task = self.hass.async_add_executor_job(
                self._sync_tts_worker, model, voice, api_key, callback, text_queue
            )

            async def get_async_generator() -> AsyncGenerator[bytes, None]:
                fill_task = None
                header_sent = False
                try:
                    _LOGGER.debug("Qwen generator started")

                    async def fill_queue():
                        try:
                            async for msg in request.message_gen:
                                _LOGGER.debug("Qwen request received text chunk: %s...",
                                              msg[:50] if len(msg) > 50 else msg)
                                text_queue.put(msg)
                        finally:
                            _LOGGER.debug("finished message_gen (fill_queue)")
                            text_queue.put(None)

                    fill_task = asyncio.create_task(fill_queue())

                    while True:
                        chunk = await callback.get_audio_chunk()
                        if chunk is None:
                            _LOGGER.debug("Qwen generator received None (finished)")
                            break
                        if callback.error:
                            _LOGGER.error("Qwen generator encountered error: %s", callback.error)
                            raise callback.error
                        if not header_sent:
                            header = AliyunBaiLianTTSEntity._create_wav_header(data_size=0)
                            yield header
                            header_sent = True
                        yield chunk

                    await fill_task
                    await worker_task
                except Exception as e:
                    _LOGGER.error("Qwen Stream generator exception: %s", e, exc_info=True)
                    text_queue.put(None)
                finally:
                    if fill_task: fill_task.cancel()
                    await self.hass.async_add_executor_job(callback.wait_for_finished)
                    _LOGGER.debug("Qwen Stream generator finished")

            return TTSAudioResponse("wav", get_async_generator())
        else:
            _LOGGER.error("Unsupported streaming model: %s", model)
            raise HomeAssistantError(f"Unsupported model: {model}")
