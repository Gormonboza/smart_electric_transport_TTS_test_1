"""
⚡ Fast TTS Pipeline v3: GPT-4o-mini Streaming → ElevenLabs Streaming
=====================================================================
Оптимизации vs tts_test_1.py:
1. GPT-4o-mini streaming — текст идёт в TTS по мере генерации, не ждём полный ответ
2. Параллельный pipeline — WebSocket ElevenLabs открывается ДО начала генерации GPT
3. chunk_length_schedule снижен до [50] — ElevenLabs начинает синтез раньше
4. Async OpenAI client — неблокирующие вызовы
5. Sentence-based flushing — отправляем в TTS по предложениям для естественной речи
6. Pre-warmed WebSocket — соединение готово к моменту прихода первого токена

Целевая метрика: TTFA < 1.5с (было ~4с)

Запуск:
    python tts_fast_pipeline.py                  # интерактивный режим
    python tts_fast_pipeline.py --benchmark      # бенчмарк
    python tts_fast_pipeline.py --compare        # сравнение old vs new
"""

import asyncio
import websockets
import json
import base64
import sounddevice as sd
import numpy as np
import io
import soundfile as sf
import threading
import queue
import time
import os
import sys
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

VOICE_ID = "y2Y5MeVPm6ZQXK64WUui"
MODEL_ID = "eleven_flash_v2_5"
GPT_MODEL = "gpt-4o-mini"

# Оптимизация: агрессивный старт TTS
TTS_CHUNK_SCHEDULE = [50]  # Было [60, 100, 140] — теперь начинаем с 50 символов
TTS_MIN_BUFFER = 25        # Минимум символов перед первой отправкой в TTS

# Системные промпты (укороченные для скорости — меньше токенов = быстрее)
SYSTEM_PROMPT_RU = (
    "Ты — голос умного электротранспорта в городе Нуану. "
    "Шуточно и кратко опиши то, что видишь, как будто разговариваешь с пассажирами. "
    "Максимум 2 предложения. По-русски."
)

SYSTEM_PROMPT_EN = (
    "You are the voice of a smart electric shuttle in NUANU city. "
    "Briefly and humorously describe what you see, as if talking to passengers. "
    "Max 2 sentences."
)


# ═══════════════════════════════════════════════════════════════
# МЕТРИКИ
# ═══════════════════════════════════════════════════════════════

@dataclass
class PipelineMetrics:
    """Детальные метрики пайплайна"""
    input_text: str
    generated_text: str = ""
    mode: str = "parallel"

    pipeline_start: float = 0
    ws_connected: float = 0        # WebSocket ElevenLabs готов
    llm_first_token: float = 0     # Первый токен от GPT
    first_text_to_tts: float = 0   # Первый текст отправлен в TTS
    tts_first_audio: float = 0     # Первый аудио-чанк получен
    llm_end: float = 0             # GPT закончил генерацию
    tts_end: float = 0             # TTS закончил синтез
    playback_end: float = 0        # Воспроизведение закончено

    @property
    def ws_connect_time(self) -> float:
        return self.ws_connected - self.pipeline_start if self.ws_connected else 0

    @property
    def llm_ttft(self) -> float:
        return self.llm_first_token - self.pipeline_start if self.llm_first_token else 0

    @property
    def llm_total(self) -> float:
        return self.llm_end - self.pipeline_start if self.llm_end else 0

    @property
    def time_to_tts_send(self) -> float:
        return self.first_text_to_tts - self.pipeline_start if self.first_text_to_tts else 0

    @property
    def total_ttfa(self) -> float:
        """Главная метрика: от старта до первого звука из колонки"""
        return self.tts_first_audio - self.pipeline_start if self.tts_first_audio else 0

    @property
    def total_time(self) -> float:
        return self.playback_end - self.pipeline_start if self.playback_end else 0

    def print_report(self):
        print("\n" + "═" * 65)
        print(f"📊 PIPELINE METRICS [{self.mode.upper()}]")
        print("═" * 65)
        print(f"📝 Input:  \"{self.input_text}\"")
        print(f"🗣️ Output: \"{self.generated_text[:80]}{'...' if len(self.generated_text) > 80 else ''}\"")
        print(f"📏 Length: {len(self.generated_text)} chars")
        print("─" * 65)
        print(f"  🔌 WebSocket connect:       {self.ws_connect_time*1000:>7.0f} ms")
        print(f"  🧠 LLM first token:         {self.llm_ttft*1000:>7.0f} ms")
        print(f"  📤 First text → TTS:        {self.time_to_tts_send*1000:>7.0f} ms")
        print(f"  🧠 LLM total:               {self.llm_total*1000:>7.0f} ms")
        print("─" * 65)
        status = "✅" if self.total_ttfa < 1.5 else "⚠️" if self.total_ttfa < 2.5 else "❌"
        print(f"  ⚡ TOTAL TTFA:              {self.total_ttfa*1000:>7.0f} ms  {status}")
        print(f"  ⏱️  TOTAL time:             {self.total_time*1000:>7.0f} ms")
        print("═" * 65)


# ═══════════════════════════════════════════════════════════════
# AUDIO PLAYER (оптимизированный)
# ═══════════════════════════════════════════════════════════════

class AudioPlayer:
    """Потоковое воспроизведение с отслеживанием первого чанка"""

    def __init__(self):
        self.audio_queue = queue.Queue()
        self.stream = None
        self.current_rate = None
        self.running = False
        self.thread = None
        self.first_chunk_event = threading.Event()

    def start(self):
        self.running = True
        self.first_chunk_event.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.audio_queue.put(None)
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def enqueue(self, audio_bytes: bytes):
        self.audio_queue.put(audio_bytes)

    def wait_until_done(self):
        self.audio_queue.join()

    def _worker(self):
        while self.running:
            try:
                data = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if data is None:
                self.audio_queue.task_done()
                break

            if len(data) < 300:
                self.audio_queue.task_done()
                continue

            try:
                with io.BytesIO(data) as f:
                    chunk, samplerate = sf.read(f, dtype="float32")

                if self.stream is None or samplerate != self.current_rate:
                    if self.stream:
                        self.stream.stop()
                        self.stream.close()
                    self.stream = sd.OutputStream(
                        samplerate=samplerate,
                        channels=chunk.shape[1] if chunk.ndim > 1 else 1,
                        dtype="float32",
                        blocksize=1024,  # Меньший буфер = меньше задержка
                    )
                    self.stream.start()
                    self.current_rate = samplerate

                self.stream.write(chunk)

                if not self.first_chunk_event.is_set():
                    self.first_chunk_event.set()

            except Exception as e:
                print(f"[⚠️ Audio]: {e}")

            self.audio_queue.task_done()


# ═══════════════════════════════════════════════════════════════
# OLD PIPELINE (как было — для сравнения)
# ═══════════════════════════════════════════════════════════════

class OldPipeline:
    """Оригинальный последовательный пайплайн из tts_test_1.py"""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate_reply(self, trigger: str) -> str:
        """Синхронная генерация — ждём полный ответ"""
        response = self.client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_RU},
                {"role": "user", "content": trigger},
            ],
        )
        return response.choices[0].message.content.strip()

    async def run(self, trigger: str, metrics: PipelineMetrics, player: AudioPlayer):
        """Полный последовательный пайплайн"""
        metrics.pipeline_start = time.time()
        metrics.mode = "sequential"

        # Шаг 1: ждём ВЕСЬ ответ от GPT
        reply = self.generate_reply(trigger)
        metrics.generated_text = reply
        metrics.llm_first_token = time.time()
        metrics.llm_end = time.time()

        # Шаг 2: отправляем весь текст в ElevenLabs
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?model_id={MODEL_ID}"

        async with websockets.connect(uri, ping_interval=None) as ws:
            metrics.ws_connected = time.time()

            init_msg = {
                "xi_api_key": ELEVENLABS_API_KEY,
                "text": " ",
                "voice_settings": {"stability": 0.4, "similarity_boost": 0.9},
                "generation_config": {"chunk_length_schedule": [60, 100, 140]},
            }
            await ws.send(json.dumps(init_msg))

            metrics.first_text_to_tts = time.time()
            await ws.send(json.dumps({"text": reply, "try_trigger_generation": True}))
            await ws.send(json.dumps({"text": ""}))

            first_audio = False
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue

                audio_b64 = data.get("audio")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    if not first_audio:
                        metrics.tts_first_audio = time.time()
                        first_audio = True
                    player.enqueue(audio_bytes)

                if data.get("isFinal"):
                    metrics.tts_end = time.time()
                    break


# ═══════════════════════════════════════════════════════════════
# NEW PIPELINE (оптимизированный)
# ═══════════════════════════════════════════════════════════════

class FastPipeline:
    """
    Оптимизированный параллельный пайплайн:
    GPT-4o-mini streaming → буфер по предложениям → ElevenLabs WebSocket streaming

    Ключевые отличия от OldPipeline:
    1. WebSocket открывается ПЕРВЫМ (параллельно с GPT запросом)
    2. GPT стримит токены, мы накапливаем по предложениям
    3. Каждое предложение сразу уходит в ElevenLabs
    4. chunk_length_schedule = [50] для быстрого старта синтеза
    """

    def __init__(self):
        self.async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def run(self, trigger: str, metrics: PipelineMetrics, player: AudioPlayer):
        """Параллельный пайплайн"""
        metrics.pipeline_start = time.time()
        metrics.mode = "parallel"

        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?model_id={MODEL_ID}"

        async with websockets.connect(uri, ping_interval=None) as ws:
            metrics.ws_connected = time.time()

            # Инициализация ElevenLabs WebSocket
            init_msg = {
                "xi_api_key": ELEVENLABS_API_KEY,
                "text": " ",
                "voice_settings": {"stability": 0.4, "similarity_boost": 0.9},
                "generation_config": {"chunk_length_schedule": TTS_CHUNK_SCHEDULE},
            }
            await ws.send(json.dumps(init_msg))

            # Флаги
            first_text_sent = False
            first_audio_received = False
            full_text = ""

            async def send_llm_to_tts():
                """Стримит токены GPT → накапливает → отправляет в TTS по частям"""
                nonlocal first_text_sent, full_text

                buffer = ""
                sent_chars = 0

                stream = await self.async_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_RU},
                        {"role": "user", "content": trigger},
                    ],
                    stream=True,
                    max_tokens=100,  # Ограничиваем длину для скорости
                )

                first_token = False

                async for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        token = delta.content

                        if not first_token:
                            metrics.llm_first_token = time.time()
                            first_token = True

                        buffer += token
                        full_text += token

                        # Стратегия отправки:
                        # 1. Первый фрагмент — как только набрали TTS_MIN_BUFFER символов
                        # 2. Далее — по границам предложений (. ! ? , ;) или по 80+ символов
                        should_flush = False

                        if not first_text_sent and len(buffer) >= TTS_MIN_BUFFER:
                            should_flush = True
                        elif first_text_sent:
                            # Ищем границу предложения
                            for delim in ['. ', '! ', '? ', ', ', '; ', '— ']:
                                if delim in buffer:
                                    should_flush = True
                                    break
                            # Или если буфер слишком большой
                            if len(buffer) >= 80:
                                should_flush = True

                        if should_flush and buffer.strip():
                            await ws.send(json.dumps({
                                "text": buffer,
                                "try_trigger_generation": True
                            }))
                            sent_chars += len(buffer)

                            if not first_text_sent:
                                metrics.first_text_to_tts = time.time()
                                first_text_sent = True

                            buffer = ""

                metrics.llm_end = time.time()

                # Отправляем остаток
                if buffer.strip():
                    await ws.send(json.dumps({
                        "text": buffer,
                        "try_trigger_generation": True
                    }))

                # Сигнал завершения текста
                await ws.send(json.dumps({"text": ""}))

                metrics.generated_text = full_text.strip()

            async def receive_audio():
                """Получает аудио-чанки и отправляет в плеер"""
                nonlocal first_audio_received

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue

                    audio_b64 = data.get("audio")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        if not first_audio_received:
                            metrics.tts_first_audio = time.time()
                            first_audio_received = True
                        player.enqueue(audio_bytes)

                    if data.get("isFinal"):
                        metrics.tts_end = time.time()
                        break

            # Запуск параллельно: отправка текста + приём аудио
            await asyncio.gather(send_llm_to_tts(), receive_audio())


# ═══════════════════════════════════════════════════════════════
# ЗАПУСК ТЕСТОВ
# ═══════════════════════════════════════════════════════════════

async def run_single(pipeline, trigger: str, play_audio: bool = True) -> PipelineMetrics:
    """Запуск одного теста"""
    metrics = PipelineMetrics(input_text=trigger)
    player = AudioPlayer()

    if play_audio:
        player.start()

    await pipeline.run(trigger, metrics, player)

    if play_audio:
        player.wait_until_done()
        await asyncio.sleep(0.3)
        player.stop()

    metrics.playback_end = time.time()
    return metrics


async def run_benchmark():
    """Бенчмарк оптимизированного пайплайна"""
    triggers = [
        "камера, две собаки слева на обочине",
        "впереди группа туристов, человек 8, фотографируют храм",
        "справа мотоцикл обгоняет, на нём двое без шлемов",
        "проезжаем рисовые поля, красиво",
        "перекрёсток, слева едет грузовик с кокосами",
    ]

    print("=" * 65)
    print("🚀 FAST PIPELINE BENCHMARK: GPT-4o-mini → ElevenLabs")
    print("=" * 65)

    if not OPENAI_API_KEY or not ELEVENLABS_API_KEY:
        print("❌ Установите OPENAI_API_KEY и ELEVENLABS_API_KEY в .env")
        return

    pipeline = FastPipeline()
    results = []

    for trigger in triggers:
        print(f"\n🎯 \"{trigger}\"")
        try:
            metrics = await run_single(pipeline, trigger, play_audio=True)
            metrics.print_report()
            results.append(metrics)
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print("\n" + "=" * 65)
        print("📈 СВОДКА")
        print("=" * 65)
        ttfa = [r.total_ttfa * 1000 for r in results]
        llm_ttft = [r.llm_ttft * 1000 for r in results]
        ws_times = [r.ws_connect_time * 1000 for r in results]
        print(f"  WebSocket connect:   avg={sum(ws_times)/len(ws_times):.0f}ms")
        print(f"  LLM first token:     avg={sum(llm_ttft)/len(llm_ttft):.0f}ms")
        print(f"  TOTAL TTFA:          avg={sum(ttfa)/len(ttfa):.0f}ms")
        print(f"                       min={min(ttfa):.0f}ms")
        print(f"                       max={max(ttfa):.0f}ms")
        ok = sum(1 for t in ttfa if t < 1500)
        print(f"\n  ✅ TTFA < 1.5s:     {ok}/{len(ttfa)}")
        ok2 = sum(1 for t in ttfa if t < 2000)
        print(f"  ✅ TTFA < 2.0s:     {ok2}/{len(ttfa)}")


async def run_comparison():
    """Сравнение старого и нового пайплайна"""
    triggers = [
        "камера, две собаки слева",
        "впереди группа туристов, фотографируют храм",
        "справа скутер обгоняет",
    ]

    print("=" * 65)
    print("🔬 COMPARISON: Sequential (old) vs Parallel (new)")
    print("=" * 65)

    if not OPENAI_API_KEY or not ELEVENLABS_API_KEY:
        print("❌ Установите OPENAI_API_KEY и ELEVENLABS_API_KEY в .env")
        return

    old = OldPipeline()
    new = FastPipeline()

    old_results = []
    new_results = []

    for trigger in triggers:
        print(f"\n{'='*65}")
        print(f"🎯 \"{trigger}\"")

        # Old
        print("\n  ▶️ SEQUENTIAL (old)...")
        try:
            m_old = await run_single(old, trigger, play_audio=True)
            m_old.print_report()
            old_results.append(m_old)
        except Exception as e:
            print(f"  ❌ {e}")

        await asyncio.sleep(1.5)

        # New
        print("\n  ▶️ PARALLEL (new)...")
        try:
            m_new = await run_single(new, trigger, play_audio=True)
            m_new.print_report()
            new_results.append(m_new)
        except Exception as e:
            print(f"  ❌ {e}")

        await asyncio.sleep(1.5)

    # Сводка
    if old_results and new_results:
        print("\n" + "=" * 65)
        print("📊 СРАВНЕНИЕ")
        print("=" * 65)
        old_ttfa = [r.total_ttfa * 1000 for r in old_results]
        new_ttfa = [r.total_ttfa * 1000 for r in new_results]
        avg_old = sum(old_ttfa) / len(old_ttfa)
        avg_new = sum(new_ttfa) / len(new_ttfa)
        improvement = (1 - avg_new / avg_old) * 100 if avg_old > 0 else 0

        print(f"\n  {'Метрика':<25} {'Old (seq)':>12} {'New (par)':>12} {'Speedup':>12}")
        print("  " + "─" * 61)
        print(f"  {'TTFA avg':<25} {avg_old:>10.0f}ms {avg_new:>10.0f}ms {improvement:>10.0f}%")
        print(f"  {'TTFA min':<25} {min(old_ttfa):>10.0f}ms {min(new_ttfa):>10.0f}ms")
        print(f"  {'TTFA max':<25} {max(old_ttfa):>10.0f}ms {max(new_ttfa):>10.0f}ms")
        print(f"\n  🏆 Ускорение: {improvement:.0f}%")


async def interactive_mode():
    """Интерактивный режим с быстрым пайплайном"""
    print("=" * 65)
    print("⚡ FAST PIPELINE — Интерактивный режим")
    print("=" * 65)
    print("GPT-4o-mini streaming → ElevenLabs streaming")
    print("Введите триггер (exit для выхода):\n")

    if not OPENAI_API_KEY or not ELEVENLABS_API_KEY:
        print("❌ Установите OPENAI_API_KEY и ELEVENLABS_API_KEY в .env")
        return

    pipeline = FastPipeline()

    while True:
        try:
            trigger = input("> ").strip()
            if not trigger:
                continue
            if trigger.lower() in ["exit", "quit", "q"]:
                break

            metrics = await run_single(pipeline, trigger, play_audio=True)
            metrics.print_report()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

    print("\n👋 До свидания!")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--benchmark":
            asyncio.run(run_benchmark())
        elif sys.argv[1] == "--compare":
            asyncio.run(run_comparison())
        else:
            print("Usage: python tts_fast_pipeline.py [--benchmark|--compare]")
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
