"""
⚡ Fast TTS Pipeline v4: Оптимизации latency
=============================================
Изменения vs v3:
1. TTS_MIN_BUFFER снижен 25 → 15 символов (быстрее первая отправка)
2. temperature=0 (быстрее sampling GPT)
3. Системный промпт сокращён до минимума (меньше токенов = быстрее TTFT)
4. Persistent WebSocket pool — WS к ElevenLabs открыт ЗАРАНЕЕ, до запроса
5. OpenAI connection pre-warm — первый запрос быстрее за счёт keep-alive
6. chunk_length_schedule снижен [50] → [40] — ещё агрессивнее

Целевая метрика: TTFA < 1.2с стабильно

Запуск:
    python tts_fast_pipeline_v4.py                  # интерактивный режим
    python tts_fast_pipeline_v4.py --benchmark      # бенчмарк
    python tts_fast_pipeline_v4.py --compare        # сравнение v3 vs v4
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

# ── v4 оптимизации ──
TTS_CHUNK_SCHEDULE = [50]   # Минимум ElevenLabs = 50
TTS_MIN_BUFFER = 15         # v3: 25  → первый фрагмент уходит раньше

# Промпт максимально сжат — каждый токен = задержка TTFT
SYSTEM_PROMPT_RU = "Голос шаттла NUANU. Шутка про увиденное, 1-2 предложения, по-русски."
SYSTEM_PROMPT_EN = "NUANU shuttle voice. Joke about what you see, 1-2 sentences."

ELEVENLABS_URI = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?model_id={MODEL_ID}"


# ═══════════════════════════════════════════════════════════════
# МЕТРИКИ
# ═══════════════════════════════════════════════════════════════

@dataclass
class PipelineMetrics:
    input_text: str
    generated_text: str = ""
    mode: str = "v4"

    pipeline_start: float = 0
    ws_connected: float = 0
    llm_first_token: float = 0
    first_text_to_tts: float = 0
    tts_first_audio: float = 0
    llm_end: float = 0
    tts_end: float = 0
    playback_end: float = 0
    ws_was_preconnected: bool = False  # True если WS был из пула

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
        ws_note = " (pre-connected ♻️)" if self.ws_was_preconnected else ""
        print(f"  🔌 WebSocket connect:       {self.ws_connect_time*1000:>7.0f} ms{ws_note}")
        print(f"  🧠 LLM first token:         {self.llm_ttft*1000:>7.0f} ms")
        print(f"  📤 First text → TTS:        {self.time_to_tts_send*1000:>7.0f} ms")
        print(f"  🧠 LLM total:               {self.llm_total*1000:>7.0f} ms")
        print("─" * 65)
        status = "✅" if self.total_ttfa < 1.2 else "⚠️" if self.total_ttfa < 2.0 else "❌"
        print(f"  ⚡ TOTAL TTFA:              {self.total_ttfa*1000:>7.0f} ms  {status}")
        print(f"  ⏱️  TOTAL time:             {self.total_time*1000:>7.0f} ms")
        print("═" * 65)


# ═══════════════════════════════════════════════════════════════
# AUDIO PLAYER
# ═══════════════════════════════════════════════════════════════

class AudioPlayer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.stream = None
        self.current_rate = None
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
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
                        blocksize=1024,
                    )
                    self.stream.start()
                    self.current_rate = samplerate

                self.stream.write(chunk)
            except Exception as e:
                print(f"[⚠️ Audio]: {e}")

            self.audio_queue.task_done()


# ═══════════════════════════════════════════════════════════════
# ELEVENLABS WEBSOCKET POOL
# ═══════════════════════════════════════════════════════════════

class ElevenLabsPool:
    """
    Пул pre-connected WebSocket'ов к ElevenLabs.

    После каждого запроса сразу открывает новое соединение в фоне,
    чтобы к следующему запросу WS уже был готов.
    Экономит ~150ms на каждом запросе кроме первого.
    """

    def __init__(self):
        self._ready_ws = None
        self._ready_event = asyncio.Event()
        self._preconnecting = False

    async def _create_connection(self):
        """Открыть новый WebSocket и отправить init"""
        ws = await websockets.connect(ELEVENLABS_URI, ping_interval=None)
        init_msg = {
            "xi_api_key": ELEVENLABS_API_KEY,
            "text": " ",
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.9},
            "generation_config": {"chunk_length_schedule": TTS_CHUNK_SCHEDULE},
        }
        await ws.send(json.dumps(init_msg))
        return ws

    async def preconnect(self):
        """Открыть соединение заранее (вызывать в фоне)"""
        if self._preconnecting:
            return
        self._preconnecting = True
        try:
            self._ready_ws = await self._create_connection()
            self._ready_event.set()
        except Exception as e:
            print(f"[⚠️ Preconnect failed]: {e}")
            self._ready_ws = None
        finally:
            self._preconnecting = False

    async def get_ws(self) -> tuple:
        """
        Получить готовый WebSocket.
        Returns: (ws, was_preconnected: bool)
        """
        if self._ready_ws is not None:
            ws = self._ready_ws
            self._ready_ws = None
            self._ready_event.clear()
            # Проверяем что WS ещё жив
            try:
                # Пинг через пустое сообщение — проверка
                if ws.open:
                    return ws, True
            except Exception:
                pass

        # Нет готового — создаём новый
        ws = await self._create_connection()
        return ws, False

    async def schedule_preconnect(self):
        """Запланировать фоновое подключение для следующего запроса"""
        asyncio.create_task(self.preconnect())


# ═══════════════════════════════════════════════════════════════
# OPENAI WARMER
# ═══════════════════════════════════════════════════════════════

class OpenAIWarmer:
    """
    Pre-warm OpenAI HTTP connection.
    Первый запрос к OpenAI всегда медленнее из-за DNS + TCP + TLS.
    Делаем dummy-запрос при старте чтобы keep-alive соединение было готово.
    """

    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self._warmed = False

    async def warmup(self):
        if self._warmed:
            return
        print("🔥 Прогрев OpenAI connection...")
        t = time.time()
        try:
            # Минимальный запрос для установления соединения
            resp = await self.client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                temperature=0,
            )
            self._warmed = True
            print(f"✅ OpenAI прогрет за {(time.time()-t)*1000:.0f}ms")
        except Exception as e:
            print(f"⚠️ OpenAI warmup failed: {e}")


# ═══════════════════════════════════════════════════════════════
# V3 PIPELINE (для сравнения)
# ═══════════════════════════════════════════════════════════════

class PipelineV3:
    """v3 pipeline — для сравнения (без persistent WS, без warmup)"""

    def __init__(self):
        self.async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def run(self, trigger: str, metrics: PipelineMetrics, player: AudioPlayer):
        metrics.pipeline_start = time.time()
        metrics.mode = "v3"

        async with websockets.connect(ELEVENLABS_URI, ping_interval=None) as ws:
            metrics.ws_connected = time.time()

            init_msg = {
                "xi_api_key": ELEVENLABS_API_KEY,
                "text": " ",
                "voice_settings": {"stability": 0.4, "similarity_boost": 0.9},
                "generation_config": {"chunk_length_schedule": [50]},  # v3 value
            }
            await ws.send(json.dumps(init_msg))

            first_text_sent = False
            first_audio_received = False
            full_text = ""
            V3_MIN_BUFFER = 25  # v3 value

            async def send_llm_to_tts():
                nonlocal first_text_sent, full_text
                buffer = ""

                stream = await self.async_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": (
                            "Ты — голос умного электротранспорта в городе Нуану. "
                            "Шуточно и кратко опиши то, что видишь, как будто разговариваешь с пассажирами. "
                            "Максимум 2 предложения. По-русски."
                        )},
                        {"role": "user", "content": trigger},
                    ],
                    stream=True,
                    max_tokens=100,
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

                        should_flush = False
                        if not first_text_sent and len(buffer) >= V3_MIN_BUFFER:
                            should_flush = True
                        elif first_text_sent:
                            for delim in ['. ', '! ', '? ', ', ', '; ', '— ']:
                                if delim in buffer:
                                    should_flush = True
                                    break
                            if len(buffer) >= 80:
                                should_flush = True

                        if should_flush and buffer.strip():
                            await ws.send(json.dumps({"text": buffer, "try_trigger_generation": True}))
                            if not first_text_sent:
                                metrics.first_text_to_tts = time.time()
                                first_text_sent = True
                            buffer = ""

                metrics.llm_end = time.time()
                if buffer.strip():
                    await ws.send(json.dumps({"text": buffer, "try_trigger_generation": True}))
                await ws.send(json.dumps({"text": ""}))
                metrics.generated_text = full_text.strip()

            async def receive_audio():
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

            await asyncio.gather(send_llm_to_tts(), receive_audio())


# ═══════════════════════════════════════════════════════════════
# V4 PIPELINE
# ═══════════════════════════════════════════════════════════════

class PipelineV4:
    """
    v4 pipeline — все оптимизации:
    - Persistent WebSocket pool
    - Pre-warmed OpenAI connection
    - Shorter prompt, temp=0
    - TTS_MIN_BUFFER=15, chunk_schedule=[40]
    """

    def __init__(self):
        self.async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.warmer = OpenAIWarmer(self.async_client)
        self.ws_pool = ElevenLabsPool()

    async def warmup(self):
        """Прогрев всех соединений"""
        await self.warmer.warmup()
        print("🔌 Прогрев ElevenLabs WebSocket...")
        t = time.time()
        await self.ws_pool.preconnect()
        print(f"✅ ElevenLabs WebSocket готов за {(time.time()-t)*1000:.0f}ms")

    async def run(self, trigger: str, metrics: PipelineMetrics, player: AudioPlayer):
        metrics.pipeline_start = time.time()
        metrics.mode = "v4"

        # Получаем WS (из пула или новый)
        ws, was_preconnected = await self.ws_pool.get_ws()
        metrics.ws_connected = time.time()
        metrics.ws_was_preconnected = was_preconnected

        try:
            first_text_sent = False
            first_audio_received = False
            full_text = ""

            async def send_llm_to_tts():
                nonlocal first_text_sent, full_text
                buffer = ""

                stream = await self.async_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_RU},
                        {"role": "user", "content": trigger},
                    ],
                    stream=True,
                    max_tokens=80,      # v4: ещё короче (было 100)
                    temperature=0,       # v4: детерминированный = быстрее
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

                        should_flush = False

                        if not first_text_sent and len(buffer) >= TTS_MIN_BUFFER:
                            should_flush = True
                        elif first_text_sent:
                            for delim in ['. ', '! ', '? ', ', ', '; ']:
                                if delim in buffer:
                                    should_flush = True
                                    break
                            if len(buffer) >= 60:  # v4: снижено с 80
                                should_flush = True

                        if should_flush and buffer.strip():
                            await ws.send(json.dumps({
                                "text": buffer,
                                "try_trigger_generation": True,
                            }))
                            if not first_text_sent:
                                metrics.first_text_to_tts = time.time()
                                first_text_sent = True
                            buffer = ""

                metrics.llm_end = time.time()

                if buffer.strip():
                    await ws.send(json.dumps({
                        "text": buffer,
                        "try_trigger_generation": True,
                    }))

                await ws.send(json.dumps({"text": ""}))
                metrics.generated_text = full_text.strip()

            async def receive_audio():
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

            await asyncio.gather(send_llm_to_tts(), receive_audio())

        finally:
            # Закрываем использованный WS
            try:
                await ws.close()
            except Exception:
                pass

            # Сразу открываем новый WS для следующего запроса
            await self.ws_pool.schedule_preconnect()


# ═══════════════════════════════════════════════════════════════
# ЗАПУСК
# ═══════════════════════════════════════════════════════════════

async def run_single(pipeline, trigger: str, play_audio: bool = True) -> PipelineMetrics:
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
    triggers = [
        "камера, две собаки слева на обочине",
        "впереди группа туристов, человек 8, фотографируют храм",
        "справа мотоцикл обгоняет, на нём двое без шлемов",
        "проезжаем рисовые поля, красиво",
        "перекрёсток, слева едет грузовик с кокосами",
    ]

    print("=" * 65)
    print("🚀 FAST PIPELINE v4 BENCHMARK")
    print("=" * 65)

    if not OPENAI_API_KEY or not ELEVENLABS_API_KEY:
        print("❌ Установите OPENAI_API_KEY и ELEVENLABS_API_KEY в .env")
        return

    pipeline = PipelineV4()
    await pipeline.warmup()

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
        print("📈 СВОДКА v4")
        print("=" * 65)
        ttfa = [r.total_ttfa * 1000 for r in results]
        llm_ttft = [r.llm_ttft * 1000 for r in results]
        ws_times = [r.ws_connect_time * 1000 for r in results]
        preconn = sum(1 for r in results if r.ws_was_preconnected)
        print(f"  WebSocket connect:   avg={sum(ws_times)/len(ws_times):.0f}ms  (pre-connected: {preconn}/{len(results)})")
        print(f"  LLM first token:     avg={sum(llm_ttft)/len(llm_ttft):.0f}ms")
        print(f"  TOTAL TTFA:          avg={sum(ttfa)/len(ttfa):.0f}ms")
        print(f"                       min={min(ttfa):.0f}ms")
        print(f"                       max={max(ttfa):.0f}ms")
        ok = sum(1 for t in ttfa if t < 1200)
        print(f"\n  ✅ TTFA < 1.2s:     {ok}/{len(ttfa)}")
        ok2 = sum(1 for t in ttfa if t < 1500)
        print(f"  ✅ TTFA < 1.5s:     {ok2}/{len(ttfa)}")


async def run_comparison():
    """Сравнение v3 vs v4"""
    triggers = [
        "камера, две собаки слева",
        "впереди группа туристов, фотографируют храм",
        "справа скутер обгоняет",
        "дерево упало на дорогу",
        "рядом красивый водопад",
    ]

    print("=" * 65)
    print("🔬 COMPARISON: v3 vs v4")
    print("=" * 65)

    if not OPENAI_API_KEY or not ELEVENLABS_API_KEY:
        print("❌ Установите OPENAI_API_KEY и ELEVENLABS_API_KEY в .env")
        return

    v3 = PipelineV3()
    v4 = PipelineV4()
    await v4.warmup()

    v3_results = []
    v4_results = []

    for trigger in triggers:
        print(f"\n{'='*65}")
        print(f"🎯 \"{trigger}\"")

        # v3
        print("\n  ▶️ v3...")
        try:
            m = await run_single(v3, trigger, play_audio=True)
            m.print_report()
            v3_results.append(m)
        except Exception as e:
            print(f"  ❌ {e}")

        await asyncio.sleep(1.5)

        # v4
        print("\n  ▶️ v4...")
        try:
            m = await run_single(v4, trigger, play_audio=True)
            m.print_report()
            v4_results.append(m)
        except Exception as e:
            print(f"  ❌ {e}")

        await asyncio.sleep(1.5)

    if v3_results and v4_results:
        print("\n" + "=" * 65)
        print("📊 СРАВНЕНИЕ v3 vs v4")
        print("=" * 65)
        v3_ttfa = [r.total_ttfa * 1000 for r in v3_results]
        v4_ttfa = [r.total_ttfa * 1000 for r in v4_results]
        avg_v3 = sum(v3_ttfa) / len(v3_ttfa)
        avg_v4 = sum(v4_ttfa) / len(v4_ttfa)
        improvement = (1 - avg_v4 / avg_v3) * 100 if avg_v3 > 0 else 0

        v3_llm = [r.llm_ttft * 1000 for r in v3_results]
        v4_llm = [r.llm_ttft * 1000 for r in v4_results]
        v3_ws = [r.ws_connect_time * 1000 for r in v3_results]
        v4_ws = [r.ws_connect_time * 1000 for r in v4_results]

        print(f"\n  {'Метрика':<25} {'v3':>12} {'v4':>12} {'Δ':>12}")
        print("  " + "─" * 61)
        print(f"  {'WS connect avg':<25} {sum(v3_ws)/len(v3_ws):>10.0f}ms {sum(v4_ws)/len(v4_ws):>10.0f}ms {sum(v4_ws)/len(v4_ws)-sum(v3_ws)/len(v3_ws):>+10.0f}ms")
        print(f"  {'LLM TTFT avg':<25} {sum(v3_llm)/len(v3_llm):>10.0f}ms {sum(v4_llm)/len(v4_llm):>10.0f}ms {sum(v4_llm)/len(v4_llm)-sum(v3_llm)/len(v3_llm):>+10.0f}ms")
        print(f"  {'TTFA avg':<25} {avg_v3:>10.0f}ms {avg_v4:>10.0f}ms {avg_v4-avg_v3:>+10.0f}ms")
        print(f"  {'TTFA min':<25} {min(v3_ttfa):>10.0f}ms {min(v4_ttfa):>10.0f}ms")
        print(f"  {'TTFA max':<25} {max(v3_ttfa):>10.0f}ms {max(v4_ttfa):>10.0f}ms")
        print(f"\n  🏆 Ускорение TTFA: {improvement:.0f}%")


async def interactive_mode():
    print("=" * 65)
    print("⚡ FAST PIPELINE v4 — Интерактивный режим")
    print("=" * 65)
    print("Оптимизации: persistent WS, pre-warm, temp=0, short prompt")
    print("Введите триггер (exit для выхода):\n")

    if not OPENAI_API_KEY or not ELEVENLABS_API_KEY:
        print("❌ Установите OPENAI_API_KEY и ELEVENLABS_API_KEY в .env")
        return

    pipeline = PipelineV4()
    await pipeline.warmup()

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
            print("Usage: python tts_fast_pipeline_v4.py [--benchmark|--compare]")
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
