"""
⚡ Fast TTS Pipeline v5: Groq LPU — целевой TTFA < 700ms
=========================================================
Главное изменение: замена GPT-4o-mini на Groq (Llama 3.3 70B)
Groq использует специальные LPU чипы → TTFT ~100-200ms vs GPT ~900ms

Groq API полностью совместим с OpenAI SDK — меняем только base_url и model.

Запуск:
    python tts_fast_pipeline_v5.py                    # интерактив (Groq)
    python tts_fast_pipeline_v5.py --gpt              # интерактив (GPT для сравнения)
    python tts_fast_pipeline_v5.py --benchmark         # бенчмарк Groq
    python tts_fast_pipeline_v5.py --compare           # Groq vs GPT бок о бок

Настройка:
    Добавьте в .env:
    GROQ_API_KEY=ваш_ключ_от_console.groq.com
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
from openai import AsyncOpenAI

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# ElevenLabs
VOICE_ID = "y2Y5MeVPm6ZQXK64WUui"
MODEL_ID = "eleven_flash_v2_5"
TTS_CHUNK_SCHEDULE = [50]
TTS_MIN_BUFFER = 15

# LLM провайдеры
LLM_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "label": "Groq Llama-3.3-70B",
    },
    "groq-small": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.1-8b-instant",
        "label": "Groq Llama-3.1-8B",
    },
    "gpt": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "label": "GPT-4o-mini",
    },
}

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
    mode: str = "groq"

    pipeline_start: float = 0
    ws_connected: float = 0
    llm_first_token: float = 0
    first_text_to_tts: float = 0
    tts_first_audio: float = 0
    llm_end: float = 0
    tts_end: float = 0
    playback_end: float = 0
    ws_was_preconnected: bool = False

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
        print(f"📊 [{self.mode.upper()}]")
        print("═" * 65)
        print(f"📝 Input:  \"{self.input_text}\"")
        out = self.generated_text[:80]
        if len(self.generated_text) > 80:
            out += "..."
        print(f"🗣️ Output: \"{out}\"")
        print(f"📏 Length: {len(self.generated_text)} chars")
        print("─" * 65)
        ws_note = " ♻️" if self.ws_was_preconnected else ""
        print(f"  🔌 WS connect:      {self.ws_connect_time*1000:>7.0f} ms{ws_note}")
        print(f"  🧠 LLM first token: {self.llm_ttft*1000:>7.0f} ms")
        print(f"  📤 Text → TTS:      {self.time_to_tts_send*1000:>7.0f} ms")
        print(f"  🧠 LLM total:       {self.llm_total*1000:>7.0f} ms")
        print("─" * 65)
        ttfa_ms = self.total_ttfa * 1000
        if ttfa_ms < 700:
            status = "🔥"
        elif ttfa_ms < 1200:
            status = "✅"
        elif ttfa_ms < 2000:
            status = "⚠️"
        else:
            status = "❌"
        print(f"  ⚡ TOTAL TTFA:      {ttfa_ms:>7.0f} ms  {status}")
        print(f"  ⏱️  TOTAL time:     {self.total_time*1000:>7.0f} ms")
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
    def __init__(self):
        self._ready_ws = None
        self._ready_event = asyncio.Event()
        self._preconnecting = False

    async def _create_connection(self):
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
        if self._preconnecting:
            return
        self._preconnecting = True
        try:
            self._ready_ws = await self._create_connection()
            self._ready_event.set()
        except Exception as e:
            print(f"[⚠️ Preconnect]: {e}")
            self._ready_ws = None
        finally:
            self._preconnecting = False

    async def get_ws(self) -> tuple:
        if self._ready_ws is not None:
            ws = self._ready_ws
            self._ready_ws = None
            self._ready_event.clear()
            try:
                if ws.open:
                    return ws, True
            except Exception:
                pass
        ws = await self._create_connection()
        return ws, False

    async def schedule_preconnect(self):
        asyncio.create_task(self.preconnect())


# ═══════════════════════════════════════════════════════════════
# UNIVERSAL PIPELINE — работает с любым провайдером
# ═══════════════════════════════════════════════════════════════

class Pipeline:
    """
    Универсальный пайплайн: LLM streaming → ElevenLabs streaming.
    Работает с любым OpenAI-совместимым API (GPT, Groq, etc.)
    """

    def __init__(self, provider: str = "groq"):
        if provider not in LLM_PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Use: {list(LLM_PROVIDERS.keys())}")

        self.provider = provider
        self.config = LLM_PROVIDERS[provider]
        self.label = self.config["label"]

        api_key = os.getenv(self.config["api_key_env"])
        if not api_key:
            raise ValueError(f"Set {self.config['api_key_env']} in .env")

        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config["base_url"],
        )
        self.model = self.config["model"]
        self.ws_pool = ElevenLabsPool()

    async def warmup(self):
        """Прогрев LLM + ElevenLabs"""
        # LLM warmup
        print(f"🔥 Прогрев {self.label}...")
        t = time.time()
        try:
            await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                temperature=0,
            )
            print(f"✅ {self.label} прогрет за {(time.time()-t)*1000:.0f}ms")
        except Exception as e:
            print(f"⚠️ {self.label} warmup: {e}")

        # ElevenLabs warmup
        print("🔌 Прогрев ElevenLabs...")
        t = time.time()
        await self.ws_pool.preconnect()
        print(f"✅ ElevenLabs готов за {(time.time()-t)*1000:.0f}ms")

    async def run(self, trigger: str, metrics: PipelineMetrics, player: AudioPlayer):
        metrics.pipeline_start = time.time()
        metrics.mode = f"{self.provider} ({self.model})"

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
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_RU},
                        {"role": "user", "content": trigger},
                    ],
                    stream=True,
                    max_tokens=80,
                    temperature=0,
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
                            if len(buffer) >= 60:
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
            try:
                await ws.close()
            except Exception:
                pass
            await self.ws_pool.schedule_preconnect()


# ═══════════════════════════════════════════════════════════════
# ЗАПУСК
# ═══════════════════════════════════════════════════════════════

TRIGGERS = [
    "камера, две собаки слева на обочине",
    "впереди группа туристов, фотографируют храм",
    "справа скутер обгоняет",
    "проезжаем рисовые поля, красиво",
    "перекрёсток, слева грузовик с кокосами",
]


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


async def run_benchmark(provider: str = "groq"):
    print("=" * 65)
    print(f"🚀 BENCHMARK: {LLM_PROVIDERS[provider]['label']}")
    print("=" * 65)

    pipeline = Pipeline(provider)
    await pipeline.warmup()

    results = []
    for trigger in TRIGGERS:
        print(f"\n🎯 \"{trigger}\"")
        try:
            m = await run_single(pipeline, trigger, play_audio=True)
            m.print_report()
            results.append(m)
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f"❌ {e}")
            import traceback
            traceback.print_exc()

    _print_summary(results, provider)


async def run_comparison():
    """Groq vs GPT — бок о бок"""
    print("=" * 65)
    print("🔬 COMPARISON: Groq vs GPT-4o-mini")
    print("=" * 65)

    # Проверяем ключи
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY")
    if missing:
        print(f"❌ Не хватает: {', '.join(missing)} в .env")
        return

    groq_pipe = Pipeline("groq")
    gpt_pipe = Pipeline("gpt")

    await groq_pipe.warmup()
    await gpt_pipe.warmup()

    groq_results = []
    gpt_results = []

    for trigger in TRIGGERS:
        print(f"\n{'='*65}")
        print(f"🎯 \"{trigger}\"")

        # GPT first (slower, so results feel fair)
        print(f"\n  ▶️ {gpt_pipe.label}...")
        try:
            m = await run_single(gpt_pipe, trigger, play_audio=True)
            m.print_report()
            gpt_results.append(m)
        except Exception as e:
            print(f"  ❌ {e}")

        await asyncio.sleep(1.5)

        # Groq
        print(f"\n  ▶️ {groq_pipe.label}...")
        try:
            m = await run_single(groq_pipe, trigger, play_audio=True)
            m.print_report()
            groq_results.append(m)
        except Exception as e:
            print(f"  ❌ {e}")

        await asyncio.sleep(1.5)

    # Сводка
    if groq_results and gpt_results:
        print("\n" + "=" * 65)
        print("📊 ИТОГИ: Groq vs GPT-4o-mini")
        print("=" * 65)

        g_ttfa = [r.total_ttfa * 1000 for r in groq_results]
        o_ttfa = [r.total_ttfa * 1000 for r in gpt_results]
        g_llm = [r.llm_ttft * 1000 for r in groq_results]
        o_llm = [r.llm_ttft * 1000 for r in gpt_results]
        g_ws = [r.ws_connect_time * 1000 for r in groq_results]
        o_ws = [r.ws_connect_time * 1000 for r in gpt_results]

        avg_g = sum(g_ttfa) / len(g_ttfa)
        avg_o = sum(o_ttfa) / len(o_ttfa)
        speedup = (1 - avg_g / avg_o) * 100 if avg_o > 0 else 0

        print(f"\n  {'Метрика':<25} {'GPT-4o-mini':>12} {'Groq 70B':>12} {'Δ':>12}")
        print("  " + "─" * 61)
        print(f"  {'LLM TTFT avg':<25} {sum(o_llm)/len(o_llm):>10.0f}ms {sum(g_llm)/len(g_llm):>10.0f}ms {sum(g_llm)/len(g_llm)-sum(o_llm)/len(o_llm):>+10.0f}ms")
        print(f"  {'WS connect avg':<25} {sum(o_ws)/len(o_ws):>10.0f}ms {sum(g_ws)/len(g_ws):>10.0f}ms")
        print(f"  {'TTFA avg':<25} {avg_o:>10.0f}ms {avg_g:>10.0f}ms {avg_g-avg_o:>+10.0f}ms")
        print(f"  {'TTFA min':<25} {min(o_ttfa):>10.0f}ms {min(g_ttfa):>10.0f}ms")
        print(f"  {'TTFA max':<25} {max(o_ttfa):>10.0f}ms {max(g_ttfa):>10.0f}ms")
        print(f"\n  🏆 Ускорение TTFA: {speedup:.0f}%")

        if avg_g < 700:
            print("  🔥 Groq < 700ms — ЦЕЛЬ ДОСТИГНУТА!")
        elif avg_g < 1000:
            print("  ✅ Groq < 1s — отличный результат")


def _print_summary(results, provider):
    if not results:
        return
    print("\n" + "=" * 65)
    print(f"📈 СВОДКА: {LLM_PROVIDERS[provider]['label']}")
    print("=" * 65)
    ttfa = [r.total_ttfa * 1000 for r in results]
    llm = [r.llm_ttft * 1000 for r in results]
    ws = [r.ws_connect_time * 1000 for r in results]
    print(f"  WS connect:      avg={sum(ws)/len(ws):.0f}ms")
    print(f"  LLM first token: avg={sum(llm)/len(llm):.0f}ms")
    print(f"  TOTAL TTFA:      avg={sum(ttfa)/len(ttfa):.0f}ms")
    print(f"                   min={min(ttfa):.0f}ms")
    print(f"                   max={max(ttfa):.0f}ms")
    ok7 = sum(1 for t in ttfa if t < 700)
    ok12 = sum(1 for t in ttfa if t < 1200)
    print(f"\n  🔥 TTFA < 700ms:  {ok7}/{len(ttfa)}")
    print(f"  ✅ TTFA < 1.2s:   {ok12}/{len(ttfa)}")


async def interactive_mode(provider: str = "groq"):
    cfg = LLM_PROVIDERS[provider]
    print("=" * 65)
    print(f"⚡ FAST PIPELINE v5 — {cfg['label']}")
    print("=" * 65)
    print(f"Model: {cfg['model']}")
    print("Введите триггер (exit для выхода):\n")

    pipeline = Pipeline(provider)
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
            print(f"❌ {e}")
            import traceback
            traceback.print_exc()

    print("\n👋 До свидания!")


def main():
    args = sys.argv[1:]

    if "--compare" in args:
        asyncio.run(run_comparison())
    elif "--benchmark" in args:
        provider = "groq"
        if "--gpt" in args:
            provider = "gpt"
        elif "--groq-small" in args:
            provider = "groq-small"
        asyncio.run(run_benchmark(provider))
    elif "--gpt" in args:
        asyncio.run(interactive_mode("gpt"))
    elif "--groq-small" in args:
        asyncio.run(interactive_mode("groq-small"))
    elif "--help" in args or "-h" in args:
        print("""
Usage: python tts_fast_pipeline_v5.py [options]

Modes:
  (default)         Interactive mode with Groq
  --gpt             Interactive mode with GPT-4o-mini
  --groq-small      Interactive mode with Llama-3.1-8B (fastest)
  --benchmark       Run benchmark (add --gpt or --groq-small to switch)
  --compare         Run Groq vs GPT comparison

Setup:
  Add to .env:
    GROQ_API_KEY=your_key
    OPENAI_API_KEY=your_key
    ELEVENLABS_API_KEY=your_key

  Get Groq key: https://console.groq.com/keys
""")
    else:
        asyncio.run(interactive_mode("groq"))


if __name__ == "__main__":
    main()
