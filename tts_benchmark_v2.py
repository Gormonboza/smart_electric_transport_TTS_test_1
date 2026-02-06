"""
TTS Benchmark V2: Оптимизированный параллельный pipeline
Vikhr 12B (streaming) + ElevenLabs (streaming)

Оптимизации:
1. Прогрев LLM перед тестами
2. Параллельный pipeline: LLM токены → буфер → TTS
3. TTS начинает после накопления ~30 символов
4. Уменьшен chunk_length_schedule для быстрого старта

Запуск:
    python tts_benchmark_v2.py
    python tts_benchmark_v2.py --interactive
    python tts_benchmark_v2.py --compare  # сравнение sequential vs parallel
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
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "y2Y5MeVPm6ZQXK64WUui"
MODEL_ID = "eleven_flash_v2_5"

OLLAMA_MODEL = "rscr/vikhr_nemo_12b:Q4_K_M"

# Оптимизация: минимальный буфер для старта TTS
TTS_MIN_CHARS = 30  # Начинаем TTS после 30 символов
TTS_CHUNK_SCHEDULE = [50, 80, 120]  # Уменьшено с [60, 100, 140]

# ═══════════════════════════════════════════════════════════════
# МЕТРИКИ
# ═══════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Результаты замера"""
    input_text: str
    generated_text: str = ""
    mode: str = "sequential"  # sequential или parallel
    
    pipeline_start: float = 0
    llm_first_token: float = 0
    llm_end: float = 0
    tts_start: float = 0
    tts_first_audio: float = 0
    tts_end: float = 0
    playback_end: float = 0
    
    @property
    def llm_ttft(self) -> float:
        return self.llm_first_token - self.pipeline_start if self.llm_first_token else 0
    
    @property
    def llm_total(self) -> float:
        return self.llm_end - self.pipeline_start if self.llm_end else 0
    
    @property
    def tts_ttfa(self) -> float:
        return self.tts_first_audio - self.tts_start if self.tts_first_audio and self.tts_start else 0
    
    @property
    def total_ttfa(self) -> float:
        """Главная метрика: время от старта до первого звука"""
        return self.tts_first_audio - self.pipeline_start if self.tts_first_audio else 0
    
    @property
    def total_time(self) -> float:
        return self.playback_end - self.pipeline_start if self.playback_end else 0
    
    def print_report(self):
        print("\n" + "═" * 60)
        print(f"📊 РЕЗУЛЬТАТЫ [{self.mode.upper()}]")
        print("═" * 60)
        print(f"📝 Вход: \"{self.input_text}\"")
        print(f"🗣️ Ответ: \"{self.generated_text}\"")
        print(f"📏 Длина: {len(self.generated_text)} символов")
        print("-" * 60)
        print(f"🧠 LLM First Token:          {self.llm_ttft*1000:>7.0f} ms")
        print(f"🧠 LLM Total:                {self.llm_total*1000:>7.0f} ms")
        print("-" * 60)
        print(f"🔊 TTS Start (от pipeline):  {(self.tts_start - self.pipeline_start)*1000:>7.0f} ms")
        print(f"🔊 TTS TTFA (от tts_start):  {self.tts_ttfa*1000:>7.0f} ms")
        print("-" * 60)
        status = "✅" if self.total_ttfa < 0.5 else "⚠️" if self.total_ttfa < 1.0 else "❌"
        print(f"⚡ TOTAL TTFA:               {self.total_ttfa*1000:>7.0f} ms  {status}")
        print(f"⏱️ TOTAL Time:               {self.total_time*1000:>7.0f} ms")
        print("═" * 60)


# ═══════════════════════════════════════════════════════════════
# AUDIO PLAYER (без изменений)
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
            self.stream.stop()
            self.stream.close()
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
                
            if len(data) < 500:
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
                        dtype="float32"
                    )
                    self.stream.start()
                    self.current_rate = samplerate
                    
                self.stream.write(chunk)
            except Exception as e:
                print(f"[⚠️ Audio error]: {e}")
                
            self.audio_queue.task_done()


# ═══════════════════════════════════════════════════════════════
# LLM STREAMING
# ═══════════════════════════════════════════════════════════════

class StreamingLLM:
    """Vikhr 12B со стримингом токенов"""
    
    SYSTEM_PROMPT = """Ты — голосовой ассистент умного дома Вилла.
Генерируй короткие (до 15 слов) ответы на команды.
Кратко, дружелюбно. Без эмодзи.

Примеры:
включить свет в гостиной → Включаю свет в гостиной!
выключить свет → Хорошо, выключаю свет.
заказать пиццу маргариту → Заказываю пиццу маргариту.
какая погода → Сейчас плюс пятнадцать, солнечно.
включи кондиционер на 22 → Устанавливаю двадцать два градуса."""

    def __init__(self):
        import ollama
        self.client = ollama
        self._warmed_up = False
        
    def warmup(self):
        """Прогрев модели"""
        if self._warmed_up:
            return
        print("🔥 Прогрев LLM...")
        start = time.time()
        # Простой запрос для загрузки модели в память
        self.client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "привет"}],
            options={"num_predict": 5}
        )
        print(f"✅ LLM прогрет за {time.time()-start:.1f}s")
        self._warmed_up = True
        
    def generate_stream(self, command: str, result: BenchmarkResult):
        """Генерация со стримингом токенов"""
        first_token = False
        full_text = ""
        
        stream = self.client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Команда: {command}"}
            ],
            options={"temperature": 0.7, "num_predict": 50},
            stream=True
        )
        
        for chunk in stream:
            token = chunk["message"]["content"]
            full_text += token
            
            if not first_token:
                result.llm_first_token = time.time()
                first_token = True
                
            yield token
            
        result.llm_end = time.time()
        result.generated_text = full_text.strip()
        
    def generate_sync(self, command: str, result: BenchmarkResult) -> str:
        """Синхронная генерация (для sequential режима)"""
        response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Команда: {command}"}
            ],
            options={"temperature": 0.7, "num_predict": 50}
        )
        result.llm_first_token = time.time()
        result.llm_end = time.time()
        result.generated_text = response["message"]["content"].strip()
        return result.generated_text


# ═══════════════════════════════════════════════════════════════
# ELEVENLABS TTS STREAMING
# ═══════════════════════════════════════════════════════════════

class ElevenLabsTTS:
    def __init__(self):
        self.api_key = ELEVENLABS_API_KEY
        self.voice_id = VOICE_ID
        self.model_id = MODEL_ID
        
    async def synthesize_text(self, text: str, result: BenchmarkResult, player: AudioPlayer):
        """Синтез готового текста"""
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id={self.model_id}"
        
        result.tts_start = time.time()
        first_audio = False
        
        async with websockets.connect(uri, ping_interval=None) as ws:
            init_msg = {
                "xi_api_key": self.api_key,
                "text": " ",
                "voice_settings": {"stability": 0.4, "similarity_boost": 0.9},
                "generation_config": {"chunk_length_schedule": TTS_CHUNK_SCHEDULE},
            }
            await ws.send(json.dumps(init_msg))
            await ws.send(json.dumps({"text": text, "try_trigger_generation": True}))
            await ws.send(json.dumps({"text": ""}))
            
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except:
                    continue
                    
                audio_b64 = data.get("audio")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    if not first_audio:
                        result.tts_first_audio = time.time()
                        first_audio = True
                    player.enqueue(audio_bytes)
                    
                if data.get("isFinal"):
                    result.tts_end = time.time()
                    break

    async def synthesize_streaming(
        self, 
        text_queue: asyncio.Queue, 
        result: BenchmarkResult, 
        player: AudioPlayer
    ):
        """
        Стриминговый синтез: получает текст по частям из очереди.
        Начинает синтез после накопления TTS_MIN_CHARS символов.
        """
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id={self.model_id}"
        
        first_audio = False
        buffer = ""
        tts_started = False
        
        async with websockets.connect(uri, ping_interval=None) as ws:
            # Инициализация WebSocket
            init_msg = {
                "xi_api_key": self.api_key,
                "text": " ",
                "voice_settings": {"stability": 0.4, "similarity_boost": 0.9},
                "generation_config": {"chunk_length_schedule": TTS_CHUNK_SCHEDULE},
            }
            await ws.send(json.dumps(init_msg))
            
            async def send_text():
                """Корутина для отправки текста в TTS"""
                nonlocal buffer, tts_started
                
                while True:
                    item = await text_queue.get()
                    
                    if item is None:  # Сигнал завершения
                        # Отправляем оставшийся буфер
                        if buffer:
                            await ws.send(json.dumps({"text": buffer, "try_trigger_generation": True}))
                        await ws.send(json.dumps({"text": ""}))  # Финализация
                        break
                        
                    buffer += item
                    
                    # Начинаем TTS после накопления минимального буфера
                    if len(buffer) >= TTS_MIN_CHARS and not tts_started:
                        result.tts_start = time.time()
                        tts_started = True
                        await ws.send(json.dumps({"text": buffer, "try_trigger_generation": True}))
                        buffer = ""
                    elif tts_started and len(buffer) >= 20:
                        # Отправляем накопленный текст порциями
                        await ws.send(json.dumps({"text": buffer, "try_trigger_generation": True}))
                        buffer = ""
            
            async def receive_audio():
                """Корутина для приёма аудио"""
                nonlocal first_audio
                
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except:
                        continue
                        
                    audio_b64 = data.get("audio")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        if not first_audio:
                            result.tts_first_audio = time.time()
                            first_audio = True
                        player.enqueue(audio_bytes)
                        
                    if data.get("isFinal"):
                        result.tts_end = time.time()
                        break
            
            # Запускаем отправку и приём параллельно
            await asyncio.gather(send_text(), receive_audio())


# ═══════════════════════════════════════════════════════════════
# BENCHMARK RUNNERS
# ═══════════════════════════════════════════════════════════════

async def run_sequential(command: str, llm: StreamingLLM, tts: ElevenLabsTTS, play_audio: bool = True) -> BenchmarkResult:
    """Последовательный режим: сначала весь LLM, потом TTS"""
    result = BenchmarkResult(input_text=command, mode="sequential")
    player = AudioPlayer()
    
    result.pipeline_start = time.time()
    
    # 1. LLM (полная генерация)
    text = llm.generate_sync(command, result)
    
    # 2. TTS
    if play_audio:
        player.start()
    await tts.synthesize_text(text, result, player)
    
    # 3. Воспроизведение
    if play_audio:
        player.wait_until_done()
        await asyncio.sleep(0.3)
        player.stop()
        
    result.playback_end = time.time()
    return result


async def run_parallel(command: str, llm: StreamingLLM, tts: ElevenLabsTTS, play_audio: bool = True) -> BenchmarkResult:
    """Параллельный режим: LLM стримит токены в TTS"""
    result = BenchmarkResult(input_text=command, mode="parallel")
    player = AudioPlayer()
    text_queue = asyncio.Queue()
    
    result.pipeline_start = time.time()
    
    async def llm_producer():
        """Генерирует токены и кладёт в очередь"""
        for token in llm.generate_stream(command, result):
            await text_queue.put(token)
        await text_queue.put(None)  # Сигнал завершения
    
    if play_audio:
        player.start()
    
    # Запускаем LLM и TTS параллельно
    await asyncio.gather(
        llm_producer(),
        tts.synthesize_streaming(text_queue, result, player)
    )
    
    # Воспроизведение
    if play_audio:
        player.wait_until_done()
        await asyncio.sleep(0.3)
        player.stop()
        
    result.playback_end = time.time()
    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

async def run_comparison():
    """Сравнение sequential vs parallel"""
    
    test_commands = [
        "включить свет в гостиной",
        "выключить свет",
        "какая погода",
        "заказать пиццу маргариту",
        "включи кондиционер на 22 градуса",
    ]
    
    print("=" * 60)
    print("🚀 TTS BENCHMARK V2: Sequential vs Parallel")
    print("=" * 60)
    
    if not ELEVENLABS_API_KEY:
        print("❌ ELEVENLABS_API_KEY не найден в .env")
        return
    
    llm = StreamingLLM()
    tts = ElevenLabsTTS()
    
    # Прогрев
    llm.warmup()
    
    seq_results = []
    par_results = []
    
    for cmd in test_commands:
        print(f"\n{'='*60}")
        print(f"🎯 Тест: \"{cmd}\"")
        print("=" * 60)
        
        # Sequential
        print("\n▶️ SEQUENTIAL режим...")
        seq_result = await run_sequential(cmd, llm, tts, play_audio=True)
        seq_result.print_report()
        seq_results.append(seq_result)
        
        await asyncio.sleep(1.0)
        
        # Parallel
        print("\n▶️ PARALLEL режим...")
        par_result = await run_parallel(cmd, llm, tts, play_audio=True)
        par_result.print_report()
        par_results.append(par_result)
        
        await asyncio.sleep(1.0)
    
    # Сводка
    print("\n" + "=" * 60)
    print("📈 СВОДКА СРАВНЕНИЯ")
    print("=" * 60)
    
    seq_ttfa = [r.total_ttfa * 1000 for r in seq_results]
    par_ttfa = [r.total_ttfa * 1000 for r in par_results]
    
    print(f"\n{'Метрика':<25} {'Sequential':>12} {'Parallel':>12} {'Улучшение':>12}")
    print("-" * 60)
    print(f"{'TTFA avg':<25} {sum(seq_ttfa)/len(seq_ttfa):>10.0f}ms {sum(par_ttfa)/len(par_ttfa):>10.0f}ms {(1 - sum(par_ttfa)/sum(seq_ttfa))*100:>10.0f}%")
    print(f"{'TTFA min':<25} {min(seq_ttfa):>10.0f}ms {min(par_ttfa):>10.0f}ms")
    print(f"{'TTFA max':<25} {max(seq_ttfa):>10.0f}ms {max(par_ttfa):>10.0f}ms")
    
    seq_success = sum(1 for t in seq_ttfa if t < 500)
    par_success = sum(1 for t in par_ttfa if t < 500)
    print(f"\n{'TTFA < 500ms':<25} {seq_success}/{len(seq_ttfa):>10} {par_success}/{len(par_ttfa):>10}")


async def run_parallel_only():
    """Только parallel режим"""
    
    test_commands = [
        "включить свет в гостиной",
        "выключить свет",
        "какая погода",
        "заказать пиццу маргариту",
        "включи кондиционер на 22 градуса",
    ]
    
    print("=" * 60)
    print("🚀 TTS BENCHMARK V2: Parallel Pipeline")
    print("=" * 60)
    
    if not ELEVENLABS_API_KEY:
        print("❌ ELEVENLABS_API_KEY не найден в .env")
        return
    
    llm = StreamingLLM()
    tts = ElevenLabsTTS()
    
    llm.warmup()
    
    results = []
    
    for cmd in test_commands:
        print(f"\n🎯 Тест: \"{cmd}\"")
        result = await run_parallel(cmd, llm, tts, play_audio=True)
        result.print_report()
        results.append(result)
        await asyncio.sleep(1.0)
    
    # Сводка
    print("\n" + "=" * 60)
    print("📈 СВОДКА (PARALLEL)")
    print("=" * 60)
    
    ttfa_values = [r.total_ttfa * 1000 for r in results]
    llm_ttft = [r.llm_ttft * 1000 for r in results]
    
    print(f"LLM First Token:    avg={sum(llm_ttft)/len(llm_ttft):.0f}ms")
    print(f"TOTAL TTFA:         avg={sum(ttfa_values)/len(ttfa_values):.0f}ms")
    print(f"                    min={min(ttfa_values):.0f}ms")
    print(f"                    max={max(ttfa_values):.0f}ms")
    
    success_rate = sum(1 for t in ttfa_values if t < 500) / len(ttfa_values) * 100
    print(f"\n✅ TTFA < 500ms:    {success_rate:.0f}% ({sum(1 for t in ttfa_values if t < 500)}/{len(ttfa_values)})")


async def interactive_mode():
    """Интерактивный режим"""
    print("=" * 60)
    print("🎙️ ИНТЕРАКТИВНЫЙ РЕЖИМ (Parallel Pipeline)")
    print("=" * 60)
    
    if not ELEVENLABS_API_KEY:
        print("❌ ELEVENLABS_API_KEY не найден в .env")
        return
    
    llm = StreamingLLM()
    tts = ElevenLabsTTS()
    
    llm.warmup()
    
    print("\nВведите команду (exit для выхода):\n")
    
    while True:
        try:
            command = input("> ").strip()
            if not command:
                continue
            if command.lower() in ["exit", "quit", "q"]:
                break
                
            result = await run_parallel(command, llm, tts, play_audio=True)
            result.print_report()
            
        except KeyboardInterrupt:
            break
            
    print("\n👋 До свидания!")


def main():
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            asyncio.run(interactive_mode())
        elif sys.argv[1] == "--compare":
            asyncio.run(run_comparison())
        else:
            print("Usage: python tts_benchmark_v2.py [--interactive|--compare]")
    else:
        asyncio.run(run_parallel_only())


if __name__ == "__main__":
    main()
