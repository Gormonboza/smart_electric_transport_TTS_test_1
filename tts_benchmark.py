"""
TTS Benchmark: Vikhr 12B + ElevenLabs Streaming
Измерение TTFA (Time-to-First-Audio) и общей latency

Требования:
    pip install websockets sounddevice soundfile ollama python-dotenv

Настройка:
    Создайте .env файл с ELEVENLABS_API_KEY=ваш_ключ
    
Запуск:
    python tts_benchmark.py
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
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "y2Y5MeVPm6ZQXK64WUui"  # Русский голос
MODEL_ID = "eleven_flash_v2_5"      # Быстрая модель

# Ollama
OLLAMA_MODEL = "rscr/vikhr_nemo_12b:Q4_K_M"
OLLAMA_URL = "http://localhost:11434"

# ═══════════════════════════════════════════════════════════════
# МЕТРИКИ
# ═══════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Результаты замера одного запроса"""
    input_text: str
    generated_text: str
    
    # Времена (в секундах)
    llm_start: float = 0
    llm_first_token: float = 0
    llm_end: float = 0
    tts_start: float = 0
    tts_first_audio: float = 0
    tts_end: float = 0
    playback_end: float = 0
    
    @property
    def llm_ttft(self) -> float:
        """LLM Time-to-First-Token"""
        return self.llm_first_token - self.llm_start
    
    @property
    def llm_total(self) -> float:
        """LLM полное время генерации"""
        return self.llm_end - self.llm_start
    
    @property
    def tts_ttfa(self) -> float:
        """TTS Time-to-First-Audio"""
        return self.tts_first_audio - self.tts_start
    
    @property
    def tts_total(self) -> float:
        """TTS полное время"""
        return self.tts_end - self.tts_start
    
    @property
    def total_ttfa(self) -> float:
        """Общий TTFA (от начала до первого звука)"""
        return self.tts_first_audio - self.llm_start
    
    @property
    def total_time(self) -> float:
        """Общее время от начала до конца воспроизведения"""
        return self.playback_end - self.llm_start
    
    def print_report(self):
        """Вывод отчёта"""
        print("\n" + "═" * 60)
        print("📊 РЕЗУЛЬТАТЫ БЕНЧМАРКА")
        print("═" * 60)
        print(f"📝 Вход: \"{self.input_text}\"")
        print(f"🗣️ Ответ: \"{self.generated_text}\"")
        print(f"📏 Длина ответа: {len(self.generated_text)} символов")
        print("-" * 60)
        print(f"🧠 LLM Time-to-First-Token:  {self.llm_ttft*1000:>7.0f} ms")
        print(f"🧠 LLM Total Generation:     {self.llm_total*1000:>7.0f} ms")
        print("-" * 60)
        print(f"🔊 TTS Time-to-First-Audio:  {self.tts_ttfa*1000:>7.0f} ms")
        print(f"🔊 TTS Total Synthesis:      {self.tts_total*1000:>7.0f} ms")
        print("-" * 60)
        print(f"⚡ TOTAL TTFA:               {self.total_ttfa*1000:>7.0f} ms  {'✅' if self.total_ttfa < 0.5 else '⚠️'}")
        print(f"⏱️ TOTAL Time:               {self.total_time*1000:>7.0f} ms")
        print("═" * 60)


# ═══════════════════════════════════════════════════════════════
# AUDIO PLAYER
# ═══════════════════════════════════════════════════════════════

class AudioPlayer:
    """Потоковое воспроизведение аудио"""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.stream = None
        self.current_rate = None
        self.running = False
        self.thread = None
        self.first_chunk_played = threading.Event()
        
    def start(self):
        """Запуск фонового потока"""
        self.running = True
        self.first_chunk_played.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Остановка воспроизведения"""
        self.running = False
        self.audio_queue.put(None)  # Сигнал остановки
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
    def enqueue(self, audio_bytes: bytes):
        """Добавить аудио чанк в очередь"""
        self.audio_queue.put(audio_bytes)
        
    def wait_until_done(self):
        """Ждать завершения воспроизведения"""
        self.audio_queue.join()
        
    def _worker(self):
        """Фоновый поток воспроизведения"""
        while self.running:
            try:
                data = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            if data is None:
                self.audio_queue.task_done()
                break
                
            # Пропускаем мелкие чанки
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
                
                # Сигнал что первый чанк воспроизведён
                if not self.first_chunk_played.is_set():
                    self.first_chunk_played.set()
                    
            except Exception as e:
                print(f"[⚠️ Ошибка аудио]: {e}")
                
            self.audio_queue.task_done()


# ═══════════════════════════════════════════════════════════════
# LLM RESPONSE GENERATOR (Vikhr 12B)
# ═══════════════════════════════════════════════════════════════

class ResponseGenerator:
    """Генератор ответов через Vikhr 12B"""
    
    SYSTEM_PROMPT = """Ты — дружелюбный голосовой ассистент умного дома по имени Вилла.
Генерируй короткие (до 15 слов) естественные ответы на команды.
Отвечай кратко, по делу, дружелюбно. Не используй эмодзи.

Примеры:
Команда: включить свет в гостиной → "Включаю свет в гостиной!"
Команда: выключить свет → "Хорошо, выключаю свет."
Команда: заказать пиццу маргариту → "Заказываю пиццу маргариту. Что-нибудь ещё?"
Команда: какая погода → "Сейчас плюс пятнадцать, солнечно."
Команда: включи кондиционер на 22 градуса → "Устанавливаю двадцать два градуса."
"""

    def __init__(self):
        try:
            import ollama
            self.client = ollama
        except ImportError:
            raise ImportError("Установите ollama: pip install ollama")
    
    def generate(self, command: str, result: BenchmarkResult) -> str:
        """Генерация ответа (без стриминга, для простоты бенчмарка)"""
        result.llm_start = time.time()
        
        response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Команда: {command}"}
            ],
            options={"temperature": 0.7, "num_predict": 50}
        )
        
        result.llm_first_token = time.time()  # Для non-streaming это то же что и end
        result.llm_end = time.time()
        
        return response["message"]["content"].strip()
    
    def generate_streaming(self, command: str, result: BenchmarkResult):
        """Генерация ответа со стримингом (возвращает генератор токенов)"""
        result.llm_start = time.time()
        first_token = False
        
        stream = self.client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Команда: {command}"}
            ],
            options={"temperature": 0.7, "num_predict": 50},
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            token = chunk["message"]["content"]
            full_response += token
            
            if not first_token:
                result.llm_first_token = time.time()
                first_token = True
                
            yield token
            
        result.llm_end = time.time()
        result.generated_text = full_response.strip()


# ═══════════════════════════════════════════════════════════════
# ELEVENLABS TTS STREAMING
# ═══════════════════════════════════════════════════════════════

class ElevenLabsTTS:
    """ElevenLabs WebSocket TTS"""
    
    def __init__(self, api_key: str, voice_id: str, model_id: str):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        
    async def synthesize(self, text: str, result: BenchmarkResult, player: AudioPlayer):
        """Синтез речи через WebSocket"""
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id={self.model_id}"
        
        result.tts_start = time.time()
        first_audio = False
        
        async with websockets.connect(uri, ping_interval=None) as ws:
            # Инициализация
            init_msg = {
                "xi_api_key": self.api_key,
                "text": " ",
                "voice_settings": {"stability": 0.4, "similarity_boost": 0.9},
                "generation_config": {"chunk_length_schedule": [60, 100, 140]},
            }
            await ws.send(json.dumps(init_msg))
            
            # Отправка текста
            await ws.send(json.dumps({"text": text, "try_trigger_generation": True}))
            await ws.send(json.dumps({"text": ""}))  # Сигнал завершения
            
            # Приём аудио чанков
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except:
                    continue
                    
                audio_b64 = data.get("audio")
                if audio_b64:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        
                        if not first_audio:
                            result.tts_first_audio = time.time()
                            first_audio = True
                            print(f"🔊 Первый аудио чанк получен!")
                            
                        player.enqueue(audio_bytes)
                        
                    except Exception as e:
                        print(f"[⚠️ Ошибка декодирования]: {e}")
                        
                if data.get("isFinal"):
                    result.tts_end = time.time()
                    break


# ═══════════════════════════════════════════════════════════════
# БЕНЧМАРК
# ═══════════════════════════════════════════════════════════════

async def run_benchmark(command: str, play_audio: bool = True) -> BenchmarkResult:
    """Запуск одного теста"""
    result = BenchmarkResult(input_text=command, generated_text="")
    
    # Инициализация
    generator = ResponseGenerator()
    tts = ElevenLabsTTS(ELEVENLABS_API_KEY, VOICE_ID, MODEL_ID)
    player = AudioPlayer()
    
    print(f"\n🎯 Тест: \"{command}\"")
    print("-" * 40)
    
    # 1. Генерация текста
    print("🧠 Генерация ответа через Vikhr 12B...")
    response_text = generator.generate(command, result)
    result.generated_text = response_text
    print(f"   → \"{response_text}\" ({result.llm_total*1000:.0f}ms)")
    
    # 2. TTS
    print("🔊 Синтез речи через ElevenLabs...")
    if play_audio:
        player.start()
        
    await tts.synthesize(response_text, result, player)
    
    # 3. Ждём завершения воспроизведения
    if play_audio:
        print("🎧 Воспроизведение...")
        player.wait_until_done()
        # Небольшая пауза для завершения
        await asyncio.sleep(0.5)
        player.stop()
        
    result.playback_end = time.time()
    
    return result


async def run_full_benchmark():
    """Полный бенчмарк с несколькими тестами"""
    
    test_commands = [
        "включить свет в гостиной",
        "выключить свет",
        "какая погода",
        "заказать пиццу маргариту",
        "включи кондиционер на 22 градуса",
    ]
    
    print("=" * 60)
    print("🚀 TTS BENCHMARK: Vikhr 12B + ElevenLabs")
    print("=" * 60)
    
    if not ELEVENLABS_API_KEY:
        print("❌ ELEVENLABS_API_KEY не найден в .env")
        return
        
    results = []
    
    for cmd in test_commands:
        try:
            result = await run_benchmark(cmd, play_audio=True)
            result.print_report()
            results.append(result)
            
            # Пауза между тестами
            await asyncio.sleep(1.0)
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    
    # Сводка
    if results:
        print("\n" + "=" * 60)
        print("📈 СВОДКА")
        print("=" * 60)
        
        ttfa_values = [r.total_ttfa * 1000 for r in results]
        llm_values = [r.llm_total * 1000 for r in results]
        tts_ttfa_values = [r.tts_ttfa * 1000 for r in results]
        
        print(f"LLM Generation:     avg={sum(llm_values)/len(llm_values):.0f}ms")
        print(f"TTS TTFA:           avg={sum(tts_ttfa_values)/len(tts_ttfa_values):.0f}ms")
        print(f"TOTAL TTFA:         avg={sum(ttfa_values)/len(ttfa_values):.0f}ms")
        print(f"                    min={min(ttfa_values):.0f}ms")
        print(f"                    max={max(ttfa_values):.0f}ms")
        
        success_rate = sum(1 for t in ttfa_values if t < 500) / len(ttfa_values) * 100
        print(f"\n✅ TTFA < 500ms:    {success_rate:.0f}% ({sum(1 for t in ttfa_values if t < 500)}/{len(ttfa_values)})")


async def interactive_mode():
    """Интерактивный режим"""
    print("=" * 60)
    print("🎙️ ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("=" * 60)
    print("Введите команду для озвучки (exit для выхода):\n")
    
    if not ELEVENLABS_API_KEY:
        print("❌ ELEVENLABS_API_KEY не найден в .env")
        return
    
    while True:
        try:
            command = input("> ").strip()
            if not command:
                continue
            if command.lower() in ["exit", "quit", "q"]:
                break
                
            result = await run_benchmark(command, play_audio=True)
            result.print_report()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            
    print("\n👋 До свидания!")


def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_mode())
    else:
        asyncio.run(run_full_benchmark())


if __name__ == "__main__":
    main()
