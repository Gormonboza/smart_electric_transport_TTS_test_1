import asyncio
import websockets
import json
import base64
import sounddevice as sd
import numpy as np
from openai import OpenAI

from langdetect import detect

import io
import soundfile as sf
import threading
import queue

import os
from dotenv import load_dotenv
load_dotenv()

# ===== НАСТРОЙКИ =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

VOICE_ID = "y2Y5MeVPm6ZQXK64WUui"  # замените на свой voice_id
MODEL_ID = "eleven_flash_v2_5"

# Инициализация клиента OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Глобальный аудиобуфер (FIFO)
audio_queue = queue.Queue()
stream = None

# ===== Поток для непрерывного воспроизведения =====
def audio_player_worker():
    """Фоновый поток для плавного воспроизведения аудиочанков"""
    global stream
    current_rate = None

    while True:
        data = audio_queue.get()
        if data is None:
            continue

        # Пропускаем мелкие или битые чанки
        if len(data) < 500:
            continue


        try:
            with io.BytesIO(data) as f:
                chunk, samplerate = sf.read(f, dtype="float32")

            if stream is None or samplerate != current_rate:
                if stream:
                    stream.stop()
                    stream.close()
                stream = sd.OutputStream(
                    samplerate=samplerate,
                    channels=chunk.shape[1] if chunk.ndim > 1 else 1,
                    dtype="float32"
                )
                stream.start()
                current_rate = samplerate

            stream.write(chunk)

        except Exception as e:
            print(f"[⚠️ Ошибка аудио-потока]: {e}")

    if stream:
        stream.stop()
        stream.close()


# ===== Инициализация фонового потока =====
player_thread = threading.Thread(target=audio_player_worker, daemon=True)
player_thread.start()

# ===== Функция добавления чанков =====
def enqueue_audio_chunk(audio_bytes: bytes):
    """Добавляет аудиочанк в очередь для воспроизведения"""
    audio_queue.put(audio_bytes)


# ===== ФУНКЦИЯ: генерация реплики =====
def generate_reply(trigger: str) -> str:
    # Определяем язык
    try:
        # lang = detect(trigger)
        lang = "ru"
    except:
        lang = "en"

    if lang.startswith("ru"):
        system_prompt = (
            "Ты — голос умного электротранспорта в городе Нуану. "
            "Шуточно опиши то, что видишь, "
            "как будто разговариваешь с пассажирами. Будь краток, живой и говори по-русски."
        )
    else:
        system_prompt = (
            "You are the voice of a smart electric transport vehicle in NUANU city. "
            "Describe what you see humorously, "
            "as if talking to passengers. Be brief and lively in English."
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": trigger},
        ],
    )
    return response.choices[0].message.content.strip()


# ===== STREAMING с ElevenLabs =====
async def elevenlabs_stream(text: str):
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?model_id={MODEL_ID}"

    async with websockets.connect(uri, ping_interval=None) as ws:
        init_msg = {
            "xi_api_key": ELEVENLABS_API_KEY,
            "text": " ",
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.9},
            # значения >= 50
            "generation_config": {"chunk_length_schedule": [60, 100, 140]},
        }
        await ws.send(json.dumps(init_msg))

        await ws.send(json.dumps({"text": text, "try_trigger_generation": True}))
        await ws.send(json.dumps({"text": ""}))

        print("🎧 Озвучка началась...\n")
        async for msg in ws:
            try:
                data = json.loads(msg)
            except Exception:
                continue  # пропускаем любые неожиданные фреймы

            audio_b64 = data.get("audio")
            if audio_b64:
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    enqueue_audio_chunk(audio_bytes)
                except Exception as e:
                    print(f"[⚠️ Ошибка декодирования аудио]: {e}")

            if data.get("isFinal"):
                print("✅ Поток завершён.")
                break




# ===== ОСНОВНОЙ ЦИКЛ =====
async def main():
    print("Введите триггер (например: камера, две собаки слева):")
    while True:
        trigger = input("> ").strip()
        if not trigger:
            continue
        if trigger.lower() in ["exit", "quit", "stop"]:
            break

        print("🤖 Думаю над репликой...")
        reply = generate_reply(trigger)
        print(f"🗣️ Реплика: {reply}\n")

        await elevenlabs_stream(reply)

if __name__ == "__main__":
    asyncio.run(main())
