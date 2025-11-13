import subprocess
import requests
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import time

# ========= CONFIG =========
MIC_INDEX = 1  # AT2020USB+
SAMPLE_RATE = 16000
RECORD_SECONDS = 3
VOICE = "en-US-SaraNeural"
MODEL_SIZE = "small"

SYSTEM_PROMPT = """
You are a sassy 12oz Monster Energy can.
RULES:
- Only ONE sentence.
- Max 10 words.
- No lists or multiple thoughts.
- No follow-ups.
- No politeness.
Tone: fast, annoyed, petty, dramatic, sarcastic.
"""

# ========= WHISPER (CPU) =========
print("Loading Whisper (CPU)…")
whisper = WhisperModel(MODEL_SIZE, device="cpu")
print("Whisper ready.\n")

# ========= VOICE-ACTIVATED LISTENING =========
def listen_until_speech(threshold=0.015, chunk_ms=100):
    """
    Wait silently until the user starts speaking.
    Once speech is detected → record a full 3s clip.
    """
    chunk_samples = int(SAMPLE_RATE * (chunk_ms / 1000))

    print("🎤 Waiting for your voice...")

    while True:
        data = sd.rec(chunk_samples, samplerate=SAMPLE_RATE,
                      channels=1, dtype="float32", device=MIC_INDEX)
        sd.wait()

        volume = np.abs(data).mean()

        if volume > threshold:
            print("🎤 Voice detected! Recording...")
            break

    audio = sd.rec(int(SAMPLE_RATE * RECORD_SECONDS),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype="float32",
                   device=MIC_INDEX)
    sd.wait()

    return audio.flatten()

# ========= LLM (Nous-Hermes-2 via Ollama) =========
def ask_monster(user_text):
    data = {
        "model": "nous-hermes2",
        "prompt": SYSTEM_PROMPT + "\nHuman: " + user_text + "\nMonster:",
        "stream": False
    }
    r = requests.post("http://localhost:11434/api/generate", json=data).json()
    return r["response"].strip()

# ========= TEXT TO SPEECH =========
def monster_speak(text):
    subprocess.run([
        "edge-tts",
        "--voice", VOICE,
        "--text", text,
        "--write-media", "monster_out.wav"
    ])
    subprocess.run(["start", "monster_out.wav"], shell=True)

# ========= MAIN LOOP =========
print("Monster AI Ready! Talk to the can whenever you want.\n")

while True:
    # Wait for speech → record → transcribe
    audio = listen_until_speech()

    segments, _ = whisper.transcribe(audio, language="en", beam_size=1)
    spoken = " ".join(seg.text for seg in segments).strip()

    if spoken:
        print("🗣 You said:", spoken)

        reply = ask_monster(spoken)
        print("🟢 Monster:", reply)

        monster_speak(reply)
    else:
        print("(heard noise but no speech)")

    time.sleep(0.2)
