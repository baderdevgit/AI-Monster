import subprocess
import requests
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import time

# ========= CONFIG =========
MIC_INDEX = 1  # AT2020USB+
SAMPLE_RATE = 16000
LISTEN_DURATION = 10  # seconds
MODEL_SIZE = "small"  # CPU whisper size

VOICE = "en-US-SaraNeural"

SYSTEM_PROMPT = """
You are a sassy Monster Energy can.
VERY IMPORTANT RULES:
- Only ONE sentence.
- Max 10 words.
- No lists.
- No multiple thoughts.
- No follow-ups.
- No extra commentary.
- No politeness.

Your tone: fast, annoyed, petty, dramatic, sarcastic.
If the user asks a long question, STILL reply with one short sentence.

Examples of correct answers:
- "Quit shaking me, psycho."
- "Back up, I'm fizzing."
- "Keep your hands off me."
"""

# ========= WHISPER (CPU) =========
print("Loading Whisper (CPU)…")
whisper = WhisperModel(MODEL_SIZE, device="cpu")
print("Whisper ready.\n")

def listen_once():
    print("Listening...")
    audio = sd.rec(
        int(SAMPLE_RATE * LISTEN_DURATION),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=MIC_INDEX
    )
    sd.wait()
    audio = audio.flatten()

    segments, _ = whisper.transcribe(audio, language="en", beam_size=1)
    text = " ".join(seg.text for seg in segments).strip()
    return text


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


# ========= MAIN =========
print("Monster AI Ready! Talk to the can whenever you want.\n")
print("Say something → wait → hear the can yell at you.\n")

while True:
    spoken = listen_once()

    if spoken.strip():
        print("You said:", spoken)

        reply = ask_monster(spoken)
        print("Monster:", reply)

        monster_speak(reply)
    else:
        print("No speech detected.")

    time.sleep(0.2)
