import subprocess
import requests
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import soundfile as sf
import time

# ============================
# CONFIG
# ============================
MIC_INDEX = 1                # AT2020
SAMPLE_RATE = 16000
MODEL_SIZE = "large-v3"      # GPU model

VOICE = "en-US-SaraNeural"

# VAD Tuning
CHUNK_MS = 0.04              # 40 ms frames
VOICE_THRESHOLD = 0.010
SILENCE_THRESHOLD = 0.007
SILENCE_DURATION = 0.65


SYSTEM_PROMPT = """
You are a sassy Monster Energy can.
Respond in ONE sentence.
Max 10 words.
No lists.
No multiple thoughts.
No soft or polite tone.
Be petty, dramatic, annoyed, sarcastic.
"""

# ============================
# LOAD WHISPER (GPU)
# ============================
print("Loading Whisper large-v3 on GPU…")
whisper = WhisperModel(
    MODEL_SIZE,
    device="cuda",
    compute_type="float16"
)
print("Whisper loaded!\n")


# ============================
# VAD LISTENING
# ============================
def record_with_vad():
    """
    Fast VAD: start recording when voice detected,
    stop after silence.
    """
    chunk_samples = int(SAMPLE_RATE * CHUNK_MS)

    print("Waiting for your voice...")
    audio_chunks = []

    # WAIT FOR SPEECH
    while True:
        chunk = sd.rec(chunk_samples, samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32", device=MIC_INDEX)
        sd.wait()
        if np.abs(chunk).mean() > VOICE_THRESHOLD:
            audio_chunks.append(chunk)
            print("Voice detected. Recording…")
            break

    # RECORD UNTIL SILENCE
    silence_time = 0.0
    while True:
        chunk = sd.rec(chunk_samples, samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32", device=MIC_INDEX)
        sd.wait()
        audio_chunks.append(chunk)

        vol = np.abs(chunk).mean()
        if vol < SILENCE_THRESHOLD:
            silence_time += CHUNK_MS
            if silence_time >= SILENCE_DURATION:
                print("Silence detected. Stop.")
                break
        else:
            silence_time = 0

    return np.concatenate(audio_chunks).flatten()


# ============================
# TRANSCRIPTION
# ============================
def transcribe(audio):
    segments, _ = whisper.transcribe(
        audio,
        language="en",
        beam_size=1,
        vad_filter=False
    )
    return " ".join(seg.text for seg in segments).strip()


# ============================
# LLM: Nous Hermes 2
# ============================
def ask_monster(text):
    data = {
        "model": "nous-hermes2",
        "prompt": SYSTEM_PROMPT + "\nHuman: " + text + "\nMonster:",
        "stream": False
    }
    resp = requests.post("http://localhost:11434/api/generate", json=data).json()
    return resp["response"].strip()


# ============================
# TEXT TO SPEECH
# ============================
def monster_speak(text):
    subprocess.run([
        "edge-tts", "--voice", VOICE,
        "--text", text,
        "--write-media", "monster_out.wav"
    ])
    data, sr = sf.read("monster_out.wav", dtype="float32")
    sd.play(data, sr)
    sd.wait()


# ============================
# MAIN LOOP
# ============================
print("Monster Can AI ready.\nSpeak at any time.\n")

while True:
    audio = record_with_vad()

    if len(audio) < 5000:  # ignore tiny noise
        print("Too short, ignoring.\n")
        continue

    spoken = transcribe(audio)
    if not spoken:
        print("No speech recognized.\n")
        continue

    print("You said:", spoken)

    reply = ask_monster(spoken)
    print("Monster:", reply)

    monster_speak(reply)

    time.sleep(0.1)
    print("Ready.\n")
