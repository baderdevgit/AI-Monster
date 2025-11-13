import subprocess
import requests
import torch
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import soundfile as sf

# =====================================
# CONFIG
# =====================================
MIC_INDEX = 1
SAMPLE_RATE = 16000
VOICE = "en-US-SaraNeural"

WHISPER_MODEL = "large-v3"  # Now works on GPU

# Silero requirements
CHUNK_SAMPLES = 512        # MUST be 512 for 16 kHz
VAD_THRESHOLD = 0.5
SILENCE_THRESHOLD = 0.25
SILENCE_SEC = 0.40

SYSTEM_PROMPT = """
You are a sassy Monster Energy can.
RULES:
- One sentence only.
- Maximum 10 words.
- No lists.
- No multiple thoughts.
- No apologies.
Tone: angry, petty, annoyed, sarcastic, dramatic.
"""

# =====================================
# LOAD VAD
# =====================================
print("Loading Silero VAD...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
print("VAD ready.")

# =====================================
# LOAD WHISPER GPU ACCELERATED
# =====================================
print("Loading Whisper...")
whisper = WhisperModel(
    WHISPER_MODEL,
    device="cuda",
    compute_type="float16"
)
print("Whisper ready.\n")


# =====================================
# RECORD WITH SILERO VAD
# =====================================
def record_with_silero_vad():
    print("Waiting for speech...")

    # Wait for voice to start
    while True:
        audio = sd.rec(CHUNK_SAMPLES, samplerate=SAMPLE_RATE,
                        channels=1, dtype="float32", device=MIC_INDEX)
        sd.wait()

        pcm = torch.from_numpy(audio.flatten())
        vad_prob = vad_model(pcm, SAMPLE_RATE).item()

        if vad_prob > VAD_THRESHOLD:
            print("Speech detected. Recording...")
            collected = [audio]
            break

    # Keep recording until silence
    silence_chunks = 0
    silence_required = int(SILENCE_SEC / (CHUNK_SAMPLES / SAMPLE_RATE))

    while True:
        audio = sd.rec(CHUNK_SAMPLES, samplerate=SAMPLE_RATE,
                        channels=1, dtype="float32", device=MIC_INDEX)
        sd.wait()

        collected.append(audio)
        pcm = torch.from_numpy(audio.flatten())
        vad_prob = vad_model(pcm, SAMPLE_RATE).item()

        if vad_prob < SILENCE_THRESHOLD:
            silence_chunks += 1
            if silence_chunks >= silence_required:
                print("Silence detected. Stopping.")
                break
        else:
            silence_chunks = 0

    return np.concatenate(collected).flatten()


# =====================================
# TRANSCRIBE
# =====================================
def transcribe(audio):
    segments, _ = whisper.transcribe(
        audio,
        language="en",
        beam_size=1,
        vad_filter=False
    )
    return " ".join(seg.text for seg in segments).strip()


# =====================================
# LLM RESPONSE
# =====================================
def ask_monster(text):
    data = {
        "model": "nous-hermes2",
        "prompt": SYSTEM_PROMPT + "\nHuman: " + text + "\nMonster:",
        "stream": False
    }
    r = requests.post("http://localhost:11434/api/generate", json=data).json()
    return r["response"].strip()


# =====================================
# TEXT TO SPEECH
# =====================================
def monster_speak(text):
    subprocess.run([
        "edge-tts",
        "--voice", VOICE,
        "--text", text,
        "--write-media", "monster_out.wav"
    ])
    data, sr = sf.read("monster_out.wav", dtype="float32")
    sd.play(data, sr)
    sd.wait()


# =====================================
# MAIN LOOP
# =====================================
print("Monster Can AI ready.")
print("Start talking whenever you want.\n")

while True:
    audio = record_with_silero_vad()

    if len(audio) < SAMPLE_RATE * 0.3:
        print("Too short, ignoring.\n")
        continue

    spoken = transcribe(audio)

    if spoken:
        print("You said:", spoken)
        reply = ask_monster(spoken)
        print("Monster:", reply)
        monster_speak(reply)
    else:
        print("No speech recognized.\n")

    print("Ready.\n")
