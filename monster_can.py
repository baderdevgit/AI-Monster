import cv2
import subprocess
import requests
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from pvrecorder import PvRecorder
from faster_whisper import WhisperModel
import time

# ===========================================================
# CONFIG
# ===========================================================
MIC_INDEX = 0
SAMPLE_RATE = 16000
MODEL_SIZE = "medium"

VOICE = "en-US-SaraNeural"

# VAD thresholds
START_THRESHOLD = 200
SILENCE_THRESHOLD = 120
SILENCE_FRAMES = 10  # ~300ms

# Pickup detection
PICKUP_INTERVAL = 0.8
PICKUP_THRESHOLD = 0.6

audio_lock = threading.Lock()
listening_flag = False      # audio thread listens only when free

SYSTEM_PROMPT = """
You are a sassy Monster Energy can.
One short sentence. Max 10 words.
Sarcastic, annoyed, petty, dramatic.
"""


# ===========================================================
# LOAD WHISPER (GPU)
# ===========================================================
print("Loading Whisper (GPU)...")
whisper = WhisperModel(
    MODEL_SIZE,
    device="cuda",
    compute_type="float16"
)
print("Whisper loaded.")


# ===========================================================
# TEXT TO SPEECH
# ===========================================================
def monster_speak(text):
    if not text.strip():
        print("TTS skipped: empty text")
        return

    with audio_lock:
        try:
            import os
            if os.path.exists("monster_out.wav"):
                os.remove("monster_out.wav")

            subprocess.run([
                "edge-tts",
                "--voice", VOICE,
                "--text", text,
                "--write-media", "monster_out.wav"
            ], check=True)

            audio, sr = sf.read("monster_out.wav", dtype="float32")
            sd.play(audio, sr)
            sd.wait()

        except Exception as e:
            print("TTS ERROR:", e)


# ===========================================================
# LLM CALL — llama3.2-vision
# ===========================================================
def ask_monster(prompt):
    payload = {
        "model": "llama3.2",
        "prompt": SYSTEM_PROMPT + "\nHuman: " + prompt + "\nMonster:",
        "stream": False
    }
    r = requests.post("http://localhost:11434/api/generate", json=payload).json()
    return r.get("response", "").strip()


# ===========================================================
# AUDIO VAD THREAD
# ===========================================================
def audio_listener():
    global listening_flag

    while True:
        if listening_flag:   # don't double-record
            time.sleep(0.05)
            continue

        listening_flag = True
        print("(Audio) Waiting for voice...")

        recorder = PvRecorder(device_index=MIC_INDEX, frame_length=512)
        recorder.start()

        audio_frames = []
        speaking = False
        silence_count = 0

        while True:
            frame = recorder.read()
            volume = sum(abs(x) for x in frame) / len(frame)

            if not speaking:
                if volume > START_THRESHOLD:
                    speaking = True
                    audio_frames.extend(frame)
                    print("(Audio) Recording…")
            else:
                audio_frames.extend(frame)

                if volume < SILENCE_THRESHOLD:
                    silence_count += 1
                    if silence_count >= SILENCE_FRAMES:
                        break
                else:
                    silence_count = 0

        recorder.stop()
        recorder.delete()

        # Convert PCM → float32
        audio = np.array(audio_frames, dtype=np.float32) / 32768.0

        listening_flag = False

        if len(audio) > 5000:
            transcribe_and_reply(audio)


def transcribe_and_reply(audio):
    text = transcribe(audio)
    if not text or text == 'Thanks for watching!' or text == 'you':
        print("(Audio) No speech recognized.")
        return

    print("You said:", text)
    reply = ask_monster(text)
    threading.Thread(target=monster_speak, args=(reply,)).start()


# ===========================================================
# WHISPER TRANSCRIPTION
# ===========================================================
def transcribe(audio):
    segments, _ = whisper.transcribe(
        audio, language="en", beam_size=1, vad_filter=False
    )
    return " ".join(s.text for s in segments).strip()


# ===========================================================
# VISION PICKUP DETECTION
# ===========================================================
def detect_pickup(frame):
    _, jpg = cv2.imencode(".jpg", frame)

    payload = {
        "model": "llama3.2-vision",
        "prompt": """
Look at the image. Is the Monster Energy can being picked up?
Return ONLY a number 0-1.
""",
        "images": [jpg.tobytes()],
        "stream": False
    }

    r = requests.post("http://localhost:11434/api/generate", json=payload).json()
    response = r.get("response", "0").strip()

    try:
        return float(response.split()[0])
    except:
        return 0.0


# ===========================================================
# MAIN LOOP — video + pickup thread
# ===========================================================
print("Monster Can AI ready.")

# Start audio listener thread
threading.Thread(target=audio_listener, daemon=True).start()

cap = cv2.VideoCapture(0)
last_pickup = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # SHOW WEBCAM
    cv2.imshow("Monster Cam", frame)
    if cv2.waitKey(1) == 27:
        break

    # PICKUP DETECT
    now = time.time()
    if now - last_pickup > PICKUP_INTERVAL:
        prob = detect_pickup(frame)
        # print("Pickup:", prob)

        if prob > PICKUP_THRESHOLD:
            last_pickup = now
            reply = ask_m_
