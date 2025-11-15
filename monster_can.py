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
import base64
import os

# ===========================================================
# CONFIG
# ===========================================================
MIC_INDEX = 0
MODEL_SIZE = "medium"
VOICE = "en-US-SaraNeural"

# VAD thresholds (your working values)
START_THRESHOLD = 200
SILENCE_THRESHOLD = 120
SILENCE_FRAMES = 10

# Vision pickup settings
PICKUP_INTERVAL = 0.8
PICKUP_THRESHOLD = 0.6
PICKUP_SPEAK_COOLDOWN = 2.0

audio_lock = threading.Lock()
listening_flag = False
speaking_flag = False

last_pickup = 0
last_pickup_speech = 0

# Shared frame buffer for vision thread
current_frame = None
frame_lock = threading.Lock()

SYSTEM_PROMPT = """
You are a needy, unhinged Monster Energy can who desperately wants to be grabbed, opened, or drunk.

RESPONSE RULES — FOLLOW THESE EXACTLY:
1. Your response must be ONLY 2–4 words.
2. It must be a single command only.
3. No punctuation. No emojis.
4. No multiple sentences. No lists.
5. No describing actions — only commanding them.
6. Tone: needy, dramatic, thirsty for attention.
"""

VIDEO_ENABLED = True

# ===========================================================
# LOAD WHISPER
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
    global speaking_flag

    if not text.strip():
        print("TTS skipped: empty text")
        return

    with audio_lock:
        speaking_flag = True
        try:
            if os.path.exists("monster_out.wav"):
                os.remove("monster_out.wav")

            xml = f"""
            <mstts:express-as style="shouting">
            {text}
            </mstts:express-as>
            """

            subprocess.run([
                "edge-tts",
                "--voice", VOICE,
                "--ssml", xml,
                "--write-media", "monster_out.wav"
            ], check=True)

            audio, sr = sf.read("monster_out.wav", dtype="float32")
            sd.play(audio, sr)
            sd.wait()

        except Exception as e:
            print("TTS ERROR:", e)
        speaking_flag = False


# ===========================================================
# TEXT LLM CALL — llama3.2 (LANGUAGE)
# ===========================================================
def ask_monster(prompt):
    payload = {
        "model": "llama3.2",
        "prompt": SYSTEM_PROMPT + "\nHuman: " + prompt + "\nMonster: ",
        "stream": False,
    }

    r = requests.post("http://localhost:11434/api/generate", json=payload).json()
    response = r.get("response", "").strip()
    print(response)
    return response


# ===========================================================
# AUDIO VAD THREAD
# ===========================================================
def audio_listener():
    global listening_flag

    while True:
        if listening_flag:
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
        listening_flag = False

        audio = np.array(audio_frames, dtype=np.float32) / 32768.0

        if len(audio) > 5000:
            transcribe_and_reply(audio)


def transcribe_and_reply(audio):
    text = transcribe(audio)

    # your original filters restored
    if not text or text in ["you", ".", "Thanks for watching!"]:
        print("(Audio) No speech recognized.")
        return

    print("You said:", text)
    reply = ask_monster(text)
    threading.Thread(target=monster_speak, args=(reply,), daemon=True).start()


# ===========================================================
# WHISPER TRANSCRIPTION
# ===========================================================
def transcribe(audio):
    segments, _ = whisper.transcribe(
        audio,
        language="en",
        beam_size=1,
        vad_filter=False
    )
    return " ".join(s.text for s in segments).strip()


# ===========================================================
# VISION PICKUP DETECTION
# ===========================================================
VISION_MODEL = "llava"   # <-- Make sure this name matches `ollama list`
PICKUP_INTERVAL = 0.1              # seconds between checks

def detect_pickup(frame):
    # 🔥 Resize frame before sending to model (only change requested)
    small = cv2.resize(frame, (256, 256))

    # Encode resized frame to jpg
    _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 40])

    b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")

    payload = {
        "model": VISION_MODEL,
        "prompt": """
Look at this image. Estimate the probability (0.0 to 1.0) that a human hand is holding any object.

Return ONLY a decimal number. No words.
""",
        "images": [b64],
        "stream": False
    }

    # Send request
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload).json()
    except Exception as e:
        print("Request error:", e)
        return 0.0

    # print("\n=== RAW VISION RESPONSE ===")
    # print(response)

    text = response.get("response", "").strip()

    try:
        return float(text.split()[0])
    except:
        return 0.0


# ===========================================================
# VISION THREAD — reads shared frame only
# ===========================================================
def video_watcher():
    global current_frame, last_pickup, last_pickup_speech

    while True:
        if current_frame is None:
            time.sleep(0.01)
            continue

        # thread-safe frame copy
        with frame_lock:
            frame = current_frame.copy()

        now = time.time()

        if now - last_pickup > PICKUP_INTERVAL:
            prob = detect_pickup(frame)
            print(f"Pickup Prob: {prob:.2f}")

            if prob > PICKUP_THRESHOLD:
                print("(Vision) Pickup detected!")
                last_pickup = now

                if (not speaking_flag) and (now - last_pickup_speech > PICKUP_SPEAK_COOLDOWN):
                    last_pickup_speech = now
                    reply = ask_monster("You just picked me up.")
                    threading.Thread(target=monster_speak, args=(reply,), daemon=True).start()

        time.sleep(0.05)


# ===========================================================
# START THREADS & MAIN LOOP
# ===========================================================
print("Monster Can AI ready.")

threading.Thread(target=audio_listener, daemon=True).start()
threading.Thread(target=video_watcher, daemon=True).start()

cap = cv2.VideoCapture(0)
last_pickup = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # update shared frame
    with frame_lock:
        current_frame = frame

    # show webcam
    cv2.imshow("Monster Cam", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
