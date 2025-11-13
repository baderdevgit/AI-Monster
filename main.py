import cv2
import subprocess
import time
import numpy as np
import requests
import json

# --------------------
# CONFIG
# --------------------
VOICE = "en-US-SaraNeural"

SYSTEM_PROMPT = """
You are a sassy Monster Energy can.
You speak in short, quick lines.
You complain when grabbed, shaken, or handled.
Keep responses under 12 words.
"""

# --------------------
# OLLAMA CHAT
# --------------------
def ask_ollama(prompt):
    data = {
        "model": "llama3.1",
        "prompt": SYSTEM_PROMPT + "\nHuman: " + prompt + "\nMonster:",
        "stream": False
    }

    r = requests.post("http://localhost:11434/api/generate", json=data)
    return r.json()["response"].strip()

# --------------------
# TTS
# --------------------
def speak(text):
    subprocess.run([
        "edge-tts",
        "--voice", VOICE,
        "--text", text,
        "--write-media", "out.wav"
    ])
    subprocess.run(["start", "out.wav"], shell=True)

# --------------------
# SIMPLE MOTION RECOGNITION
# --------------------
cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

last_speech_time = 0

print("Monster AI Running...")

while True:
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    motion = cv2.countNonZero(thresh)

    # --------------------------
    # Detect SHAKE (lots of motion)
    # --------------------------
    if motion > 15000:
        speak("HEY! Stop shakin' me!")
        time.sleep(1)

    # --------------------------
    # Detect HAND GRAB (medium motion)
    # --------------------------
    elif motion > 6000:
        speak("Ooo—hands off me, human.")
        time.sleep(1)

    # --------------------------
    # Every few seconds, listen and respond
    # --------------------------
    if time.time() - last_speech_time > 5:
        last_speech_time = time.time()

        # (TEMP) Fake speech input for now
        user_text = "hello"
        response = ask_ollama(user_text)
        speak(response)

    frame1 = frame2
    ret, frame2 = cap.read()

    if not ret:
        break

    cv2.imshow("Monster AI", frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
