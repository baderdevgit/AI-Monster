from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

model = WhisperModel("small", device="cpu")

def record(duration=3):
    print("🎤 Listening...")
    audio = sd.rec(int(16000 * duration), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def transcribe(audio):
    segments, _ = model.transcribe(audio, language="en")
    out = ""
    for seg in segments:
        out += seg.text + " "
    return out.strip()

audio = record(3)
text = transcribe(audio)

print("You said:", text)
