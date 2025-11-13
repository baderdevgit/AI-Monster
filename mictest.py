import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import traceback

print("Loading Whisper...")
model = WhisperModel("small", device="dml")
print("Whisper ready.\n")


def listen_debug():
    print("🎤 Trying to open mic...")

    try:
        # OPEN A REAL INPUT STREAM (this exposes format problems)
        stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="int16"     # GPU preferred
        )
        stream.start()
        print("Mic stream opened successfully.")

        print("Recording 3 seconds...")
        audio = stream.read(int(16000 * 3))[0]
        stream.stop()
        stream.close()

        print("Audio shape:", audio.shape)
        return audio.flatten().astype("float32") / 32768.0

    except Exception as e:
        print("\n❌ LISTEN ERROR:")
        print(type(e).__name__, e)
        traceback.print_exc()
        return None


while True:
    print("\n----- NEW LOOP -----")
    try:
        audio = listen_debug()

        if audio is None:
            print("Audio was None, retrying...")
            continue

        print("Transcribing...")
        segments, _ = model.transcribe(audio, language="en", beam_size=1)
        text = " ".join(seg.text for seg in segments).strip()

        print(">> TRANSCRIBED:", text)

    except Exception as e:
        print("\n❌ MAIN LOOP ERROR:")
        print(type(e).__name__, e)
        traceback.print_exc()

    print("\nSleeping...")
    import time
    time.sleep(1)
