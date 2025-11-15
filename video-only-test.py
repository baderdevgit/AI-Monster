import cv2
import requests
import base64
import time

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


def main():
    print("Starting webcam for VISION DEBUG...")
    cap = cv2.VideoCapture(0)

    last_check = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Vision Debug Window", frame)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

        now = time.time()
        if now - last_check > PICKUP_INTERVAL:
            last_check = now
            prob = detect_pickup(frame)
            print(f"Pickup Probability: {prob:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
