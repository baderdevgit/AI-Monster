import ctypes

try:
    ctypes.CDLL("cudnn64_9.dll")
    print("cuDNN loaded successfully! 🎉")
except OSError as e:
    print("❌ cuDNN NOT loaded:")
    print(e)
