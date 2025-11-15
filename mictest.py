from pvrecorder import PvRecorder

print("Available devices:")
for i, device in enumerate(PvRecorder.get_available_devices()):
    print(i, device)
