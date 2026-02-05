import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import cv2
    print("OpenCV OK:", cv2.__version__)
except Exception as e:
    print("OpenCV ERROR:", repr(e))

try:
    import mediapipe as mp
    print("MediaPipe OK:", mp.__version__)
except Exception as e:
    print("MediaPipe ERROR:", repr(e))
