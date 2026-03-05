import sys
import traceback

print(f"Python version: {sys.version}")
try:
    import mediapipe as mp
    print(f"MediaPipe version: {getattr(mp, '__version__', 'unknown')}")
    print(f"MediaPipe file: {mp.__file__}")
    print(f"Solutions found: {hasattr(mp, 'solutions')}")
    if hasattr(mp, 'solutions'):
        print(f"Face Detection: {mp.solutions.face_detection}")
    else:
        print("Attribute 'solutions' is MISSING")
        print(f"Available attributes: {dir(mp)}")
except Exception:
    traceback.print_exc()
