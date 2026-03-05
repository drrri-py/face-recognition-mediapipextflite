import urllib.request
import os

# Set project root and model directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_URL = "https://github.com/MCarlomagno/FaceRecognitionAuth/raw/master/assets/mobilefacenet.tflite"
MODEL_PATH = os.path.join(MODEL_DIR, "mobilefacenet.tflite")

# MediaPipe Task Model
MP_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MP_MODEL_PATH = os.path.join(MODEL_DIR, "face_detector.bbox.tflite")

def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return

    print(f"Downloading model from {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model downloaded successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("Please manually download mobilefacenet.tflite and place it in the 'models/' folder.")

    # Download MediaPipe Face Detector Model
    if os.path.exists(MP_MODEL_PATH):
        print(f"MediaPipe model already exists at {MP_MODEL_PATH}")
    else:
        print(f"Downloading MediaPipe model from {MP_MODEL_URL}...")
        try:
            print(f"Note: This might take a moment (approx 5MB)...")
            urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)
            print(f"MediaPipe model downloaded successfully to {MP_MODEL_PATH}")
        except Exception as e:
            print(f"Failed to download MediaPipe model: {e}")

if __name__ == "__main__":
    download_model()
