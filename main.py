import cv2
import numpy as np
import numpy as np
import os
import time
import tensorflow as tf
from database import initialize_db, log_attendance, has_attended_today
from utils import draw_status

# Configuration
KNOWN_FACES_DIR = "known_faces"
MODEL_PATH = "models/mobilefacenet.tflite"
THRESHOLD = 0.5  # Cosine Similarity threshold (higher is more similar)
COOLDOWN_TIME = 3
IMG_SIZE = 112 # MobileFaceNet input size

# Initialize OpenCV Face Detection (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Could not load Haar Cascade XML file.")

# Initialize TFLite Interpreter
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_embedding(interpreter, face_img):
    # Preprocess
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = (face_img - 127.5) / 128.0
    face_img = np.expand_dims(face_img, axis=0).astype(np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], face_img)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])
    
    # L2 Normalize
    norm = np.linalg.norm(embedding)
    return embedding / norm

def cosine_similarity(v1, v2):
    return np.dot(v1, v2.T)[0][0]

def load_known_faces(interpreter):
    known_embeddings = []
    known_info = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Directory '{KNOWN_FACES_DIR}' created.")
        return known_embeddings, known_info

    print("Encoding known faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                name_parts = os.path.splitext(filename)[0].split("_", 1)
                if len(name_parts) == 2:
                    user_id, name = name_parts
                else:
                    user_id = "UnknownID"
                    name = name_parts[0]

                image = cv2.imread(path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect face in the reference image (use OpenCV)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Take the largest detection
                    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                    left, top = max(0, x), max(0, y)
                    
                    face_roi = image_rgb[top:top+h, left:left+w]
                    if face_roi.size > 0:
                        embedding = get_embedding(interpreter, face_roi)
                        known_embeddings.append(embedding)
                        known_info.append((user_id, name))
                        print(f"Loaded and encoded: {name}")
                else:
                    print(f"No face found in {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    return known_embeddings, known_info

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please run download_model.py first.")
        return

    initialize_db()
    interpreter = load_tflite_model(MODEL_PATH)
    known_embeddings, known_info = load_known_faces(interpreter)
    
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video source.")
        return

    last_status = None
    last_status_name = None
    status_timer = 0

    print("MediaPipe + TFLite starting... Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        ih, iw, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                left, top = max(0, x), max(0, y)
                right, bottom = min(iw, x + w), min(ih, y + h)

                face_roi = rgb_frame[top:bottom, left:right]
                
                name = "Unknown"
                user_id = None

                if face_roi.size > 0:
                    current_embedding = get_embedding(interpreter, face_roi)
                    
                    # Compare with known embeddings
                    max_sim = -1
                    best_match_idx = -1
                    
                    for i, known_emb in enumerate(known_embeddings):
                        sim = cosine_similarity(current_embedding, known_emb)
                        if sim > max_sim:
                            max_sim = sim
                            best_match_idx = i
                    
                    if max_sim > THRESHOLD:
                        user_id, name = known_info[best_match_idx]

                # UI: Draw Box
                color = (0, 255, 0) if user_id else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, f"{name} ({max_sim:.2f})", (left, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Attendance Logic
                if time.time() > status_timer:
                    if user_id:
                        if has_attended_today(user_id):
                            last_status = 'ALREADY_ATTENDED'
                            last_status_name = name
                        else:
                            log_attendance(user_id, name)
                            last_status = 'SUCCESS'
                            last_status_name = name
                        status_timer = time.time() + COOLDOWN_TIME
                    elif max_sim < THRESHOLD and max_sim > 0:
                        last_status = 'UNRECOGNIZED'
                        last_status_name = None
                        status_timer = time.time() + COOLDOWN_TIME

        if time.time() > status_timer:
            last_status = None
            last_status_name = None

        frame = draw_status(frame, last_status, last_status_name)
        cv2.imshow('Face Recognition TFLite', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
