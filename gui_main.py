import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageSequence
import tensorflow as tf
from database import initialize_db, log_attendance, has_attended_today, get_last_attendance
import threading
import queue

# Configuration
KNOWN_FACES_DIR = "known_faces"
MODEL_PATH = "models/mobilefacenet.tflite"
THRESHOLD = 0.5
COOLDOWN_TIME = 3
IMG_SIZE = 112
DETECTION_SCALE = 0.5  # Scale factor for face detection (0.5 = 50% size)

# CustomTkinter Setup
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class AnimationLabel(tk.Label):
    def __init__(self, master, frames, **kwargs):
        # Handle "transparent" color name from CustomTkinter
        bg_color = master._fg_color if hasattr(master, "_fg_color") else "#FFFFFF"
        if bg_color == "transparent" or not isinstance(bg_color, str) or not bg_color.startswith("#"):
            bg_color = "#FFFFFF" # Default to white if "transparent" or invalid
        
        super().__init__(master, bg=bg_color, borderwidth=0, highlightthickness=0, **kwargs)
        self.frames = frames
        self.duration = 40 
        self.idx = 0
        self.cancel_id = None
        self.animate()

    def animate(self):
        if not self.frames: return
        self.config(image=self.frames[self.idx])
        self.idx = (self.idx + 1) % len(self.frames)
        self.cancel_id = self.after(self.duration, self.animate)

    def stop(self):
        if self.cancel_id:
            self.after_cancel(self.cancel_id)
            self.cancel_id = None

class FaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Frameless Window
        self.overrideredirect(True)
        self.title("Face Recognition Attendance System")
        self.geometry("1100x650")
        
        # Center Window
        self.center_window(1100, 650)
        
        # Draggable Window Support
        self.bind("<ButtonPress-1>", self.start_move)
        self.bind("<ButtonRelease-1>", self.stop_move)
        self.bind("<B1-Motion>", self.do_move)

        # macOS Colors
        self.bg_color = "#F5F5F7"
        self.sidebar_color = "#FFFFFF"
        self.accent_color = "#007AFF"
        self.configure(fg_color=self.bg_color)

        # Grid layout
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Side: Camera View
        self.camera_card = ctk.CTkFrame(self, corner_radius=20, fg_color="white", border_width=1, border_color="#E5E5E5")
        self.camera_card.grid(row=0, column=0, padx=30, pady=30, sticky="nsew")
        self.camera_label = tk.Label(self.camera_card, bg="white")
        self.camera_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Right Side: Status Panel
        self.panel = ctk.CTkFrame(self, width=320, corner_radius=20, fg_color=self.sidebar_color, border_width=1, border_color="#E5E5E5")
        self.panel.grid(row=0, column=1, padx=(0, 30), pady=30, sticky="nsew")
        
        self.title_label = ctk.CTkLabel(self.panel, text="Face Attendance", font=ctk.CTkFont(family="SF Pro Display", size=22, weight="bold"))
        self.title_label.pack(pady=(40, 30))

        self.info_container = ctk.CTkFrame(self.panel, fg_color="transparent")
        self.info_container.pack(fill="x", padx=30, pady=20)

        self.create_info_row("NAMA", "name_label")
        self.create_info_row("ID USER", "id_label")
        self.create_info_row("TERAKHIR", "time_label")

        self.gif_container = ctk.CTkFrame(self.panel, width=200, height=200, fg_color="transparent")
        self.gif_container.pack(pady=20, expand=True)
        self.active_anim = None

        self.hint_label = ctk.CTkLabel(self.panel, text="Press Ctrl+Shift+Q to Exit", font=ctk.CTkFont(size=10), text_color="#AEAEB2")
        self.hint_label.pack(side="bottom", pady=20)

        # Initialize Logic & Threading
        self.init_logic()
        
        self.bind_all("<Control-Shift-KeyPress-Q>", lambda e: self.on_closing())
        self.bind_all("<Control-Shift-KeyPress-q>", lambda e: self.on_closing())
        
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def center_window(self, width, height):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def stop_move(self, event):
        self.x = None
        self.y = None

    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.winfo_x() + deltax
        y = self.winfo_y() + deltay
        self.geometry(f"+{x}+{y}")

    def create_info_row(self, label_text, attr_name):
        row = ctk.CTkFrame(self.info_container, fg_color="transparent")
        row.pack(fill="x", pady=8)
        lbl = ctk.CTkLabel(row, text=label_text, font=ctk.CTkFont(size=10, weight="bold"), text_color="#8E8E93")
        lbl.pack(anchor="w")
        val = ctk.CTkLabel(row, text="---", font=ctk.CTkFont(size=15, weight="normal"), text_color="#1C1C1E")
        val.pack(anchor="w")
        setattr(self, attr_name, val)

    def init_logic(self):
        initialize_db()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.known_embeddings, self.known_info = self.load_known_faces()
        
        # Pre-load status animations to prevent freeze
        self.preloaded_anims = {
            "success": self.load_sequence_into_memory(os.path.join("assets", "animations", "success")),
            "failure": self.load_sequence_into_memory(os.path.join("assets", "animations", "failure"))
        }
        
        self.status_timer = 0
        self.unknown_buffer = 0  # Buffer untuk menangani kedipan/flicker
        self.current_status_type = None
        self.cam_w, self.cam_h = 780, 580
        
        self.camera_mask = Image.new('L', (self.cam_w, self.cam_h), 0)
        draw = ImageDraw.Draw(self.camera_mask)
        draw.rounded_rectangle((0, 0, self.cam_w, self.cam_h), radius=35, fill=255)

        # Threading Setup
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.processing_thread = threading.Thread(target=self.process_faces_thread, daemon=True)
        self.processing_thread.start()
        
        self.current_results = []
        self.frame_count = 0

    def load_sequence_into_memory(self, folder):
        folder_path = os.path.join(os.getcwd(), folder)
        if not os.path.exists(folder_path): return []
        
        frames = []
        bg_rgb = self.hex_to_rgb(self.sidebar_color)
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
        
        for filename in files:
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                frame = img.convert("RGBA")
                bg_image = Image.new("RGBA", frame.size, bg_rgb + (255,))
                composite = Image.alpha_composite(bg_image, frame)
                frames.append(ImageTk.PhotoImage(composite.convert("RGB")))
        return frames

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def load_known_faces(self):
        embeddings = []
        info = []
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
            return embeddings, info

        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.endswith((".jpg", ".png")):
                path = os.path.join(KNOWN_FACES_DIR, filename)
                img = cv2.imread(path)
                if img is None: continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                    face_roi = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                    emb = self.get_embedding(face_roi)
                    embeddings.append(emb)
                    
                    # Logika penamaan fleksibel: ID_NAMA_OPSIONAL.jpg
                    # Contoh : "101_Rendriyan_Oktaviadi_Saputra_2.jpg" -> ID: 101, Nama: Rendriyan Oktaviadi Saputra
                    name_parts = os.path.splitext(filename)[0].split("_")
                    user_id = name_parts[0]
                    
                    if len(name_parts) >= 3:
                        # Gabungkan bagian tengah sebagai nama (menangani nama panjang dengan underscore)
                        user_name = " ".join(name_parts[1:-1])
                    elif len(name_parts) == 2:
                        user_name = name_parts[1]
                    else:
                        user_name = "Unknown"
                        
                    info.append((user_id, user_name))
        return embeddings, info

    def get_embedding(self, face_img):
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_img = (face_img - 127.5) / 128.0
        face_img = np.expand_dims(face_img, axis=0).astype(np.float32)

        input_det = self.interpreter.get_input_details()
        output_det = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_det[0]['index'], face_img)
        self.interpreter.invoke()
        emb = self.interpreter.get_tensor(output_det[0]['index'])
        return emb / (np.linalg.norm(emb) + 1e-10)

    def process_faces_thread(self):
        """Heavy processing happens here (Detection + Recognition)"""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # 1. Downsample for faster detection
                small_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                # 2. Faster detection on small image
                faces = self.face_cascade.detectMultiScale(gray, 1.2, 4)
                
                results = []
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                for (x, y, w, h) in faces:
                    # Scale back coordinates
                    sx, sy, sw, sh = int(x/DETECTION_SCALE), int(y/DETECTION_SCALE), int(w/DETECTION_SCALE), int(h/DETECTION_SCALE)
                    face_roi = rgb_frame[sy:sy+sh, sx:sx+sw]
                    
                    name, user_id = "Unknown", None
                    max_sim = 0

                    if face_roi.size > 0:
                        current_emb = self.get_embedding(face_roi)
                        for i, k_emb in enumerate(self.known_embeddings):
                            sim = np.dot(current_emb, k_emb.T)[0][0]
                            if sim > max_sim:
                                max_sim = sim
                                user_id, name = self.known_info[i]
                    
                    results.append({
                        'rect': (sx, sy, sw, sh),
                        'sim': max_sim,
                        'name': name,
                        'user_id': user_id
                    })
                
                self.result_queue.put(results)
            except queue.Empty:
                continue

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1) # Mirror
            self.frame_count += 1
            
            # Feed frame to processing thread every 3 frames if queue is empty
            if self.frame_count % 3 == 0 and self.frame_queue.empty():
                self.frame_queue.put(frame.copy())

            # Get latest results from queue
            try:
                while not self.result_queue.empty():
                    self.current_results = self.result_queue.get_nowait()
            except queue.Empty:
                pass

            current_time = time.time()
            anyone_detected = False
            anyone_recognized = False

            # Draw rectangles from results
            for res in self.current_results:
                sx, sy, sw, sh = res['rect']
                max_sim = res['sim']
                name = res['name']
                user_id = res['user_id']
                anyone_detected = True
                
                color = (0, 122, 255) if max_sim > THRESHOLD else (255, 59, 48)
                cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), color, 2)
                
                if max_sim > THRESHOLD:
                    anyone_recognized = True
                    self.unknown_buffer = 0 # Reset buffer saat kenal
                    self.update_ui_info(name, user_id)
                    if current_time > self.status_timer:
                        if has_attended_today(user_id):
                            self.trigger_status("SUDAH ABSEN", "#FF9500", "success")
                        else:
                            log_attendance(user_id, name)
                            self.trigger_status("BERHASIL", "#34C759", "success")
                        self.status_timer = current_time + COOLDOWN_TIME
                else:
                    # Tidak dikenal (mungkin berkedip)
                    if current_time > self.status_timer:
                        self.unknown_buffer += 1
                        # Jika sudah 8 frame (~1 detik) terus menerus tidak kenal
                        if self.unknown_buffer > 8:
                            self.trigger_status("TIDAK DIKENAL", "#FF3B30", "failure")
                            self.status_timer = current_time + (COOLDOWN_TIME // 2)

            # Jika tidak ada wajah sama sekali di layar
            if not anyone_detected and current_time > self.status_timer:
                self.reset_ui_info()

            self.display_image(frame)

        self.after(10, self.update_frame)

    def trigger_status(self, label, color, type_key):
        if self.current_status_type == label: return
        self.current_status_type = label
        
        if self.active_anim:
            self.active_anim.stop()
            self.active_anim.destroy()
        
        frames = self.preloaded_anims.get(type_key, [])
        if frames:
            self.active_anim = AnimationLabel(self.gif_container, frames)
            self.active_anim.pack()

    def reset_ui_info(self):
        self.current_status_type = None
        self.unknown_buffer = 0
        self.name_label.configure(text="---", text_color="#8E8E93")
        self.id_label.configure(text="---", text_color="#8E8E93")
        self.time_label.configure(text="---", text_color="#8E8E93")
        if self.active_anim:
            self.active_anim.stop()
            self.active_anim.destroy()
            self.active_anim = None

    def display_image(self, frame):
        # Convert OpenCV BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        
        # Use BILINEAR for a smoother look on video
        img = img.resize((self.cam_w, self.cam_h), Image.Resampling.BILINEAR)
        img.putalpha(self.camera_mask)
        
        tk_img = ImageTk.PhotoImage(img)
        self.camera_label.configure(image=tk_img)
        self.camera_label.image = tk_img

    def update_ui_info(self, name, user_id):
        if self.name_label.cget("text") != name:
            self.name_label.configure(text=name, text_color="#1C1C1E")
        if self.id_label.cget("text") != user_id:
            self.id_label.configure(text=user_id, text_color="#1C1C1E")
        
        last_time = get_last_attendance(user_id)
        if self.time_label.cget("text") != last_time:
            self.time_label.configure(text=last_time, text_color="#1C1C1E")

    def on_closing(self):
        self.stop_event.set()
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = FaceApp()
    app.mainloop()
