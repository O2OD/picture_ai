import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.neighbors import KNeighborsClassifier
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TeachableMachineClone:
    def __init__(self, root):
        self.root = root
        self.root.title("Teachable Machine - Asinxron o'qitish")
        
        self.model = load_model("keras_model.h5", compile=False)
        self.classes_h5 = [line.strip()[2:] for line in open("labels.txt", "r", encoding="utf-8").readlines()]
        
        self.extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.X, self.y = [], []
        self.is_trained = False
        
        self.cap = cv2.VideoCapture(0)
        
        self.video = tk.Label(root)
        self.video.pack()
        
        self.entry = tk.Entry(root, font=("Arial", 16))
        self.entry.pack(pady=5)
        
        tk.Button(root, text="Yangi obyekt qo'shish", command=self.add_class, bg="green", fg="white", font=("Arial", 12)).pack(pady=5)
        self.status = tk.Label(root, text="Tayyor", font=("Arial", 12))
        self.status.pack()
        
        self.frame_count = 0
        self.current_text = "Fon / Noma'lum"
        self.current_color = (0, 0, 255)
        
        # Yangi jarayon boshqaruvi
        self.is_capturing = False
        self.capture_count = 0
        self.target_name = ""
        
        self.update_frame()
        
    def add_class(self):
        name = self.entry.get().strip()
        if not name: return
        self.target_name = name
        self.capture_count = 0
        self.is_capturing = True # Jarayonni ishga tushirish
        self.status.config(text=f"[{name}] o'rganilmoqda... Obyektni aylantiring!", fg="red")
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_count += 1
            
            if self.is_capturing:
                # 1. Rasmga olish rejimi (Katta qizil yozuv chiqadi)
                cv2.putText(frame, f"RASM: {self.capture_count+1}/30", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                
                # Har 5-kadrni olamiz (Obyektni aylantirish uchun 5 sekund vaqt beradi)
                if self.frame_count % 5 == 0:
                    res = preprocess_input(np.expand_dims(cv2.resize(rgb, (224, 224)), axis=0))
                    feat = self.extractor.predict(res, verbose=0)
                    self.X.append(feat[0])
                    self.y.append(self.target_name)
                    self.capture_count += 1
                    
                # 30 ta rasm tugasa to'xtaydi va bazaga yozadi
                if self.capture_count >= 30:
                    self.is_capturing = False
                    self.knn.fit(self.X, self.y)
                    self.is_trained = True
                    self.entry.delete(0, tk.END)
                    self.status.config(text=f"Zo'r! [{self.target_name}] bazaga qo'shildi!", fg="green")
                    
            else:
                # 2. Oddiy ishlash rejimi (Skaner)
                if self.frame_count % 3 == 0:
                    res_h5 = np.asarray(cv2.resize(rgb, (224, 224)), dtype=np.float32).reshape(1, 224, 224, 3)
                    res_h5 = (res_h5 / 127.5) - 1
                    pred_h5 = self.model.predict(res_h5, verbose=0)
                    idx = np.argmax(pred_h5)
                    conf_h5 = pred_h5[0][idx]
                    
                    if conf_h5 > 0.85:
                        self.current_text = f"{self.classes_h5[idx]}"
                        self.current_color = (0, 255, 0)
                    elif self.is_trained:
                        res_knn = preprocess_input(np.expand_dims(cv2.resize(rgb, (224, 224)), axis=0))
                        feat = self.extractor.predict(res_knn, verbose=0)
                        self.current_text = f"{self.knn.predict(feat)[0]}"
                        self.current_color = (255, 255, 0)
                    else:
                        self.current_text = "Fon / Noma'lum"
                        self.current_color = (0, 0, 255)
                        
                cv2.putText(frame, self.current_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
                cv2.putText(frame, self.current_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.current_color, 2)
            
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video.imgtk = imgtk
            self.video.configure(image=imgtk)
            
        self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    TeachableMachineClone(root)
    root.mainloop()