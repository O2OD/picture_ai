import cv2
import numpy as np
from tensorflow.keras.models import load_model  # Kerasni TensorFlow ichidan chaqiramiz

# Terminalda ortiqcha qizil yozuvlar chiqmasligi uchun
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r", encoding="utf-8").readlines()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera ishlamayapti!")
        break

    image_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1

    prediction = model.predict(image_array, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()[2:]
    confidence_score = prediction[0][index]

    text = f"{class_name} ({int(confidence_score * 100)}%)"
    
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Shaxsiy AI Loyiha", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()