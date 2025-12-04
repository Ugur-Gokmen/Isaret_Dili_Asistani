import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import os

# --- Ayarlar (Eğitimde kullandığınızla aynı olmalı) ---
SEQUENCE_LENGTH = 30
MODEL_PATH = 'C:/Users/PC/PycharmProjects/PythonProject1/isaret_dili_lstm_model.h5'
NUM_FEATURES = 42

# --- Etiketler (Eğitimde kullandığınız sırayla aynı olmalı) ---
# Modelin çıktısı olan 0, 1, 2 indekslerini gerçek anlama çeviririz.
ISARET_ETIKETLERI = {0: "YARDIM TALEP", 1: "ACIL DURUM", 2: "FATURA SORUNU"}

# Model Yükleme
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' başarıyla yüklendi.")
except Exception as e:
    print(
        f"HATA: Model yüklenemedi. Eğitim dosyanızdan 'isaret_dili_lstm_model.h5' dosyasının doğru yolda olduğundan emin olun.")
    print(e)
    exit()

# MediaPipe ve Sekans Yapısı
mp_hands = mp.solutions.hands
sequence = deque(maxlen=SEQUENCE_LENGTH)

# Canlı Tanıma Döngüsü Başlatılıyor
cap = cv2.VideoCapture(0)  # 0 = Varsayılan kamera

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    current_sign = "BASLANIYOR..."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Görüntüyü döndürme (Ayna görüntüsü için isteğe bağlı)
        frame = cv2.flip(frame, 1)

        # MediaPipe İşleme
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # El tespit edildiyse
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmark_frame = []

            for landmark in hand_landmarks.landmark:
                # Sadece X ve Y koordinatlarını al
                landmark_frame.extend([landmark.x, landmark.y])

            full_frame_array = np.array(landmark_frame)

            # 1. RÖLATİF KOORDİNATLARA DÖNÜŞTÜRME (Bileğe Göre Sıfırlama)
            bilek_koordinatlari = full_frame_array[0:2]
            bilek_tiled = np.tile(bilek_koordinatlari, 21)
            relative_frame_array = full_frame_array - bilek_tiled

            # Sekansa ekle
            sequence.append(relative_frame_array)

            # Görüntü üzerine MediaPipe çizimlerini ekle (opsiyonel)
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 2. TAHMİN: Yeterli kare toplandıysa
            if len(sequence) == SEQUENCE_LENGTH:
                input_seq = np.array(sequence)
                input_seq = np.expand_dims(input_seq, axis=0)  # Şekil: (1, 30, 42)

                predictions = model.predict(input_seq, verbose=0)[0]
                predicted_class_index = np.argmax(predictions)

                # Güven eşiği (%80) ile tanıma
                confidence = np.max(predictions)
                if confidence > 0.80:
                    current_sign = ISARET_ETIKETLERI[predicted_class_index]
                else:
                    current_sign = "Taninmiyor..."

                # Sekansı temizle veya kısalt (İşaret tanındıktan sonra akışı devam ettirmek için)
                # Yeni bir sekans toplamaya başla.
                sequence.clear()

        # Ekrana Bilgi Yazdırma (Müşteri Hizmetleri Ara yüzünü temsil eder)
        cv2.putText(frame, "GSM ASISTANI", (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, f"ISARET: {current_sign}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, f"GECERLI ISARETLER: {', '.join(ISARET_ETIKETLERI.values())}", (20, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Canli Isaret Dili Tanima Sistemi', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
