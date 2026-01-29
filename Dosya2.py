import cv2
import mediapipe as mp
import numpy as np
import os

VİDEO_YOLU = 'ornek_isaret.mp4'

if not os.path.exists(VİDEO_YOLU):
    print(f"HATA: '{VİDEO_YOLU}' dosyası bulunamadı. Lütfen bir video dosyası (örneğin el hareketi) koyun.")
    exit()

mp_hands = mp.solutions.hands
sequence_data = []  # Tüm videonun kilit noktası serilerini tutacak liste


with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    cap = cv2.VideoCapture(VİDEO_YOLU)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Video bittiğinde döngüden çık

       
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
        image_rgb.flags.writeable = False

       
        results = hands.process(image_rgb)

        
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

           
            landmark_frame = []
            for landmark in hand_landmarks.landmark:
                
                landmark_frame.extend([landmark.x, landmark.y])

            
            sequence_data.append(np.array(landmark_frame))
            frame_count += 1

      

    cap.release()
    # cv2.destroyAllWindows()


if sequence_data:
    full_sequence_array = np.array(sequence_data)

    print("--- Video İşleme Sonuçları ---")
    print(f"İşlenen Toplam Kare Sayısı (El Tespit Edilen): {frame_count}")
    print(f"Oluşturulan Zaman Serisi NumPy Şekli: {full_sequence_array.shape}")
else:
    print("Video işlenemedi veya hiçbir karede el tespit edilemedi.")

if sequence_data:
    full_sequence_array = np.array(sequence_data)

    print("--- Video İşleme Sonuçları ---")
    print(f"Oluşturulan Zaman Serisi NumPy Şekli: {full_sequence_array.shape}")
   
    bilek_koordinatlari = full_sequence_array[:, 0:2]  # Shape: (493, 2)
    bilek_tiled = np.tile(bilek_koordinatlari, 21)  # Shape: (493, 42)

    relative_sequence_array = full_sequence_array - bilek_tiled

    print("\n--- Ön İşleme Sonuçları (Rölatif Koordinatlar) ---")
    print(f"Rölatif Dizi Şekli: {relative_sequence_array.shape}")

    print(f"İlk 5 karenin Yeni Bilek Koordinatları (0'a yakın olmalı):\n{relative_sequence_array[:5, 0:2]}")

else:

    print("Video işlenemedi veya hiçbir karede el tespit edilemedi.")
