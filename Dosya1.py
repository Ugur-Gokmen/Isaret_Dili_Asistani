import cv2
import mediapipe as mp
import numpy as np
import os

# NOT: 'test_el.jpg' dosyasını aynı klasöre koymayı unutmayın.
GÖRÜNTÜ_YOLU = 'test_el.jpg'

if not os.path.exists(GÖRÜNTÜ_YOLU):
    print(f"HATA: '{GÖRÜNTÜ_YOLU}' dosyası bulunamadı. Lütfen bir resim dosyası koyun.")
    exit()

# MediaPipe Hands modelini başlat
mp_hands = mp.solutions.hands
# 21 kilit noktasını çizmek için yardımcı çizim modülünü başlat
mp_drawing = mp.solutions.drawing_utils

# Eller modülünü oluştur. max_num_hands=1 diyerek tek el odaklı çalışıyoruz.
with mp_hands.Hands(
        static_image_mode=True,  # Resimler için en iyi modu seçer
        max_num_hands=1,  # Maksimum 1 el tespit et
        min_detection_confidence=0.5) as hands:  # Tespit güven aralığı

    # Görüntüyü oku
    image = cv2.imread(GÖRÜNTÜ_YOLU)

    # MediaPipe, BGR formatı yerine RGB formatında çalışır, bu yüzden dönüşüm yapıyoruz.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. İşaret dili verisini işle ve kilit noktalarını al
    results = hands.process(image_rgb)

    # El tespit edildiyse
    if results.multi_hand_landmarks:
        # Sadece ilk eli alıyoruz
        hand_landmarks = results.multi_hand_landmarks[0]

        # 3. Kilit noktalarını (Landmark) NumPy'a dönüştürme ve Analiz
        print("El Başarıyla Tespit Edildi!")
        print("-" * 30)

        # Tüm 21 kilit noktasını bir liste içinde tutmak için boş bir NumPy dizisi oluştur.
        # Her nokta için 3 koordinat (x, y, z) tutacağız.
        landmark_data = []

        # Her bir kilit noktasını (landmark) döngüye al
        for landmark in hand_landmarks.landmark:
            # x, y, z koordinatlarını listeye ekle
            # Koordinatlar 0 ile 1 arasında normalize edilmiştir (görüntünün oranı)
            landmark_data.append([landmark.x, landmark.y, landmark.z])

        # Listeyi NumPy dizisine çevir (21 kilit noktası x 3 koordinat = 21, 3 boyutlu dizi)
        landmark_array = np.array(landmark_data)

        # NumPy İncelemesi
        print(f"Kilit Noktası NumPy Dizisi Şekli: {landmark_array.shape}")
        print(f"Başlangıç Noktası (Bilek, Landmark 0) Koordinatları: {landmark_array[0]}")

        # # OPSİYONEL: Görüntü üzerine kilit noktalarını ve iskeleti çizme
        # annotated_image = image.copy()
        # mp_drawing.draw_landmarks(
        #     annotated_image,
        #     hand_landmarks,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        #     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # # cv2.imshow('Tespit Edilen El', annotated_image)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()

    else:
        print("Görüntüde el tespit edilemedi.")