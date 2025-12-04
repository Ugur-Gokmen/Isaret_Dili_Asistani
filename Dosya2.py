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

# Eller modelini video işleme modu için başlat
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

        # BGR'den RGB'ye dönüşüm
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Görüntüyü çevirmeyi engelle (sadece okuma)
        image_rgb.flags.writeable = False

        # Kilit noktalarını işle
        results = hands.process(image_rgb)

        # Görüntüyü tekrar yazılabilir yap ve BGR'ye dönüştür
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # El tespit edildiyse
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # 1. Kilit Noktalarını NumPy dizisine çevir
            landmark_frame = []
            for landmark in hand_landmarks.landmark:
                # Sadece (x, y) koordinatlarını alalım (z, derinlik bilgisi, şimdilik atlanabilir)
                landmark_frame.extend([landmark.x, landmark.y])

            # NumPy dizisi olarak listeye ekle
            sequence_data.append(np.array(landmark_frame))
            frame_count += 1

        # Eğer canlı görüntülemek isterseniz bu kısmı açabilirsiniz
        # cv2.imshow('Frame', image)
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows()

# Tüm video serisini tek bir NumPy dizisine dönüştür
if sequence_data:
    full_sequence_array = np.array(sequence_data)

    print("--- Video İşleme Sonuçları ---")
    print(f"İşlenen Toplam Kare Sayısı (El Tespit Edilen): {frame_count}")
    # Şekil: (Toplam Kare Sayısı, Kilit Noktası Sayısı * Koordinat Sayısı) => (Kare, 21 * 2 = 42)
    print(f"Oluşturulan Zaman Serisi NumPy Şekli: {full_sequence_array.shape}")
else:
    print("Video işlenemedi veya hiçbir karede el tespit edilemedi.")

if sequence_data:
    full_sequence_array = np.array(sequence_data)

    print("--- Video İşleme Sonuçları ---")
    print(f"Oluşturulan Zaman Serisi NumPy Şekli: {full_sequence_array.shape}")

    # -------------------------------------------------------------
    # GÖREV 2: Rölatif (Göreli) Koordinat Hesaplama
    # -------------------------------------------------------------

    # Her karedeki bilek noktasının (Landmark 0) (x, y) koordinatlarını al.
    # Bilek noktası ilk 2 sütundur (0. ve 1. index)
    bilek_koordinatlari = full_sequence_array[:, 0:2]  # Shape: (493, 2)

    # Bilek koordinatlarını, tüm kilit noktalarından çıkarmak için matrisi genişletelim.
    # 493 x 2 olan bilek koordinatlarını, 493 x 42'lik ana matrisin her 2 sütunu için tekrar etmeli.

    # NumPy'ın reshape ve tile fonksiyonları ile bu çıkarma işlemini hazırlıyoruz.
    # Bilek koordinatları (x_b, y_b) -> [x_b, y_b, x_b, y_b, ..., x_b, y_b] (42 sütun)
    # tile, 42/2 = 21 defa tekrar et demektir.
    bilek_tiled = np.tile(bilek_koordinatlari, 21)  # Shape: (493, 42)

    # Ana matristen bilek koordinatlarını çıkar
    relative_sequence_array = full_sequence_array - bilek_tiled

    print("\n--- Ön İşleme Sonuçları (Rölatif Koordinatlar) ---")
    print(f"Rölatif Dizi Şekli: {relative_sequence_array.shape}")

    # Bilek noktasının yeni koordinatları (sıfırlanmış mı?) kontrol edelim:
    # İlk 5 karenin bilek (Landmark 0) koordinatları (0. ve 1. sütun)
    print(f"İlk 5 karenin Yeni Bilek Koordinatları (0'a yakın olmalı):\n{relative_sequence_array[:5, 0:2]}")

else:
    print("Video işlenemedi veya hiçbir karede el tespit edilemedi.")