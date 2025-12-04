import numpy as np
import cv2
import mediapipe as mp
import os
import glob  # Dosya yollarını aramak için yeni kütüphane

# MediaPipe modelini global olarak tanımla
mp_hands = mp.solutions.hands
# hands nesnesini with bloğu dışında bir kez tanımlayalım
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def process_and_save_sequence(video_path, output_dir, sign_name, video_num):
    """
    Bir video dosyasını işler, kilit noktası serilerini çıkarır,
    rölatif koordinatlara dönüştürür ve .npy dosyası olarak kaydeder.

    sign_name (str): İşaretin adı (örn: 'yardim')
    video_num (int): İşaretin sıra numarası (örn: 1, 2, 3...)
    """
    if not os.path.exists(video_path):
        print(f"HATA: Video bulunamadı: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    sequence_data = []

    # MediaPipe el tespitini döngü boyunca kullan
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # İşleme adımları aynı kalır
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # 'hands' nesnesini with bloğu dışında tanımladığımız için direkt kullanabiliriz.
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmark_frame = []

            for landmark in hand_landmarks.landmark:
                # Sadece X ve Y koordinatlarını al (Z'yi model mimarinize uygun tutalım)
                # Not: Önceki kodunuzda Z'yi atlamıştınız. Eğer modeliniz 42 öz. istiyorsa (21*2), bu uygundur.
                landmark_frame.extend([landmark.x, landmark.y])

            sequence_data.append(np.array(landmark_frame))

    cap.release()

    if not sequence_data:
        print(f"UYARI: {video_path} içinde el tespit edilemedi veya sequence 0 uzunlukta. İŞLEM ATLANDI.")
        return None

    # --- Ön İşleme ve Rölatif Koordinat Dönüşümü ---
    full_sequence_array = np.array(sequence_data)
    bilek_koordinatlari = full_sequence_array[:, 0:2]

    # 21 (landmark sayısı) * 2 (x, y) = 42 özellik için döşeme
    bilek_tiled = np.tile(bilek_koordinatlari, 21)
    relative_sequence_array = full_sequence_array - bilek_tiled

    # --- Kilit Noktası: Sıralı Dosya Adını Oluşturma ---
    # video_num'ı sıfırla doldurularak 2 haneli stringe çeviriyoruz: 1 -> '01', 10 -> '10'
    formatted_num = f"{video_num:02d}"

    # Yeni dosya adı (örn: yardim_01.npy)
    new_filename = f"{sign_name}_{formatted_num}.npy"

    # Kayıt Yolu: isaret_verisi/YARDIM/yardim_01.npy
    save_path = os.path.join(output_dir, sign_name.upper(), new_filename)

    # Klasörü kontrol et ve yoksa oluştur (Örn: isaret_verisi/YARDIM)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # NumPy dizisini kaydet
    np.save(save_path, relative_sequence_array)

    print(f"Başarıyla kaydedildi: {save_path} (Şekil: {relative_sequence_array.shape})")
    return save_path


# ======================================================================
# --- ANA VERİ TOPLAMA DÖNGÜSÜ ---
# ======================================================================

OUTPUT_DATA_DIR = 'isaret_verisi_npy'  # Kayıt klasörünü daha belirgin yaptık
INPUT_VIDEO_DIR = 'video_verilerim'  # .mp4 dosyalarının bulunduğu ana klasör

# 1. Tanımlanan İşaret Etiketleri
ISARET_ETIKETLERI = ["yardim", "fatura", "acil"]  # Video adlarının ön eki olmalı!

# Lütfen .mp4 videolarınızı aşağıdaki gibi bir klasör yapısına yerleştirin:
# video_verilerim/
# ├── yardim_01.mp4
# ├── yardim_02.mp4
# ├── fatura_01.mp4
# └── acil_01.mp4

# Tüm video dosyalarını bul ve işle
for sign_name in ISARET_ETIKETLERI:

    # 2. İşaret Adı ile Başlayan Tüm Videoları Bul
    # Örn: 'video_verilerim/yardim_*.mp4'
    search_pattern = os.path.join(INPUT_VIDEO_DIR, f"{sign_name}_*.mp4")
    video_files = sorted(glob.glob(search_pattern))  # Alfabetik sıralama (01, 02, 10...)

    if not video_files:
        print(f"UYARI: '{sign_name}' etiketi için hiç video bulunamadı: {search_pattern}")
        continue

    print(f"\n--- {sign_name.upper()} ({len(video_files)} Video) İşaret İşleniyor ---")

    # 3. Her Video için Sıralı İşleme ve Kaydetme
    for i, video_path in enumerate(video_files):
        # i+1: 0'dan değil, 1'den başlayan sıra numarası (01, 02, ...)
        video_num = i + 30
        print(f"({video_num}/{len(video_files)}) İşleniyor: {os.path.basename(video_path)}")

        process_and_save_sequence(video_path, OUTPUT_DATA_DIR, sign_name, video_num)

print("\n=======================================================")
print("Tüm videolar başarıyla işlendi ve NPY dosyaları kaydedildi.")
print("=======================================================")