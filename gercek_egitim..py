import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Sabitler ---
# Veri Toplayici'da kaydettiğiniz ana klasör adı
DATA_PATH = 'isaret_verisi_npy'
# Tanımlamak istediğiniz etiketler
ACTIONS = ["YARDIM", "FATURA", "ACIL"]
# Önceki model mimarinizdeki sekans uzunluğu
SEQUENCE_LENGTH = 30
# Özellik sayısı: 21 kilit noktası * 2 koordinat (x, y)
NUM_FEATURES = 42

# Etiketleri Sayısal Hale Çeviren Sözlük
label_map = {label: num for num, label in enumerate(ACTIONS)}


# Örn: {'YARDIM': 0, 'FATURA': 1, 'ACIL': 2}

# --- 1. Gerçek Veriyi Yükleme Fonksiyonu ---
def load_real_data(data_path, actions, sequence_length):
    X, y = [], []

    # Her etiket klasörünü döngüye al
    for action in actions:
        # Klasördeki tüm .npy dosyalarını bul
        path = os.path.join(data_path, action, '*.npy')
        npy_files = glob.glob(path)

        print(f"Yükleniyor: {action} ({len(npy_files)} dosya bulundu)")

        for npy_file in npy_files:
            try:
                # NPY dosyasını yükle (Şekil: (Kare Sayısı, 42))
                sequence = np.load(npy_file)

                # Modelimiz 30 kareden oluşan sekansta çalışıyor (SEQUENCE_LENGTH)
                # Yeterli kareye sahip olanları alalım
                if len(sequence) >= sequence_length:

                    # Veri büyütme (Data Augmentation) veya kaydırma (Sliding Window)
                    # Bu, modelin farklı başlangıç noktalarını öğrenmesini sağlar.
                    # 5 adımlık kaydırma ile sekansları alıyoruz.
                    for start in range(0, len(sequence) - sequence_length, 5):
                        end = start + sequence_length
                        # 30 kareden oluşan sekansı X'e ekle
                        X.append(sequence[start:end])
                        # Etiketini (0, 1, 2) y'ye ekle
                        y.append(label_map[action])

            except Exception as e:
                print(f"Hata oluştu: {npy_file} - {e}")

    return np.array(X), np.array(y)


# --- 2. Model Mimarisi (Aynı Kalsın) ---
def build_lstm_model(sequence_length, num_features, num_classes):
    model = Sequential([
        # İlk LSTM katmanı: return_sequences=True (bir sonraki LSTM için çıktı dizisi döndür)
        LSTM(64, return_sequences=True, input_shape=(sequence_length, num_features)),
        Dropout(0.2),
        # İkinci LSTM katmanı: return_sequences=False (son tahmin için tek bir çıktı vektörü döndür)
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        # Yoğun (Dense) katmanlar
        Dense(64, activation='relu'),
        # Çıkış katmanı: Softmax ile 3 sınıf (YARDIM, FATURA, ACİL) için olasılık dağılımı
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# --- 3. Ana Eğitim Akışı ---
if __name__ == '__main__':
    # Gerçek veriyi yükle
    X, y = load_real_data(DATA_PATH, ACTIONS, SEQUENCE_LENGTH)

    if len(X) == 0:
        print("\n!!! HATA: Yüklenen toplam sekans sayısı 0. Lütfen NPY dosyalarınızı kontrol edin. !!!")
    else:
        # Etiketleri tek sıcaklık kodlamasına (One-Hot Encoding) çevir
        y_one_hot = tf.keras.utils.to_categorical(y, num_classes=len(ACTIONS))

        print(f"\n--- Gerçek Veri Yükleme Başarılı ---")
        print(f"Toplam Girdi Sekansı (X) Şekli: {X.shape}")
        print(f"Toplam Çıktı Etiket (Y) Şekli: {y_one_hot.shape}")

        # Eğitim ve Test Verisini Ayırma (%80 Eğitim, %20 Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.20, random_state=42)

        print(f"Eğitim Sekansı Şekli: {X_train.shape}")

        # Modeli oluştur ve özetini göster
        model = build_lstm_model(SEQUENCE_LENGTH, NUM_FEATURES, len(ACTIONS))
        print("\n--- Model Mimarisi ---")
        model.summary()

        # Modeli Eğit
        print("\n--- Model Eğitimi Başlatılıyor (Gerçek Veri ile) ---")
        history = model.fit(
            X_train, y_train,
            epochs=20,  # Eğitim döngüsü sayısı
            batch_size=32,
            validation_data=(X_test, y_test)  # Eğitimi doğrulama için ayrılan test verisi
        )

        # Eğitilmiş Modeli Kaydetme
        MODEL_SAVE_PATH = 'isaret_dili_model.h5'
        model.save(MODEL_SAVE_PATH)
        print(f"\nModel başarıyla kaydedildi: {MODEL_SAVE_PATH}")

        # Modelin Test Verisi Üzerindeki Başarısını Değerlendirme
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Verisi Doğruluğu: {accuracy * 100:.2f}%")