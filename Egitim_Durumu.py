import matplotlib.pyplot as plt
import numpy as np

# Model Veri Seti (Gerçek TÜİK verisi değildir, eğilimleri temsil eder)
# Değerler: Her yaş grubundaki işitme engelli bireylerin oransal dağılımı (%)
# Eğitim Düzeyleri: Okuma/Yazma Bilmeyen, İlkokul/Ortaokul, Lise, Yükseköğretim

yas_gruplari = ['15-24', '25-44', '45-64', '65+']

# İşitme Engelliler İçin Model Dağılımlar (%)
# Her liste sırasıyla 'Okuma/Yazma Bilmeyen', 'İlkokul/Ortaokul', 'Lise', 'Yükseköğretim' oranlarını temsil eder.
okur_yazar_degil = [2, 5, 15, 30]  # Yaşlandıkça okuma yazma bilmeyen oranı artar
ilk_ortaokul = [20, 35, 60, 55]    # Orta yaşlarda yoğunlaşma
lise = [40, 35, 20, 10]            # Genç/orta yaşlarda yüksek
yuksekogretim = [38, 25, 5, 5]     # Genç yaşlarda en yüksek

# Verilerin hazırlanması
veriler = np.array([okur_yazar_degil, ilk_ortaokul, lise, yuksekogretim])

# Yığılmış Sütun Grafiği
fig, ax = plt.subplots(figsize=(10, 6))

genislik = 0.50
bottom = np.zeros(len(yas_gruplari))

# Renkler ve Etiketler
renkler = ['#e67e22', '#3498db', '#2ecc71', '#9b59b6']
egitim_seviyeleri = ['Okuma/Yazma Bilmeyen', 'İlkokul/Ortaokul', 'Lise', 'Yükseköğretim']

for i in range(len(egitim_seviyeleri)):
    p = ax.bar(yas_gruplari, veriler[i], genislik, label=egitim_seviyeleri[i], bottom=bottom, color=renkler[i])
    bottom += veriler[i]

    # Her sütuna toplam değeri ekleme (Opsiyonel)
    # for rect in p:
    #     height = rect.get_height()
    #     ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2),
    #                 xytext=(0, 0), textcoords="offset points", ha='center', va='center', fontsize=8, color='white')


# Başlık ve Eksenler
ax.set_title('İşitme Engelli Bireylerin Yaş Grubuna Göre Eğitim Durumu Dağılımı (Model Veri)', fontsize=14, pad=20)
ax.set_ylabel('Oran (%)', fontsize=12)
ax.set_xlabel('Yaş Grubu', fontsize=12)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(egitim_seviyeleri))
ax.set_yticks(np.arange(0, 101, 10)) # Y eksenini 0'dan 100'e ayarlama
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()