import matplotlib.pyplot as plt
import numpy as np

# Veriler (TÜİK Türkiye Sağlık Araştırması 2019 - 15 yaş ve üzeri işitme sorunu olanların oranı)
yas_gruplari = ['15-44 Yaş', '45-54 Yaş', '55-64 Yaş', '65-74 Yaş', '75+ Yaş']
oranlar_toplam = [4.1, 3.9, 5.2, 11.9, 31.5]
oranlar_erkek = [3.0, 3.6, 4.8, 12.4, 30.8]
oranlar_kadin = [5.2, 4.2, 5.5, 11.5, 32.0]

# Grafik ayarları
fig, ax = plt.subplots(figsize=(10, 6))
bar_genisligi = 0.25
x = np.arange(len(yas_gruplari)) # Yaş grupları için etiket pozisyonları

# Sütunları çizme
rects1 = ax.bar(x - bar_genisligi, oranlar_erkek, bar_genisligi, label='Erkek', color='#1f77b4')
rects2 = ax.bar(x, oranlar_kadin, bar_genisligi, label='Kadın', color='#ff7f0e')
rects3 = ax.bar(x + bar_genisligi, oranlar_toplam, bar_genisligi, label='Toplam', color='#2ca02c')

# Başlık ve eksen etiketleri
ax.set_title('İşitme Sorunu Olan Bireylerin Yaş Grubuna ve Cinsiyete Göre Dağılımı (%)', fontsize=14, pad=20)
ax.set_ylabel('İşitme Sorunu Oranı (%)', fontsize=12)
ax.set_xlabel('Yaş Grubu', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(yas_gruplari)
ax.legend(loc='upper left')

# Veri etiketlerini sütunların üzerine ekleme
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 point vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Grid çizgilerini ekleme
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Grafiği gösterme
plt.tight_layout()
plt.show()