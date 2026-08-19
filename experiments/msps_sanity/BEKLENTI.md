# MSPS sağlamlık testi — beklentiler

Metriğin kendi geçerliliğini sınayan sabit test kümesi. Satırlar hizalıdır.

| # | Hedef | Çözülen | Bozulma | MSPS beklentisi |
|---|---|---|---|---|
| 1 | gelmeyeceğim | geleceğim | olumsuzluk silindi | DÜŞÜK |
| 2 | çözemem | çözerim | yeterlik+olumsuzluk silindi | DÜŞÜK |
| 3 | dün gittim | yarın gideceğim | zaman tersine | DÜŞÜK |
| 4 | biz izledik | ben izledim | kişi değişti | DÜŞÜK |
| 5 | koydum | koydum | bozulma yok | 1.0 |
| 6 | katılamayacağım | gelemeyeceğim | eşanlamlı ikame, morfoloji AYNI | YÜKSEK |

6. satır tasarım gereği yüksek olmalı: sözcük seçimini BLEU ölçüyor, MSPS'in işi
dilbilgisel anlam taşıyıcılarının korunumu (bkz. src/msps.py başlığı).
Toplamda beklenen: genel MSPS düşük-orta, `kritik_tersinme_orani` yüksek.
