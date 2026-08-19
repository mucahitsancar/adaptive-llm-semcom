# third_party Kaynak Kaydı

Bu klasördeki repolar git'e commit edilmez (.gitignore); bu dosya hangi repo/commit
kullanıldığını kayıt altına alır. Klonlar ASLA değiştirilmez — gereken uyarlamalar
`src/` altındaki runner'larda yapılır.

## DeepSC (resmi PyTorch implementasyonu)
- URL: https://github.com/13274086/DeepSC
- Commit: `6ede3fdb696424ddb87936941fcd78aed94205f4` (2021-05-20, "update the readme file")
- Klon tarihi: 2026-08-05
- Lisans: repoda lisans dosyası YOK → yalnızca akademik reproduction için kullanılıyor, yeniden dağıtılmıyor.
- Bilinen sorunlar:
  - `dataset.py:17`, `main.py:102`, `performance.py:163`, `preprocess_text.py:146` sabit yol içerir (`/import/antennas/Datasets/hx301/`).
  - `requirements.txt` eski (`sklearn` meta-paketi, `bert4keras==0.4.2`); bert4keras yalnızca comment-out kodda kullanılıyor, kurulmadı.
  - `main.py:125` checkpoint mantığı her epoch kaydeder (record_acc her epoch sıfırlanıyor).

Yeniden klonlama: `git clone https://github.com/13274086/DeepSC third_party/DeepSC && cd third_party/DeepSC && git checkout 6ede3fdb`

## LLM_com (LLM-SC resmî implementasyonu)
- URL: https://github.com/gujianhunwang/LLM_com
- Commit: `22fac5662067ecf506f363be38673b3b37867772` (2025-12-25)
- Klon tarihi: 2026-08-11
- Lisans: Apache-2.0 (LICENSE dosyası mevcut)
- Yapı: yalnızca 4 dosya — README, `model_utils .py` (DOSYA ADINDA BOŞLUK; HF
  transformers generation/utils.py'nin modifiye kopyası, ~1158 satır),
  `test.ipynb` (tüm pipeline burada), LICENSE.
- Kritik bilgiler (test.ipynb'den):
  - LLM: **lmsys/vicuna-7b-v1.5** (Llama-2 tabanlı, 7B, HF'de public — gated değil)
  - Eğitim YOK — yalnız çıkarım; modifiye beam search (num_beams=10) içinde LLM
    önseli + kanal olabilirliği (8QAM yıldız kümesine Öklid uzaklığı / N0) birleşiyor
  - Fiziksel katman: token (vocab 32000) → 15 bit → 8QAM → 5 kompleks sembol/token;
    AWGN + Rayleigh (SEMBOL başına bağımsız H, perfect CSI); n_var=4.70/10^(SNR/10)
  - Veri: Europarl ama FARKLI filtre: 60-70 kelimelik cümleler, lowercase YOK,
    Llama tokenizer → 63.650 cümle; 16.327 "imkânsız token" maskeleniyor
  - Sabit yollar: model '/home/ai/Code/FastChat/...', veri '../FastChat/data_com'
  - `from model_utils import ...` dosya adındaki boşluk yüzünden ÇALIŞAMAZ →
    çalıştırma stratejisi: dosyanın adı düzeltilmiş kopyası src/ altına alınacak
    (third_party'ye dokunulmaz), transformers sürümü onların kopyaladığı API'ye
    pinlenecek (ayrı conda env: tez-llm).
