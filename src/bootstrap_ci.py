# -*- coding: utf-8 -*-
"""
Cümle düzeyi bootstrap güven aralıkları + EŞLEŞTİRİLMİŞ karşılaştırma.

Neden gerekli: makale "2B ile 4B arasında ölçülebilir kazanç yok" ya da
"nicemleme doğruluğu bozmuyor" gibi KÜÇÜK farklara dayanan iddialar içeriyor
(0.005-0.02 BLEU). Tek sayı verip geçmek, hakemin "bu gürültü değil mi?"
sorusuna açık kapı bırakır.

Neden yeni tohum koşmaya gerek yok: her nokta için 150 cümlenin TEK TEK çıktısı
kayıtlı (`<prefix>_texts/`). Baskın varyans kaynağı cümleden cümleye değişim ve
bu doğrudan örneklenebiliyor. Üstelik karşılaştırılan konfigürasyonlar AYNI 150
cümleyi gördüğü için EŞLEŞTİRİLMİŞ test yapılabilir: cümle başına farkların
dağılımı incelenir. Eşleştirme, ortak cümle zorluğunu sadeleştirdiği için
eşleştirilmemiş testten belirgin biçimde güçlüdür.

Kanal gürültüsü tohumundan gelen varyans ayrı bir bileşendir ve bu yöntemle
ölçülmez; makalede böyle belirtilir. Ancak hakemin sorduğu soru "bu iki sistem
arasındaki fark bu test setinde anlamlı mı" sorusudur ve eşleştirilmiş bootstrap
tam onu cevaplar.

Kullanım:
  python src/bootstrap_ci.py --prefix qwen2b-AWGN --snr 6
  python src/bootstrap_ci.py --karsilastir qwen2b-AWGN qwen4b-fp16-AWGN --snr 6
  python src/bootstrap_ci.py --tablo          # makaledeki kritik iddiaları sına
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
from nltk.translate.bleu_score import sentence_bleu

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
W1 = (1, 0, 0, 0)

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default=None)
parser.add_argument("--karsilastir", nargs=2, default=None, metavar=("A", "B"))
parser.add_argument("--snr", type=int, default=6)
parser.add_argument("--kip", default="auto", choices=["auto", "content", "index"],
                    help="eşleştirme kipi (--karsilastir için); paralel korpusta index")
parser.add_argument("--B", type=int, default=10000, help="bootstrap tekrarı")
parser.add_argument("--tablo", action="store_true",
                    help="makaledeki kritik küçük-fark iddialarını topluca sına")
parser.add_argument("--out", default=str(RES / "bootstrap_ci.csv"))


def temizle(x):
    x = x.replace("<START>", " ").replace("<END>", " ").replace("<PAD>", " ")
    return re.sub(r"\s+", " ", x).strip()


def cumle_bleu(prefix, snr):
    """(hedef_cumle -> BLEU-1) sözlüğü + sıralı vektör.

    Sözlük DÖNMESİ ŞART: DeepSC hattı çıktıları uzunluğa göre gruplanmış sırada
    yazıyor, LLM hattı ise dosya sırasında. Bu yüzden satır indeksiyle eşleştirmek
    DeepSC↔LLM kıyasında YANLIŞ çiftler üretiyor (ölçüldü: 300 satırın yalnız 9'u
    aynı). Eşleştirme hedef cümle METNİ üzerinden yapılmalı.
    """
    d = RES / f"{prefix}_texts"
    ft, fd = d / f"snr{snr}_target.txt", d / f"snr{snr}_decoded.txt"
    if not (ft.exists() and fd.exists()):
        return None
    t = [temizle(x) for x in ft.read_text(encoding="utf-8").splitlines() if x.strip()]
    h = [temizle(x) for x in fd.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(t) != len(h):
        # HIZA BOZUK: model satir sonu uretmis olabilir (maske kapali kosular).
        # Bu durumda cumle bazli analiz GECERSIZ; sessizce yanlis sayi uretmek
        # yerine reddediyoruz. Dogru degerler koşunun kendi CSV'sinde.
        print(f"  !! {prefix} @ {snr} dB: satir hizasi bozuk "
              f"(hedef {len(t)}, cozum {len(h)}) -> cumle bazli analiz YAPILAMAZ")
        return None
    n = len(t)
    sozluk = {}
    for i in range(n):
        b = sentence_bleu([h[i].split()], t[i].split(), weights=W1)
        sozluk.setdefault(t[i], []).append(b)   # yinelenen cümle olursa liste
    return {"vec": np.array([sentence_bleu([h[i].split()], t[i].split(), weights=W1)
                             for i in range(n)]),
            "map": {k: float(np.mean(v)) for k, v in sozluk.items()},
            "sirali": t[:n]}


def hizala(A, B, kip="auto"):
    """İki koşuyu eşleştir. Nasıl eşleştiğini de döner.

    kip="auto"    : satır sırası birebir aynıysa indeksle, değilse cümle metniyle
    kip="index"   : ZORUNLU indeks eşleştirmesi. PARALEL KORPUS için gerekli:
                    Türkçe ve İngilizce cümleler metin olarak eşleşmez ama
                    i. satırlar birbirinin çevirisidir (doğrulandı: TR[0]
                    "çok teşekkür ederim chris" ↔ EN[0] "thank you so much, chris").
                    İçerik eşleştirmesi burada 0 kesişim verir ve testi düşürür.
    kip="content" : ZORUNLU içerik eşleştirmesi
    """
    if kip == "index":
        n = min(len(A["vec"]), len(B["vec"]))
        return A["vec"][:n], B["vec"][:n], f"indeks-zorunlu ({n} cumle, paralel korpus)"
    if kip == "auto" and A["sirali"] == B["sirali"]:
        return A["vec"], B["vec"], "satir-sirasi"
    ortak = [k for k in A["map"] if k in B["map"]]
    if not ortak:
        return None, None, "eslesmedi"
    a = np.array([A["map"][k] for k in ortak])
    b = np.array([B["map"][k] for k in ortak])
    return a, b, f"icerik ({len(ortak)} cumle)"


def ci(v, B, rng):
    """Ortalamanın %95 bootstrap güven aralığı."""
    idx = rng.integers(0, len(v), size=(B, len(v)))
    ort = v[idx].mean(axis=1)
    return v.mean(), np.percentile(ort, 2.5), np.percentile(ort, 97.5)


def eslestirilmis(a, b, B, rng):
    """Eşleştirilmiş iki vektörün farkı: CI + iki yanlı p (işaret permütasyonu)."""
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    idx = rng.integers(0, n, size=(B, n))
    ort = d[idx].mean(axis=1)
    lo, hi = np.percentile(ort, 2.5), np.percentile(ort, 97.5)
    # işaret permütasyonu: H0 = fark dağılımı 0 etrafında simetrik
    isaret = rng.choice([-1.0, 1.0], size=(B, n))
    boş = (d * isaret).mean(axis=1)
    p = float((np.abs(boş) >= abs(d.mean())).mean())
    return d.mean(), lo, hi, p, n


# (etiket, A, B, snr, beklenen, eslestirme_kipi)
IDDIALAR = [
    ("olcek 2B->4B", "qwen2b-AWGN", "qwen4b-fp16-AWGN", 6, "kazanc yok", "auto"),
    ("olcek 2B->4B", "qwen2b-AWGN", "qwen4b-fp16-AWGN", 9, "kazanc yok", "auto"),
    ("nicemleme FP16->INT8", "qwen2b-AWGN", "qwen2b-int8-AWGN", 6, "kayip yok", "auto"),
    ("nicemleme FP16->INT4", "qwen2b-AWGN", "qwen2b-int4-AWGN", 6, "kayip yok", "auto"),
    ("esit bellek 2BFP16-4BINT4", "qwen2b-AWGN", "qwen4b-int4-AWGN", 0, "FP16 ustun", "auto"),
    ("esit bellek 2BFP16-4BINT4", "qwen2b-AWGN", "qwen4b-int4-AWGN", 6, "fark yok", "auto"),
    ("2B-INT4 vs 0.8B-FP16", "qwen2b-int4-AWGN", "qwen08b-AWGN", 6, "INT4 ustun", "auto"),
    # MANSET KIYAS — eslesmis 300 cumle; DeepSC uzunluga gore siraladigi icin
    # ICERIK eslestirmesi zorunlu (satir indeksi yanlis cift uretir)
    ("MANSET kodek vs 2B @0dB", "M300-widesnr300", "qwen2b-AWGN", 0, "kodek ustun", "content"),
    ("MANSET kodek vs 2B @3dB", "M300-widesnr300", "qwen2b-AWGN", 3, "kodek ustun", "content"),
    ("MANSET kodek vs 2B @6dB", "M300-base300", "qwen2b-AWGN", 6, "LLM ustun", "content"),
    ("MANSET kodek vs 2B @9dB", "M300-base300", "qwen2b-AWGN", 9, "LLM ustun", "content"),
    ("MANSET kodek vs 2B @12dB", "M300-base300", "qwen2b-AWGN", 12, "LLM ustun", "content"),
    ("iyilestirme base->widesnr @0", "M300-base300", "M300-widesnr300", 0, "widesnr ustun", "auto"),
    ("Rayleigh perH iyilestirmesi @3", "M300-base300R", "M300-perH300R", 3, "perH ustun", "auto"),
    # PARALEL KORPUS: farkli dil, icerik eslesmez ama satir i'ler birbirinin cevirisi
    ("dil TR vs EN (0.8B)", "TR-qwen08b", "ENpar-qwen08b", 0, "TR daha kotu", "index"),
    ("dil TR vs EN (0.8B)", "TR-qwen08b", "ENpar-qwen08b", 6, "TR daha kotu", "index"),
]


def main():
    args = parser.parse_args()
    rng = np.random.default_rng(12345)

    if args.tablo:
        satirlar = []
        print(f"{'iddia':30s} {'SNR':>4s} {'fark':>8s} {'%95 CI':>20s} {'p':>8s} "
              f"{'sonuc':>14s}  eslestirme")
        print("-" * 108)
        for etiket, A, B_, snr, iddia, kip in IDDIALAR:
            A_, B__ = cumle_bleu(A, snr), cumle_bleu(B_, snr)
            if A_ is None or B__ is None:
                print(f"{etiket:30s} {snr:4d}   metin yok, atlandi")
                continue
            a, b, nasil = hizala(A_, B__, kip)
            if a is None:
                print(f"{etiket:30s} {snr:4d}   eslesmedi, atlandi")
                continue
            fark, lo, hi, p, n = eslestirilmis(a, b, args.B, rng)
            anlamli = not (lo <= 0 <= hi)
            sonuc = "ANLAMLI" if anlamli else "ayirt edilemez"
            print(f"{etiket:30s} {snr:4d} {fark:+8.4f} [{lo:+7.4f},{hi:+7.4f}] "
                  f"{p:8.4f} {sonuc:>14s}  {nasil}")
            satirlar.append({"iddia": etiket, "A": A, "B": B_, "snr": snr,
                             "beklenen": iddia, "fark": round(fark, 4),
                             "ci_alt": round(lo, 4), "ci_ust": round(hi, 4),
                             "p": round(p, 4), "n": n, "eslestirme": nasil,
                             "sonuc": "anlamli" if anlamli else "ayirt_edilemez"})
        if satirlar:
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
                w.writeheader(); w.writerows(satirlar)
            print(f"\n{len(satirlar)} iddia → {Path(args.out).name}")
        return

    if args.karsilastir:
        A, B_ = args.karsilastir
        A_, B__ = cumle_bleu(A, args.snr), cumle_bleu(B_, args.snr)
        if A_ is None or B__ is None:
            raise SystemExit("metin bulunamadi")
        a, b, nasil = hizala(A_, B__, args.kip)
        if a is None:
            raise SystemExit("eslesmedi")
        fark, lo, hi, p, n = eslestirilmis(a, b, args.B, rng)
        print(f"{A} vs {B_} @ {args.snr} dB (n={n}, eslestirme: {nasil})")
        print(f"  {A}: {a.mean():.4f}   {B_}: {b.mean():.4f}")
        print(f"  fark {fark:+.4f}  %95 CI [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}")
        print("  ->", "ANLAMLI fark" if not (lo <= 0 <= hi) else "AYIRT EDILEMEZ")
        return

    r = cumle_bleu(args.prefix, args.snr)
    if r is None:
        raise SystemExit("metin bulunamadi")
    v = r["vec"]
    m, lo, hi = ci(v, args.B, rng)
    print(f"{args.prefix} @ {args.snr} dB: {m:.4f}  %95 CI [{lo:.4f}, {hi:.4f}]  n={len(v)}")


if __name__ == "__main__":
    main()
