# -*- coding: utf-8 -*-
"""POLİTİKAYI BİR KEZ YENİDEN TÜRET — ölçümler tamamlandığında.

Neden tek bir betik: politika sayıları makalenin en önemli sayıları ve gece
boyunca üç kez bayatladı (0.0108 → 0.0130 → ?). Her ölçüm damladığında elle
yeniden hesaplamak yerine, veri tamamlandığında tek seferde koşulacak bir hat
kuruyoruz.

ÖN KOŞUL: politikanın aksiyon uzayındaki her konfigürasyon her iki kanalda da
yedi SNR noktasında ölçülmüş olmalı. Değilse betik ÇALIŞMAZ ve neyin eksik
olduğunu söyler — eksik veriyle türetilmiş bir politika sessizce yanlış olur.

ÇIKTI: policy_table.csv, policy_validation.csv, policy_comparison.csv,
policy_actions.csv + makaleye girecek sayıların özeti.
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
PY = sys.executable

TAM = [0, 3, 6, 9, 12, 15, 18]

# politikanin aksiyon uzayi: 6 GB'a sigan konfigurasyonlar + kodek
AKSIYON = [
    ("DeepSC", "deepsc-AWGN", "deepsc-Rayleigh"),
    ("DeepSC-widesnr", "deepsc-AWGN-widesnr", "deepsc-Rayleigh-widesnr"),
    ("Qwen-0.8B-FP16", "qwen08b-AWGN", "qwen08b-Rayleigh"),
    ("Qwen-2B-FP16", "qwen2b-AWGN", "qwen2b-Rayleigh"),
    ("Qwen-2B-INT8", "qwen2b-int8-AWGN", "qwen2b-int8-Rayleigh"),
    ("Qwen-2B-INT4", "qwen2b-int4-AWGN", "qwen2b-int4-Rayleigh"),
    ("Qwen-4B-INT4", "qwen4b-int4-AWGN", "qwen4b-int4-Rayleigh"),
]


def snr_kumesi(prefix):
    f = RES / f"{prefix}_eval.csv"
    if not f.exists():
        return None
    return {int(float(r["snr"])) for r in csv.DictReader(f.open(encoding="utf-8"))
            if r.get("bleu1")}


def on_kosul():
    eksikler = []
    for ad, a, r in AKSIYON:
        for kanal, pre in (("AWGN", a), ("Rayleigh", r)):
            s = snr_kumesi(pre)
            if s is None:
                eksikler.append(f"{ad} {kanal}: KOŞU YOK ({pre})")
                continue
            eks = [x for x in TAM if x not in s]
            if eks:
                eksikler.append(f"{ad} {kanal}: eksik {eks}")
    yukleme = (RES / "model_load_time.csv").exists()
    return eksikler, yukleme


def kos(ad, cmd):
    print(f"\n--- {ad} ---", flush=True)
    # ENCODING: alt surecin ciktisi Turkce/matematik karakteri iceriyor;
    # varsayilan cp1254 ile cozulemez ve UnicodeDecodeError verir.
    ortam = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=ortam)
    cikti = (r.stdout or "") + (r.stderr or "")
    for satir in cikti.splitlines()[-25:]:
        print("   " + satir)
    if r.returncode:
        print(f"   !! HATA (çıkış {r.returncode})")
    return r.returncode == 0


def ozet():
    """Makaleye girecek sayıları topla."""
    print("\n" + "=" * 62)
    print("MAKALEYE GİRECEK SAYILAR")
    print("=" * 62)

    f = RES / "policy_validation.csv"
    if f.exists():
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        reg = [float(r["regret"]) for r in rows if r.get("regret")]
        if reg:
            print(f"  tutulan değerlendirme sayısı : {len(reg)}")
            print(f"  ORTALAMA REGRET              : {sum(reg)/len(reg):.4f}")
            print(f"  en kötü tek durum            : {max(reg):.4f}")
            lam = {}
            for r in rows:
                k = r.get("lam_d")
                if k and r.get("regret"):
                    lam.setdefault(k, []).append(float(r["regret"]))
            for k in sorted(lam, key=float):
                v = lam[k]
                print(f"    lambda={k}: ortalama {sum(v)/len(v):.4f} ({len(v)} nokta)")

    f = RES / "policy_comparison.csv"
    if f.exists():
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        kural = {}
        for r in rows:
            if r.get("kural") and r.get("regret"):
                kural.setdefault((r["kural"], r.get("lam", "?")), []).append(
                    float(r["regret"]))
        print("\n  KURAL KARŞILAŞTIRMASI (ortalama regret)")
        for (k, l), v in sorted(kural.items()):
            print(f"    {k:24s} lambda={l:6s} {sum(v)/len(v):.4f}  ({len(v)} nokta)")

        # oranlar
        bizim = {l: sum(v)/len(v) for (k, l), v in kural.items()
                 if "BIZIM" in k.upper() or "POLITIKA" in k.upper()}
        print("\n  ORANLAR (alternatif / bizim)")
        for (k, l), v in sorted(kural.items()):
            if l in bizim and bizim[l] > 0 and "KAHIN" not in k.upper() \
               and "BIZIM" not in k.upper() and "POLITIKA" not in k.upper():
                print(f"    {k:24s} lambda={l:6s} {(sum(v)/len(v))/bizim[l]:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zorla", action="store_true",
                    help="ön koşul sağlanmasa da koş (ÖNERİLMEZ)")
    a = ap.parse_args()

    print("ÖN KOŞUL KONTROLÜ")
    eksikler, yukleme = on_kosul()
    for e in eksikler:
        print(f"  EKSİK: {e}")
    print(f"  temiz yükleme ölçümü: {'var' if yukleme else 'YOK'}")

    if eksikler and not a.zorla:
        print("\nDURDURULDU: eksik veriyle türetilen politika sessizce yanlış olur.")
        print("Ölçümler tamamlanınca tekrar koşturun.")
        return 1
    if eksikler:
        print("\n!! --zorla ile devam ediliyor, sonuçlar EKSİK VERİYE dayanıyor")

    print("\nBütün ön koşullar sağlandı, türetiliyor.\n" if not eksikler else "")

    tamam = kos("politika türetimi (lambda taraması)",
                [PY, "src/derive_policy.py",
                 "--sweep", "0.00,0.01,0.03,0.08,0.20",
                 "--tie-lambdas", "--mem-max", "6.0",
                 "--latency-csv", str(RES / "latency_bench.csv")])
    if not tamam:
        print("\nTüretim başarısız, karşılaştırmaya geçilmiyor.")
        return 1

    for lam in ("0.01", "0.03", "0.08"):
        kos(f"kural karşılaştırması lambda={lam}",
            [PY, "src/compare_policies.py", "--lam", lam, "--mem-max", "6.0"])

    ozet()
    print("\nSonraki adım: makaledeki \\pending{} işaretlerini bu sayılarla doldur.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
