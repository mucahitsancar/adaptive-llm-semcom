# -*- coding: utf-8 -*-
"""
Kaydedilmiş cümle metinlerinden eksik `_eval.csv` satırlarını geri hesapla.

Neden var: `<prefix>_texts/snrX_{target,decoded}.txt` dosyaları ASIL DELİLDİR ve
her SNR noktası bittiğinde tek tek yazılır. `_eval.csv` ise koşunun sonunda
yazılır. Dolayısıyla koşu yarıda kesilirse (kill, oturum kopması, çökme) ölçüm
yapılmış ama satır kaydedilmemiş olabilir. Bu durum 18 Ağustos'ta üç kez yaşandı:
  1. LONG-qwen08b — 4 nokta (üzerine yazma)
  2. LONG-deepsc-retrained — 4 nokta (üzerine yazma)
  3. qwen2b-int4-AWGN 3 dB — iş durdurulduğunda CSV yazılmamış
Her üçünde de metinler sağlam kaldığı için kayıp kalıcı olmadı.

Kullanım:
  python src/rebuild_eval_from_texts.py                 # tüm koşuları tara, eksikleri bildir
  python src/rebuild_eval_from_texts.py --yaz           # eksik satırları CSV'ye ekle
  python src/rebuild_eval_from_texts.py --prefix qwen2b-int4-AWGN --yaz

Not: yalnız EKSİK satırlar eklenir; var olan satırlara dokunulmaz. Doğrulama modu
(`--dogrula`) mevcut satırları metinlerden yeniden hesaplayıp karşılaştırır.
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
from nltk.translate.bleu_score import sentence_bleu

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"

W = {1: (1, 0, 0, 0), 2: (0.5, 0.5, 0, 0),
     3: (1 / 3, 1 / 3, 1 / 3, 0), 4: (0.25,) * 4}

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default=None, help="tek koşu (varsayılan: hepsi)")
parser.add_argument("--yaz", action="store_true", help="eksik satırları CSV'ye ekle")
parser.add_argument("--dogrula", action="store_true",
                    help="mevcut satırları da metinlerden yeniden hesaplayıp karşılaştır")
parser.add_argument("--tol", type=float, default=5e-4)


def temizle(x):
    """DeepSC hattının protokol tokenlarını at. <UNK> KORUNUR: gerçek bir
    sözlük kısıtını temsil ediyor, atmak kodeği haksız avantajlı gösterir."""
    x = x.replace("<START>", " ").replace("<END>", " ").replace("<PAD>", " ")
    return re.sub(r"\s+", " ", x).strip()


def bleu(tdir, snr):
    t = [temizle(x) for x in (tdir / f"snr{snr}_target.txt").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    d = [temizle(x) for x in (tdir / f"snr{snr}_decoded.txt").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    if len(t) != len(d):
        print(f"    !! satir hizasi bozuk (hedef {len(t)}, cozum {len(d)}) -> atlandi")
        return None, 0
    n = len(t)
    if n == 0:
        return None, 0
    out = {k: float(np.mean([sentence_bleu([d[i].split()], t[i].split(), weights=W[k])
                             for i in range(n)])) for k in (1, 2, 3, 4)}
    return out, n


def main():
    args = parser.parse_args()
    dirs = ([RES / f"{args.prefix}_texts"] if args.prefix
            else sorted(RES.glob("*_texts")))
    toplam_eksik = toplam_fark = 0

    for tdir in dirs:
        pref = tdir.name[:-6]
        snrs = sorted(int(m.group(1)) for f in tdir.glob("snr*_decoded.txt")
                      if (m := re.match(r"snr(\d+)_decoded", f.name)))
        if not snrs:
            continue
        csv_p = RES / f"{pref}_eval.csv"
        satir = {}
        alanlar = None
        if csv_p.exists():
            rows = list(csv.DictReader(open(csv_p, encoding="utf-8")))
            if rows:
                alanlar = list(rows[0].keys())
                for r in rows:
                    try:
                        satir[int(float(r["snr"]))] = r
                    except (TypeError, ValueError, KeyError):
                        pass
        eksik = [s for s in snrs if s not in satir]
        if not (eksik or args.dogrula):
            continue

        print(f"\n{pref}")
        print(f"  metinde {snrs} | CSV'de {sorted(satir)}")
        if eksik:
            toplam_eksik += len(eksik)
            print(f"  EKSIK SATIR: {eksik}")

        yeni_satirlar = {}
        for s in (eksik if not args.dogrula else snrs):
            b, n = bleu(tdir, s)
            if b is None:
                print(f"    {s} dB: metin bos, atlandi")
                continue
            if s in satir and args.dogrula:
                try:
                    fark = b[1] - float(satir[s]["bleu1"])
                except (TypeError, ValueError, KeyError):
                    continue
                isaret = "OK" if abs(fark) < args.tol else "FARKLI"
                if abs(fark) >= args.tol:
                    toplam_fark += 1
                print(f"    {s:2d} dB: CSV {float(satir[s]['bleu1']):.4f} | "
                      f"metin {b[1]:.4f} | fark {fark:+.5f} {isaret}")
            else:
                print(f"    {s:2d} dB: BLEU-1 {b[1]:.4f}  B4 {b[4]:.4f}  (n={n})  "
                      f"→ metinden geri hesaplandı")
                yeni_satirlar[s] = (b, n)

        if args.yaz and yeni_satirlar:
            if alanlar is None:
                alanlar = ["snr", "bleu1", "bleu2", "bleu3", "bleu4", "n"]
            for s, (b, n) in yeni_satirlar.items():
                r = {a: "" for a in alanlar}
                r["snr"] = s
                for k in (1, 2, 3, 4):
                    if f"bleu{k}" in r:
                        r[f"bleu{k}"] = round(b[k], 4)
                if "n" in r:
                    r["n"] = n
                # kaynağı belli olsun: geri hesaplanan satır işaretlenir
                if "kaynak" not in alanlar:
                    alanlar.append("kaynak")
                r["kaynak"] = "metinden_geri_hesap"
                satir[s] = r
            for r in satir.values():
                r.setdefault("kaynak", "")
            with open(csv_p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=alanlar, restval="")
                w.writeheader()
                for s in sorted(satir):
                    w.writerow(satir[s])
            print(f"  → {len(yeni_satirlar)} satır yazıldı: {csv_p.name}")

    print(f"\n{'='*58}")
    print(f"eksik satır: {toplam_eksik}" + ("" if args.yaz else "  (--yaz ile eklenir)"))
    if args.dogrula:
        print(f"uyuşmayan mevcut satır: {toplam_fark}")


if __name__ == "__main__":
    main()
