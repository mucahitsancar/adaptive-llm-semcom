# -*- coding: utf-8 -*-
"""
KARAR KURALI KIYASI — iki hakem itirazına birlikte cevap.

İtiraz A: "Politika basit (ağırlıklı argmax), yöntem değil mühendislik."
İtiraz B: "Adaptif bir baseline'a karşı kıyas yok, sadece sabit konfigürasyonlar var."

Cevap: aynı ölçümler üzerinde beş karar kuralını yarıştırmak. Eğer bizim kural
alternatiflerden belirgin daha iyi ve kâhine yakınsa, basitliği bir kusur değil
erdem olur — çünkü daha karmaşık olmadan yeterince iyi.

Kurallar (hepsi AYNI aksiyon kümesi ve AYNI fayda tanımıyla):
  1. SABIT-EN-IYI      : Γ_fit'te ortalama faydası en yüksek TEK konfigürasyonu
                         seç, tüm SNR'larda onu kullan (adaptif olmayan baseline)
  2. DOGRULUK-OBURU    : her SNR'da en doğru aksiyonu seç, maliyeti yok say
                         (adaptif ama maliyet-kör baseline)
  3. ESIK-SEZGISI      : bir T eşiği altında en ucuz kodek, üstünde en doğru LLM.
                         T, Γ_fit üzerinde en iyi ortalama faydayı verecek şekilde
                         seçilir (naif adaptif baseline)
  4. BIZIM POLITIKA    : Γ_fit'te parçalı sabit karar bölgeleri, Γ_val'de
                         değiştirilmeden uygulanır
  5. KAHIN (ORACLE)    : her SNR'da fayda-en-iyi aksiyon (üst sınır)

Ölçüt: Γ_val = {3,9,15} dB'de kâhine göre ortalama fayda kaybı (regret).
Kural 5 tanım gereği 0; diğerleri ne kadar kaybediyor?
"""
import argparse
import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"

spec = importlib.util.spec_from_file_location("dp", ROOT / "src" / "derive_policy.py")
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)

parser = argparse.ArgumentParser()
parser.add_argument("--lam", type=float, default=0.03, help="maliyet toleransı (λ₁=λ₂)")
parser.add_argument("--mem-max", type=float, default=6.0)
parser.add_argument("--out", default=str(RES / "policy_comparison.csv"))


def main():
    args = parser.parse_args()
    acc = dp.load_accuracy()
    cost = dp.load_cost()
    if not cost:
        raise SystemExit("gecikme olcumu yok")
    ncost = dp.normalise(cost)
    eligible = dp.eligible_actions(acc, cost, args.mem_max, 1e9, None)
    lam = args.lam

    def U(a, ch, snr):
        if a not in eligible or (a, ch, snr) not in acc:
            return None
        d, m = ncost[a]
        return acc[(a, ch, snr)] - lam * d - lam * m

    FIT, VAL = dp.FIT_SNRS, dp.VAL_SNRS
    satirlar = []

    for ch in ("AWGN", "Rayleigh"):
        # --- kural 1: Γ_fit'te ortalama faydası en iyi TEK konfigürasyon
        ort = {}
        for a in eligible:
            v = [U(a, ch, s) for s in FIT]
            v = [x for x in v if x is not None]
            if len(v) == len(FIT):
                ort[a] = sum(v) / len(v)
        sabit = max(ort, key=ort.get) if ort else None

        # --- kural 3: eşiği Γ_fit üzerinde seç
        kodekler = [a for a in eligible if a.startswith("DeepSC")]
        llmler = [a for a in eligible if not a.startswith("DeepSC")]
        # ONEMLI: esik kurali test SNR'indaki dogruluga BAKAMAZ (kopya olur).
        # Her iki taraf icin de sabit bir aksiyon, YALNIZ Gamma_fit'ten secilir.
        def _fit_en_iyi(havuz, snrlar):
            skor = {}
            for a in havuz:
                v = [U(a, ch, x) for x in snrlar]
                v = [y for y in v if y is not None]
                if v:
                    skor[a] = sum(v) / len(v)
            return max(skor, key=skor.get) if skor else None

        def esik_kural(T, s, sabit_alt=None, sabit_ust=None):
            return (sabit_alt if s < T else sabit_ust)
        en_iyi_T, en_iyi_skor, en_iyi_alt, en_iyi_ust = None, -9e9, None, None
        for T in (0, 3, 6, 9, 12, 15, 18, 21):
            alt_snr = [x for x in FIT if x < T]
            ust_snr = [x for x in FIT if x >= T]
            a_alt = _fit_en_iyi(kodekler, alt_snr) if alt_snr else None
            a_ust = _fit_en_iyi(llmler, ust_snr) if ust_snr else None
            v = []
            for s in FIT:
                a = a_alt if s < T else a_ust
                u = U(a, ch, s) if a else None
                if u is not None:
                    v.append(u)
            if len(v) == len(FIT) and sum(v) / len(v) > en_iyi_skor:
                en_iyi_skor, en_iyi_T = sum(v) / len(v), T
                en_iyi_alt, en_iyi_ust = a_alt, a_ust

        # --- kural 4: bizim politika (Γ_fit'te argmax, en yakın noktanın kararı)
        pol = {}
        for s in FIT:
            c = {a: U(a, ch, s) for a in eligible}
            c = {a: u for a, u in c.items() if u is not None}
            if c:
                pol[s] = max(c, key=c.get)

        # --- Γ_val'de hepsini sına
        for s in VAL:
            mevcut = {a: U(a, ch, s) for a in eligible}
            mevcut = {a: u for a, u in mevcut.items() if u is not None}
            if not mevcut:
                continue
            kahin = max(mevcut, key=mevcut.get)
            u_kahin = mevcut[kahin]

            # doğruluk-oburu
            sadece_acc = {a: acc[(a, ch, s)] for a in eligible if (a, ch, s) in acc}
            obur = max(sadece_acc, key=sadece_acc.get) if sadece_acc else None

            # eşik sezgisi
            esik_a = (esik_kural(en_iyi_T, s, en_iyi_alt, en_iyi_ust)
                      if en_iyi_T is not None else None)

            # BIZIM-POLITIKA (v1, naif): en yakin fit noktasinin karari.
            # Kusur: Gamma_val noktalari fit noktalarinin TAM ORTASI oldugu icin
            # her birinde beraberlik olusuyor ve min() daima ALT noktayi seciyor.
            yakin = min(pol, key=lambda x: abs(x - s)) if pol else None
            bizim = pol.get(yakin)

            # BIZIM-POLITIKA (v2, komsu-ortalamali): beraberligi keyfi cozmek
            # yerine, s'yi kusatan iki fit noktasindaki ORTALAMA faydayi en
            # buyuten aksiyonu sec. Hala YALNIZ Gamma_fit bilgisi kullaniliyor.
            alt = max([x for x in FIT if x <= s], default=None)
            ust = min([x for x in FIT if x >= s], default=None)
            komsular = [x for x in (alt, ust) if x is not None]
            skor2 = {}
            for a in eligible:
                v = [U(a, ch, x) for x in komsular]
                v = [y for y in v if y is not None]
                if len(v) == len(komsular):
                    skor2[a] = sum(v) / len(v)
            bizim2 = max(skor2, key=skor2.get) if skor2 else None

            for ad, a in (("SABIT-EN-IYI", sabit), ("KAHIN-DOGRULUK", obur),
                          ("ESIK-SEZGISI", esik_a), ("BIZIM-POLITIKA-v1", bizim), ("BIZIM-POLITIKA-v2", bizim2),
                          ("KAHIN", kahin)):
                u = mevcut.get(a)
                satirlar.append({
                    "lam": lam, "kanal": ch, "snr": s, "kural": ad,
                    "secim": a or "-",
                    "fayda": round(u, 4) if u is not None else "",
                    "regret": round(u_kahin - u, 4) if u is not None else "",
                    "bleu1": round(acc[(a, ch, s)], 4) if a and (a, ch, s) in acc else "",
                })

    # --- özet
    from collections import defaultdict
    g = defaultdict(list)
    for r in satirlar:
        if r["regret"] != "":
            g[r["kural"]].append(float(r["regret"]))
    print(f"KARAR KURALI KIYASI  (λ={lam}, M_max={args.mem_max} GB, "
          f"Γ_val={VAL} dB, iki kanal)\n")
    print(f"{'Kural':18s} {'n':>3s} {'ort regret':>11s} {'en kotu':>9s}  yorum")
    print("-" * 74)
    yorum = {
        "KAHIN": "KOPYA CEKER: fayda ust siniri (tanim geregi 0)",
        "BIZIM-POLITIKA-v1": "naif: en yakin fit noktasi (beraberlikte alt secilir)",
        "BIZIM-POLITIKA-v2": "komsu-ortalamali: kusatan iki fit noktasinin ortalamasi",
        "ESIK-SEZGISI": "konuslanabilir: T ve iki aksiyon YALNIZ Gamma_fit ten",
        "KAHIN-DOGRULUK": "KOPYA CEKER: test SNR dogrulugunu gorur, konuslanabilir DEGIL",
        "SABIT-EN-IYI": "adaptif DEGIL: tek konfigurasyon her yerde",
    }
    print("[konuslanabilir kurallar: Gamma_fit bilgisiyle karar verir]")
    for k in ("BIZIM-POLITIKA-v2", "BIZIM-POLITIKA-v1", "ESIK-SEZGISI", "SABIT-EN-IYI"):
        v = g.get(k, [])
        if not v:
            continue
        print(f"{k:18s} {len(v):3d} {sum(v)/len(v):11.4f} {max(v):9.4f}  {yorum[k]}")
    print()
    print("[ust sinirlar: tutulan SNR olcumunu gorur, kural degil referans]")
    for k in ("KAHIN", "KAHIN-DOGRULUK"):
        v = g.get(k, [])
        if v:
            print(f"{k:18s} {len(v):3d} {sum(v)/len(v):11.4f} {max(v):9.4f}  {yorum[k]}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
        w.writeheader(); w.writerows(satirlar)
    print(f"\n{len(satirlar)} satir -> {Path(args.out).name}")


if __name__ == "__main__":
    main()
