# -*- coding: utf-8 -*-
"""
POLİTİKA TÜRETİMİ (ana katkı) — kanal ve kaynak farkında çıkarım politikası.

Girdi:
  - experiments/results/*_eval.csv        → doğruluk F(a, γ, κ)
  - experiments/results/latency_bench.csv → temiz gecikme D(a) + bellek M(a)
Çıktı:
  - experiments/results/policy_table.csv  → her (kanal, SNR, bütçe) için seçim
  - experiments/results/policy_validation.csv → tutulan SNR'larda pişmanlık (regret)
  - figures/fig8_politika.png             → politika haritası + doğruluk-maliyet sınırı

Yöntem (makale §IV):
  U(a,γ,κ) = F(a,γ,κ) − λ₁·D̃(a) − λ₂·M̃(a),   M(a) ≤ M_max, D(a) ≤ D_max
  π(γ,κ) = argmax_a U
  Genelleme testi: politika Γ_fit={0,6,12,18} dB üzerinde kurulur (parçalı sabit
  karar bölgeleri), Γ_val={3,9,15} dB üzerinde sınanır; ölçüt = seçilen aksiyonun
  doğruluğu ile o SNR'daki en iyi aksiyonun doğruluğu arasındaki fark (regret).
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"

# aksiyon adı → (eval prefix şablonu, gecikme tablosundaki ad)
ACTIONS = {
    "DeepSC":        ("deepsc-{ch}",            "deepsc"),
    "DeepSC-improved": ("deepsc-{ch}-{imp}",    "deepsc"),
    "Qwen-0.8B-FP16": ("qwen08b-{ch}",          "qwen0.8b-fp16"),
    "Qwen-2B-FP16":  ("qwen2b-{ch}",            "qwen2b-fp16"),
    "Qwen-2B-INT8":  ("qwen2b-int8-{ch}",       "qwen2b-int8"),
    "Qwen-2B-INT4":  ("qwen2b-int4-{ch}",       "qwen2b-int4"),
    "Qwen-4B-INT4":  ("qwen4b-int4-{ch}",       "qwen4b-int4"),
    "Qwen-4B-FP16":  ("qwen4b-fp16-{ch}",       "qwen4b-fp16"),
}
IMPROVED = {"AWGN": "widesnr", "Rayleigh": "perH"}
FIT_SNRS = [0, 6, 12, 18]
VAL_SNRS = [3, 9, 15]

parser = argparse.ArgumentParser()
parser.add_argument("--lam-d", type=float, default=0.10, help="gecikme cezası λ₁")
parser.add_argument("--lam-m", type=float, default=0.10, help="bellek cezası λ₂")
parser.add_argument("--mem-max", type=float, default=6.0, help="bellek bütçesi (GB)")
parser.add_argument("--lat-max", type=float, default=1e9, help="gecikme bütçesi (sn/cümle)")
parser.add_argument("--latency-csv", default=None,
                    help="gecikme tablosu yolu (prova/duyarlılık analizi için; "
                         "varsayılan experiments/results/latency_bench.csv)")
parser.add_argument("--tie-lambdas", action="store_true",
                    help="taramada bellek agirligini gecikme agirligina esitle "
                         "(tek bir 'maliyet toleransi' parametresi); aksi halde "
                         "lam2 sabit kalir ve bellek cezasi karari domine eder")
parser.add_argument("--sweep", default=None,
                    help="λ₁ çalışma noktaları listesi, ör. 0.02,0.10,0.30 "
                         "(gecikme toleransı arttıkça karar bölgeleri kayar)")
parser.add_argument("--tag", default="", help="çıktı dosyalarına ek etiket (prova için)")


def load_accuracy():
    """{(aksiyon, kanal, snr): bleu1}"""
    acc = {}
    for name, (tmpl, _) in ACTIONS.items():
        for ch in ("AWGN", "Rayleigh"):
            pref = tmpl.format(ch=ch, imp=IMPROVED[ch])
            p = RES / f"{pref}_eval.csv"
            if not p.exists():
                continue
            for r in csv.DictReader(open(p, encoding="utf-8")):
                acc[(name, ch, int(float(r["snr"])))] = float(r["bleu1"])
    return acc


def load_cost(path=None):
    """{aksiyon: (gecikme_sn_cumle, bellek_gb, gecikme_sn_token)}

    Ceza terimi CÜMLE BAŞINA süredir. Gerekçe: tüm konfigürasyonlar aynı test
    cümlelerinde ölçüldüğü için cümle başına değer aileler arasında doğrudan
    karşılaştırılabilir. Token başına değer ceza olarak KULLANILAMAZ, çünkü
    DeepSC kelime düzeyinde, LLM hattı alt-kelime düzeyinde tokenleştiriyor —
    iki ailenin "token"ı aynı şey değil. Token başına süre yine okunur ve
    raporlanır (aynı aile içindeki ölçek kıyası ve tanı için).

    `median_s_per_token` kolonunun VARLIĞI, ölçümün temiz protokolle (tek süreç,
    sessiz GPU, ısınma turlu) alındığının işaretidir. v1 dosyasında bu kolon
    yoktur → ölçüm güvenilmez sayılıp hiç kullanılmaz (bkz. LOG 2026-08-18).
    """
    cost = {}
    p = Path(path) if path else RES / "latency_bench.csv"
    if not p.exists():
        return cost
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows or "median_s_per_token" not in rows[0]:
        print("UYARI: latency_bench.csv v1 formatında (token başına kolon yok). "
              "Bu ölçüm güvenilmez kabul edilip KULLANILMIYOR.")
        return cost
    raw = {}
    for r in rows:
        try:
            raw[r["config"]] = (float(r["median_s"]), float(r["vram_gb"]),
                                float(r["median_s_per_token"]))
        except (TypeError, ValueError, KeyError):
            continue                      # boş/bozuk/eksik satırı atla
    for name, (_, key) in ACTIONS.items():
        if key in raw:
            cost[name] = raw[key]
    return cost


def normalise(cost):
    """min-max normalize edilmiş maliyet terimleri."""
    if not cost:
        return {}
    ds = [c[0] for c in cost.values()]
    ms = [c[1] for c in cost.values()]
    dmin, dmax = min(ds), max(ds)
    mmin, mmax = min(ms), max(ms)
    span = lambda v, lo, hi: 0.0 if hi == lo else (v - lo) / (hi - lo)
    return {k: (span(v[0], dmin, dmax), span(v[1], mmin, mmax)) for k, v in cost.items()}


def eligible_actions(acc, cost, mem_max, lat_max, res_path=None):
    """Politikanın seçebileceği aksiyon kümesi + dışlama gerekçeleri.

    KRİTİK: maliyeti ÖLÇÜLMEMİŞ bir aksiyon dışlanır. Aksi halde cezası sıfır
    kabul edilir, 'bedava' görünür ve politikayı haksız kazanır — 4B-FP16 ve
    4B-INT8 yerel 6 GB'a sığmadığı için ölçülemedi, bu tuzağa tam denk geliyor.
    Dışlamalar sessizce yapılmaz; dosyaya yazılır.
    """
    have_acc = {a for (a, _, _) in acc}
    ok, dropped = set(), []
    for a in ACTIONS:
        if a not in have_acc:
            dropped.append((a, "doğruluk ölçümü yok"))
        elif a not in cost:
            dropped.append((a, "gecikme/bellek ölçümü yok → cezası 0 sayılamaz"))
        elif cost[a][1] > mem_max:
            dropped.append((a, f"bellek {cost[a][1]:.2f} GB > bütçe {mem_max} GB"))
        elif cost[a][0] > lat_max:
            dropped.append((a, f"gecikme {cost[a][0]:.2f} sn/cümle > bütçe {lat_max}"))
        else:
            ok.add(a)
    if dropped:
        print("\n=== POLİTİKA DIŞI BIRAKILAN AKSİYONLAR (gerekçeli) ===")
        for a, why in dropped:
            print(f"  {a:18s} — {why}")
    if res_path:
        with open(res_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["aksiyon", "durum", "gerekce"])
            for a in sorted(ok):
                w.writerow([a, "dahil", ""])
            for a, why in dropped:
                w.writerow([a, "dislandi", why])
    return ok


def derive(acc, cost, ncost, eligible, lam_d, lam_m):
    """Politikayı FIT ızgarasında kur, VAL ızgarasında sına.
    Döner: (policy, fit_satirlari, dogrulama_satirlari)"""

    def utility(a, ch, snr):
        if a not in eligible or (a, ch, snr) not in acc:
            return None
        d, m = ncost[a]
        return acc[(a, ch, snr)] - lam_d * d - lam_m * m

    policy, rows = {}, []
    for ch in ("AWGN", "Rayleigh"):
        for snr in FIT_SNRS:
            cands = [(a, utility(a, ch, snr)) for a in ACTIONS]
            cands = [(a, u) for a, u in cands if u is not None]
            if not cands:
                continue
            best = max(cands, key=lambda x: x[1])
            policy[(ch, snr)] = best[0]
            c = cost[best[0]]
            rows.append({"lam_d": lam_d, "lam_m": lam_m, "kanal": ch, "snr": snr,
                         "secim": best[0], "fayda": round(best[1], 4),
                         "bleu1": round(acc[(best[0], ch, snr)], 4),
                         "gecikme_sn_cumle": c[0], "gecikme_sn_token": c[2],
                         "bellek_gb": c[1]})

    val, atlanan = [], []
    for ch in ("AWGN", "Rayleigh"):
        for snr in VAL_SNRS:
            fit_pts = [s for s in FIT_SNRS if (ch, s) in policy]
            if not fit_pts:
                continue
            # ATAMA KURALI: s'yi kuşatan iki fit noktasındaki ORTALAMA faydayı
            # en büyüten aksiyon. "En yakın fit noktası" naif alternatifi
            # Γ_val noktaları fit noktalarının tam ortasında olduğu için her
            # doğrulama noktasında beraberlik üretiyor ve keyfî biçimde alt
            # noktayı seçiyordu; bu, λ=0.03'te regreti 0.0108'den 0.0250'ye
            # çıkarıyordu (bkz. src/compare_policies.py). Komşu-ortalamalı kural
            # yalnız Γ_fit bilgisi kullanır, yani genelleme testi bozulmaz.
            alt = max([x for x in fit_pts if x <= snr], default=None)
            ust = min([x for x in fit_pts if x >= snr], default=None)
            komsu = [x for x in (alt, ust) if x is not None]
            skor = {}
            for a in eligible:
                v = [utility(a, ch, x) for x in komsu]
                v = [y for y in v if y is not None]
                if len(v) == len(komsu):
                    skor[a] = sum(v) / len(v)
            if not skor:
                continue
            chosen = max(skor, key=skor.get)
            avail = {a: acc[(a, ch, snr)] for a in eligible if (a, ch, snr) in acc}
            if chosen not in avail:
                # SESSIZ KIRPMA YOK: politikanin sectigi aksiyonun bu tutulan
                # SNR'da olcumu yok -> nokta dogrulanamiyor, gerekcesiyle bildir.
                atlanan.append((ch, snr, chosen))
                continue
            # ASIL ÖLÇÜT: regret, politikanın optimize ettiği FAYDA üzerinden.
            # Saf doğrulukla ölçmek tutarsız olur — politika bilerek doğruluk feda
            # edip gecikme/bellek kazanıyor; onu yalnız doğrulukla yargılamak amaç
            # fonksiyonunu değiştirip aynı kararı 'hata' saymak olurdu.
            util = {a: u for a, u in ((x, utility(x, ch, snr)) for x in avail)
                    if u is not None}
            u_or = max(util, key=util.get)
            acc_or = max(avail, key=avail.get)
            val.append({"lam_d": lam_d, "lam_m": lam_m, "kanal": ch, "snr": snr,
                        "politika_secimi": chosen,
                        "politika_fayda": round(util[chosen], 4),
                        "en_iyi_fayda_secimi": u_or, "en_iyi_fayda": round(util[u_or], 4),
                        "regret": round(util[u_or] - util[chosen], 4),
                        "politika_bleu1": round(avail[chosen], 4),
                        "en_dogru_secim": acc_or,
                        "en_yuksek_bleu1": round(avail[acc_or], 4),
                        "dogruluk_farki": round(avail[acc_or] - avail[chosen], 4)})
    return policy, rows, val, atlanan


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    args = parser.parse_args()
    acc, cost = load_accuracy(), load_cost(args.latency_csv)
    if not cost:
        print("DURDURULDU: kullanılabilir gecikme/bellek ölçümü yok. Maliyet terimi "
              "olmadan türetilen 'politika' yalnız doğruluk sıralamasıdır ve ana "
              "katkıyı temsil etmez. Temiz ölçüm (v2) bitince yeniden koş.")
        raise SystemExit(1)
    ncost = normalise(cost)
    eligible = eligible_actions(acc, cost, args.mem_max, args.lat_max,
                                RES / f"policy_actions{args.tag}.csv")
    if not eligible:
        raise SystemExit("Hiçbir aksiyon bütçeye uymuyor.")

    lams = ([float(x) for x in args.sweep.split(",")] if args.sweep
            else [args.lam_d])
    all_rows, all_val = [], []
    for lam_d in lams:
        lam_m = lam_d if args.tie_lambdas else args.lam_m
        policy, rows, val, atlanan = derive(acc, cost, ncost, eligible, lam_d, lam_m)
        all_rows += rows
        all_val += val
        etiket = ("MALIYETE COK DUYARLI" if lam_d >= 0.20 else
                  "DENGELI" if lam_d >= 0.05 else
                  "DOGRULUK ONCELIKLI" if lam_d > 0 else "SALT DOGRULUK")
        print("")
        print(f"=== CALISMA NOKTASI lam={lam_d} ({etiket}) ===")
        for r in rows:
            print(f"  {r['kanal']:9s} {r['snr']:2d} dB -> {r['secim']:16s} "
                  f"(BLEU-1 {r['bleu1']}, {r['gecikme_sn_cumle']} sn/cumle, "
                  f"{r['bellek_gb']} GB)")
        v = [x for x in val if x["lam_d"] == lam_d]
        if v:
            ar = sum(x["regret"] for x in v) / len(v)
            mr = max(x["regret"] for x in v)
            ad = sum(x["dogruluk_farki"] for x in v) / len(v)
            print(f"  tutulan SNR ({len(v)} nokta): ortalama fayda-regret "
                  f"{ar:.4f} | en kotu {mr:.4f} | ortalama dogruluk farki {ad:+.4f}")
        if atlanan:
            print(f"  !! DOGRULANAMAYAN {len(atlanan)} nokta (secilen aksiyonun o "
                  f"SNR'da olcumu yok):")
            for ch, snr, a in atlanan:
                print(f"       {ch} {snr} dB -> {a}")

    write_csv(RES / f"policy_table{args.tag}.csv", all_rows)
    write_csv(RES / f"policy_validation{args.tag}.csv", all_val)
    print("")
    print(f"lam2={args.lam_m} | M_max={args.mem_max} GB | {len(lams)} calisma noktasi "
          f"-> policy_table{args.tag}.csv, policy_validation{args.tag}.csv, "
          f"policy_actions{args.tag}.csv")


if __name__ == "__main__":
    main()
