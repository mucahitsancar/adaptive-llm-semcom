# -*- coding: utf-8 -*-
"""
Fig. 8 — Adaptif çıkarım politikası (ANA KATKI).

İki panel:
  (a) KARAR HARİTASI — her çalışma noktası (gecikme toleransı λ₁) ve kanal için,
      SNR ekseninde politikanın seçtiği konfigürasyon. Gecikme toleransı
      arttıkça karar bölgelerinin kayması burada görünür.
  (b) DOĞRULUK–MALİYET SINIRI — sabit bir SNR'da tüm aksiyonlar (gecikme, BLEU-1)
      düzleminde; Pareto sınırı ve politikanın seçimi işaretli. Politikanın neden
      en doğru aksiyonu seçmediği (maliyet ödünü) buradan okunur.

Girdi: experiments/results/policy_table.csv + latency_bench.csv + *_eval.csv
Bu betik gecikme ölçümü (v2) ve politika türetimi bitmeden çalışmaz; eksikse
ne eksik olduğunu söyleyip çıkar (sessiz boş figür üretmez).
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

# Aksiyon → renk. Klasik kodek ayrı bir aile olduğu için ayrı ton.
RENK = {
    "DeepSC": "#2a78d6",
    "DeepSC-improved": "#1f7a3d",
    "Qwen-0.8B-FP16": "#eb6834",
    "Qwen-2B-FP16": "#c2410c",
    "Qwen-2B-INT8": "#eda100",
    "Qwen-2B-INT4": "#a16207",
    "Qwen-4B-INT4": "#7c3aed",
    "Qwen-4B-FP16": "#4c1d95",
}
VARSAYILAN = "#898781"

# (b) panelinde gösterilecek SNR. 6 dB seçildi: hem nicemleme koşularının
# (0/6/12/18) hem de tam SNR ızgarasının kesiştiği nokta VE değerlerin en ayırt
# edici olduğu bölge (0.911-0.948). 12 dB'de her şey ~0.999 olduğu için sınır
# bilgilendirici olmaz.
FRONTIER_SNR = 6

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="",
                    help="girdi/çıktı dosyalarına ek (prova için, ör. -PROVA)")
parser.add_argument("--latency-csv", default=None,
                    help="politikanın kullandığı gecikme tablosu (aynısı verilmeli)")
parser.add_argument("--snr", type=int, default=FRONTIER_SNR,
                    help="(b) panelindeki SNR")


def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.set_title(title, color=INK, fontsize=11, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)


def read_rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def main():
    args = parser.parse_args()
    tag = args.tag
    pol = read_rows(RES / f"policy_table{tag}.csv")
    if not pol:
        raise SystemExit(
            "policy_table.csv yok. Once temiz gecikme olcumu (bench_latency.py, v2) "
            "ve ardindan derive_policy.py --sweep kosulmali.")

    lams = sorted({float(r["lam_d"]) for r in pol})
    kanallar = ["AWGN", "Rayleigh"]
    snrs = sorted({int(r["snr"]) for r in pol})

    fig = plt.figure(figsize=(12, 4.4), dpi=300, facecolor=SURFACE)
    axA = fig.add_subplot(1, 2, 1)
    axB = fig.add_subplot(1, 2, 2)

    # ---------- (a) KARAR HARİTASI ----------
    # Satırlar: (λ₁, kanal) çiftleri; sütunlar: SNR. Her hücre seçilen aksiyonun rengi.
    satirlar = [(lam, ch) for lam in lams for ch in kanallar]
    secim = {(float(r["lam_d"]), r["kanal"], int(r["snr"])): r["secim"] for r in pol}
    genislik = (snrs[1] - snrs[0]) if len(snrs) > 1 else 3

    kullanilan = []
    for i, (lam, ch) in enumerate(satirlar):
        for s in snrs:
            a = secim.get((lam, ch, s))
            if a is None:
                continue
            c = RENK.get(a, VARSAYILAN)
            if a not in kullanilan:
                kullanilan.append(a)
            axA.barh(i, genislik, left=s - genislik / 2, height=0.72,
                     color=c, edgecolor=SURFACE, linewidth=1.2, zorder=3)

    axA.set_yticks(range(len(satirlar)))
    axA.set_yticklabels([f"$\\lambda_1$={lam:g} · {ch}" for lam, ch in satirlar],
                        fontsize=8.5)
    axA.set_xticks(snrs)
    axA.set_xlim(snrs[0] - genislik / 2, snrs[-1] + genislik / 2)
    axA.invert_yaxis()
    style_ax(axA, "(a) Policy decision map", "SNR (dB)")
    axA.grid(False)
    leg = axA.legend(handles=[Patch(facecolor=RENK.get(a, VARSAYILAN), label=a)
                              for a in kullanilan],
                     frameon=False, fontsize=7.5, ncol=2,
                     loc="upper center", bbox_to_anchor=(0.5, -0.18))
    for t in leg.get_texts():
        t.set_color(INK2)

    # ---------- (b) DOĞRULUK–MALİYET SINIRI ----------
    lat_path = (Path(args.latency_csv) if args.latency_csv
                else RES / "latency_bench.csv")
    lat = {r["config"]: r for r in read_rows(lat_path)}
    # aksiyon → (eval prefix, gecikme anahtari): derive_policy ile ayni eslesme
    import importlib.util
    spec = importlib.util.spec_from_file_location("dp", ROOT / "src" / "derive_policy.py")
    dp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dp)

    acc = dp.load_accuracy()
    noktalar = []
    for ad, (_, anahtar) in dp.ACTIONS.items():
        if anahtar not in lat:
            continue
        key = (ad, "AWGN", args.snr)
        if key not in acc:
            continue
        try:
            d = float(lat[anahtar]["median_s"])
            m = float(lat[anahtar]["vram_gb"])
        except (ValueError, TypeError, KeyError):
            continue
        noktalar.append((ad, d, acc[key], m))

    if noktalar:
        # Pareto sınırı: daha az gecikme VE daha yüksek doğruluk yoksa sınırdadır
        pareto = [n for n in noktalar
                  if not any(o[1] <= n[1] and o[2] >= n[2] and o != n for o in noktalar)]
        pareto.sort(key=lambda n: n[1])
        axB.plot([p[1] for p in pareto], [p[2] for p in pareto],
                 color=MUTED, linestyle=":", linewidth=1.4, zorder=2)
        secilenler = {secim.get((lam, "AWGN", args.snr)) for lam in lams}
        axB.set_xscale("log")
        # Etiket çakışmasını önle: log eksende noktalar iki kümeye ayrılıyor
        # (hızlı kodek solda, LLM'ler sağda). Sağ kümenin etiketleri sola,
        # sol kümenin sağa yazılır; küme içinde BLEU sırasına göre dikey kaydırma.
        d_orta = (min(n[1] for n in noktalar) * max(n[1] for n in noktalar)) ** 0.5
        sagda = sorted([n for n in noktalar if n[1] >= d_orta], key=lambda n: n[2])
        solda = sorted([n for n in noktalar if n[1] < d_orta], key=lambda n: n[2])
        for kume, isaret in ((sagda, -1), (solda, +1)):
            for k, (ad, d, f, m) in enumerate(kume):
                axB.annotate(
                    f"{ad} · {m:.1f} GB", (d, f), textcoords="offset points",
                    xytext=(isaret * 9, 9 * (k - (len(kume) - 1) / 2)),
                    ha="right" if isaret < 0 else "left", va="center",
                    fontsize=7, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.6,
                                    shrinkA=1, shrinkB=3))
        for ad, d, f, m in noktalar:
            secili = ad in secilenler
            axB.scatter(d, f, s=210 if secili else 80,
                        color=RENK.get(ad, VARSAYILAN), zorder=5,
                        edgecolor=INK if secili else SURFACE,
                        linewidth=1.6 if secili else 1.0,
                        marker="*" if secili else "o")
        style_ax(axB, f"(b) Fidelity--cost frontier at {args.snr} dB (AWGN)",
                 "latency (s/sentence, log scale)", "BLEU-1")
        axB.grid(axis="both", color=GRID, linewidth=0.8, zorder=0)
        axB.margins(x=0.28, y=0.20)
        from matplotlib.lines import Line2D
        yildiz = Line2D([], [], marker="*", color=MUTED, linestyle="none",
                        markersize=11, label="policy choice")
        leg2 = axB.legend(handles=[yildiz,
                                   Line2D([], [], color=MUTED, linestyle=":",
                                          label="Pareto frontier")],
                          frameon=False, fontsize=8, loc="upper left")
        for t in leg2.get_texts():
            t.set_color(INK2)
    else:
        axB.text(0.5, 0.5, "gecikme olcumu eksik", ha="center", va="center",
                 transform=axB.transAxes, color=MUTED)
        style_ax(axB, "(b) Fidelity--cost frontier")

    val = read_rows(RES / f"policy_validation{tag}.csv")
    altyazi = ("Karar bolgeleri Gamma_fit={0,6,12,18} dB uzerinde kuruldu. ")
    if val:
        reg = [float(r["regret"]) for r in val]
        altyazi += (f"Tutulan SNR'larda ({len(val)} nokta) ortalama fayda-regret "
                    f"{sum(reg)/len(reg):.4f}, en kotu {max(reg):.4f}.")
    fig.text(0.01, -0.16, altyazi, color=MUTED, fontsize=7.5)

    fig.tight_layout()
    out = FIG / f"fig8_politika{tag}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"yazildi -> {out}")
    print(f"  {len(lams)} calisma noktasi x {len(kanallar)} kanal x {len(snrs)} SNR")
    if noktalar:
        print(f"  (b) panelinde {len(noktalar)} aksiyon, {len(pareto)} tanesi Pareto sinirinda")


if __name__ == "__main__":
    main()
