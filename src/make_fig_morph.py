# -*- coding: utf-8 -*-
"""
Fig. 9 — Morfolojik anlam korunumu (ikincil katkı).

Anlatılan tek şey: kelime düzeyi ölçütler (BLEU) ve genel MSPS, düşük SNR'da
anlamın bozulduğunu GİZLİYOR. Anlam-kritik morfemler (olumsuzluk, yeterlik,
zaman) dilbilgisel olanlardan belirgin biçimde daha kırılgan; kritik tersinme
oranı bunu tek sayıda gösteriyor.

Kaynak: experiments/results/msps_all.csv + TR-qwen08b_eval.csv (gerçek ölçüm).
Stil: make_figures.py ile aynı palet.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C_KRITIK, C_DILBILGISI, C_BLEU = "#eb6834", "#2a78d6", "#898781"

# Anlam-kritik kategoriler (bozulması anlamı TERSİNE çevirir) ve etiketleri
KRITIK = {"olumsuzluk": "negation", "yeterlik": "potentiality", "zaman": "tense"}
DIGER = {"kisi": "person", "hal": "case", "kip": "mood", "soru": "question"}


def style_ax(ax, title, xlabel=""):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.set_title(title, color=INK, fontsize=11, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2)


def load_msps():
    rows = list(csv.DictReader(open(RES / "msps_all.csv", encoding="utf-8")))
    out = {}
    for r in rows:
        lab = r["label"]
        if not lab.startswith("TR-qwen08b-snr"):
            continue
        snr = int(lab.rsplit("snr", 1)[1])
        out[snr] = r
    return out


def load_bleu():
    p = RES / "TR-qwen08b_eval.csv"
    if not p.exists():
        return {}
    return {int(float(r["snr"])): float(r["bleu1"])
            for r in csv.DictReader(open(p, encoding="utf-8"))}


def main():
    msps, bleu = load_msps(), load_bleu()
    if not msps:
        raise SystemExit("msps_all.csv içinde TR ölçümü yok")
    snrs = sorted(msps)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.1), facecolor=SURFACE)

    # --- SOL: 0 dB'de kategori kırılımı, kritik olanlar vurgulu ---
    at0 = msps[min(snrs)]
    cats, vals, colors = [], [], []
    for key, eng in list(KRITIK.items()) + list(DIGER.items()):
        v = at0.get(f"kat_{key}", "")
        if v in ("", None):
            continue
        cats.append(eng)
        vals.append(float(v))
        colors.append(C_KRITIK if key in KRITIK else C_DILBILGISI)
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    cats = [cats[i] for i in order]
    vals = [vals[i] for i in order]
    colors = [colors[i] for i in order]

    axL.barh(cats, vals, color=colors, zorder=3, height=0.62)
    for i, v in enumerate(vals):
        axL.text(v + 0.015, i, f"{v:.3f}", va="center", color=INK2, fontsize=9)
    agg = float(at0["msps"])
    axL.axvline(agg, color=MUTED, linestyle=":", linewidth=1.4, zorder=4)
    axL.text(agg + 0.015, len(vals) - 0.35, f"aggregate MSPS {agg:.3f}",
             color=MUTED, fontsize=8.5)
    axL.set_xlim(0, 1.0)
    style_ax(axL, f"(a) Feature preservation by category at {min(snrs)} dB",
             "fraction of target features preserved")
    axL.grid(axis="y", visible=False)
    axL.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axL.text(0.52, -0.5, "orange = meaning-inverting categories",
             color=C_KRITIK, fontsize=8.5)

    # --- SAĞ: kritik tersinme oranı vs SNR, BLEU ve genel MSPS ile birlikte ---
    inv = [float(msps[s]["kritik_tersinme_orani"]) for s in snrs]
    aggs = [float(msps[s]["msps"]) for s in snrs]
    axR.plot(snrs, inv, color=C_KRITIK, marker="o", linewidth=2.2, zorder=4,
             label="critical inversion rate")
    axR.plot(snrs, aggs, color=C_DILBILGISI, marker="s", linestyle="--",
             linewidth=1.8, zorder=3, label="aggregate MSPS")
    if bleu:
        axR.plot(snrs, [bleu[s] for s in snrs], color=C_BLEU, marker="^",
                 linestyle=":", linewidth=1.6, zorder=2, label="BLEU-1")
    # Açıklama sol üst köşeye: yükselen MSPS eğrisiyle kesişmesin.
    axR.annotate(f"{inv[0]:.3f} of sentences lose a\nmeaning-critical morpheme",
                 xy=(snrs[0], inv[0]), xytext=(snrs[0] + 0.4, 0.90),
                 color=C_KRITIK, fontsize=8.5,
                 arrowprops=dict(arrowstyle="-", color=C_KRITIK, linewidth=0.9,
                                 shrinkB=3))
    axR.set_ylim(0, 1.05)
    axR.set_xticks(snrs)
    style_ax(axR, "(b) Word-level scores hide meaning inversion", "SNR (dB)")
    leg = axR.legend(frameon=False, fontsize=9, loc="center right")
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle("Turkish arm, Qwen3.5-0.8B, AWGN, n=150",
                 color=INK, fontsize=12.5, y=1.02)
    out = FIG / "fig9_morfoloji.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE, dpi=200)
    print(f"yazildi -> {out}")
    print(f"  kategoriler ({min(snrs)} dB): " +
          ", ".join(f"{c}={v:.3f}" for c, v in zip(cats, vals)))
    print(f"  kritik tersinme: " +
          ", ".join(f"{s}dB={i:.3f}" for s, i in zip(snrs, inv)))


if __name__ == "__main__":
    main()
