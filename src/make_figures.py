# -*- coding: utf-8 -*-
"""
Faz 1 figürleri: BLEU-1 vs SNR (AWGN + Rayleigh), makale eğrisi ile bizim
reproduksiyon yan yana. Makale değerleri Fig. 6'dan görsel okumadır (±0.02) —
kaynak: Xie ve ark., IEEE TSP 2021 (arXiv:2006.10685), Fig. 6.

Stil: dataviz yönergesi — kategorik slot1 (mavi #2a78d6) = biz, slot2 (turuncu
#eb6834) = makale; kimlik yalnız renkte değil (kesikli çizgi + kare marker);
zemin #fcfcfb, ızgara #e1e0d9, eksen #c3c2b7, yazı #0b0b0b/#52514e/#898781.
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

# Renk ve krom (dataviz referans paleti, light mod)
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C_BIZ, C_MAKALE = "#2a78d6", "#eb6834"

SNR = [0, 3, 6, 9, 12, 15, 18]

# Makale Fig. 6 (a)/(b) BLEU-1 eğrilerinden okunan yaklaşık değerler (±0.02)
PAPER_B1 = {
    "AWGN": [0.61, 0.82, 0.885, 0.905, 0.91, 0.915, 0.92],
    "Rayleigh": [0.57, 0.74, 0.82, 0.90, 0.92, 0.93, 0.94],
}


def read_eval(prefix):
    path = RES / f"{prefix}_eval.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    return {"snr": [int(float(r["snr"])) for r in rows],
            **{k: [float(r[k]) for r in rows] for k in rows[0] if k != "snr"}}


def style_ax(ax, title):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.set_title(title, color=INK, fontsize=12, pad=10)
    ax.set_xlabel("SNR (dB)", color=INK2)
    ax.set_xticks(SNR)
    ax.set_ylim(0, 1.0)


def main():
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]

    ours = {ch: read_eval(f"deepsc-{ch}") for ch in ("AWGN", "Rayleigh")}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    fig.patch.set_facecolor(SURFACE)

    for ax, ch, panel in zip(axes, ("AWGN", "Rayleigh"), ("(a)", "(b)")):
        style_ax(ax, f"{panel} {ch}")
        ax.plot(SNR, PAPER_B1[ch], linestyle=(0, (4, 2)), linewidth=2,
                marker="s", markersize=6.5, color=C_MAKALE,
                markerfacecolor=C_MAKALE, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label="Makale (Fig. 6'dan okunan)", zorder=3)
        if ours[ch]:
            ax.plot(ours[ch]["snr"], ours[ch]["bleu1"], linewidth=2,
                    marker="o", markersize=7, color=C_BIZ,
                    markerfacecolor=C_BIZ, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label="Bizim reproduksiyon", zorder=4)
        ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK2)
    axes[0].set_ylabel("BLEU (1-gram)", color=INK2)

    fig.suptitle("DeepSC reproduksiyonu — BLEU-1 vs SNR (Europarl, kelime başına 8 sembol)",
                 color=INK, fontsize=13, y=1.02)
    fig.text(0.01, -0.04,
             "Makale eğrisi: Xie vd. 2021, Fig. 6'dan görsel okuma (±0.02). "
             "Bizim eğri: 7347 test cümlesi × 2 tur, greedy çözümleme, commit c31d5bb.",
             color=MUTED, fontsize=7.5)
    fig.tight_layout()
    out = FIG / "fig1_bleu1_makale_vs_biz.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"yazildi: {out}")

    # Similarity (yalniz bizim olcum — metrik implementasyonu makaleden farkli,
    # mutlak kiyas yapilmaz; rapora dipnotla girer)
    fig2, ax2 = plt.subplots(figsize=(5.4, 4.2), dpi=300)
    fig2.patch.set_facecolor(SURFACE)
    style_ax(ax2, "Cümle benzerliği (HF BERT, katman-11, kosinüs)")
    for ch, color, mk in (("AWGN", C_BIZ, "o"), ("Rayleigh", C_MAKALE, "s")):
        if ours[ch]:
            ax2.plot(ours[ch]["snr"], ours[ch]["similarity"], linewidth=2,
                     marker=mk, markersize=7, color=color,
                     markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.2,
                     label=ch, zorder=3)
    ax2.set_ylim(0.5, 1.0)
    ax2.set_ylabel("Benzerlik", color=INK2)
    ax2.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK2)
    fig2.text(0.01, -0.05,
              "Metrik: bert-base-uncased, pad-maskeli token toplamı, kosinüs. Makale ile\n"
              "mutlak kıyas yapılmaz (farklı BERT varyantı/normalizasyon) — eğri şekli karşılaştırılır.",
              color=MUTED, fontsize=7.5)
    fig2.tight_layout()
    out2 = FIG / "fig2_similarity_biz.png"
    fig2.savefig(out2, bbox_inches="tight", facecolor=SURFACE)
    print(f"yazildi: {out2}")

    # Fig 3: MINE dahil tam karşılaştırma (rapor versiyonu, 3 eğri/panel)
    C_MINE = "#1baf7a"  # kategorik slot 3 (aqua) — ilk 3 slot all-pairs doğrulanmış
    mine = {ch: read_eval(f"deepsc-{ch}-mine") for ch in ("AWGN", "Rayleigh")}
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    fig3.patch.set_facecolor(SURFACE)
    for ax, ch, panel in zip(axes3, ("AWGN", "Rayleigh"), ("(a)", "(b)")):
        style_ax(ax, f"{panel} {ch}")
        ax.plot(SNR, PAPER_B1[ch], linestyle=(0, (4, 2)), linewidth=2,
                marker="s", markersize=6.5, color=C_MAKALE,
                markerfacecolor=C_MAKALE, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label="Makale (Fig. 6'dan)", zorder=3)
        if ours[ch]:
            ax.plot(ours[ch]["snr"], ours[ch]["bleu1"], linewidth=2,
                    marker="o", markersize=7, color=C_BIZ,
                    markerfacecolor=C_BIZ, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label="Biz (MINE'sız)", zorder=4)
        if mine[ch]:
            ax.plot(mine[ch]["snr"], mine[ch]["bleu1"], linestyle=(0, (1, 1)), linewidth=2,
                    marker="^", markersize=7, color=C_MINE,
                    markerfacecolor=C_MINE, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label="Biz (MINE'lı)", zorder=4)
        ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK2)
    axes3[0].set_ylabel("BLEU (1-gram)", color=INK2)
    fig3.suptitle("DeepSC — MINE ablation dahil tam karşılaştırma (BLEU-1 vs SNR)",
                  color=INK, fontsize=13, y=1.02)
    fig3.text(0.01, -0.06,
              "MINE'lı koşular main.py gereği sabit n_var=0.1 (≈17 dB) gürültüyle eğitilir; düşük-SNR çöküşü\n"
              "bu rejim farkındandır — MINE etkisi ile gürültü rejimi etkisi bu konfigürasyonlarda ayrıştırılamaz.",
              color=MUTED, fontsize=7.5)
    fig3.tight_layout()
    out3 = FIG / "fig3_bleu1_mine_ablation.png"
    fig3.savefig(out3, bbox_inches="tight", facecolor=SURFACE)
    print(f"yazildi: {out3}")

    # Fig 4: FAZ 1.5 FİNAL — iyileştirmeler makale ve baseline ile birlikte
    C_IYI, C_BEAM = "#1baf7a", "#eda100"  # slot 3 aqua, slot 4 sarı
    panels = {
        "AWGN": [("deepsc-AWGN", "Baseline", C_BIZ, "o", "-"),
                 ("deepsc-AWGN-widesnr", "Geniş-SNR eğitimi", C_IYI, "^", (0, (1, 1))),
                 ("deepsc-AWGN-widesnr-beam4", "Geniş-SNR + Beam-4", C_BEAM, "D", (0, (3, 1, 1, 1)))],
        "Rayleigh": [("deepsc-Rayleigh", "Baseline", C_BIZ, "o", "-"),
                     ("deepsc-Rayleigh-perH", "Cümle-başına H", C_IYI, "^", (0, (1, 1))),
                     ("deepsc-Rayleigh-perH-beam4", "Cümle-başına H + Beam-4", C_BEAM, "D", (0, (3, 1, 1, 1)))],
    }
    fig4, axes4 = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=300)
    fig4.patch.set_facecolor(SURFACE)
    for ax, (ch, series) in zip(axes4, panels.items()):
        style_ax(ax, f"({'a' if ch == 'AWGN' else 'b'}) {ch}")
        ax.plot(SNR, PAPER_B1[ch], linestyle=(0, (4, 2)), linewidth=2, marker="s",
                markersize=6.5, color=C_MAKALE, markerfacecolor=C_MAKALE,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label="Makale (Fig. 6'dan)", zorder=3)
        for prefix, label, color, mk, ls in series:
            d = read_eval(prefix)
            if d:
                ax.plot(d["snr"], d["bleu1"], linestyle=ls, linewidth=2, marker=mk,
                        markersize=6.5, color=color, markerfacecolor=color,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=4)
        ax.legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=INK2)
    axes4[0].set_ylabel("BLEU (1-gram)", color=INK2)
    fig4.suptitle("Faz 1.5 — iyileştirmelerin makale ve baseline ile karşılaştırması (BLEU-1)",
                  color=INK, fontsize=13, y=1.02)
    fig4.text(0.01, -0.05,
              "Tüm 'biz' eğrileri: 7347 test cümlesi, aynı değerlendirme protokolü. AWGN 0 dB: baseline 0.565 → "
              "geniş-SNR+beam 0.737 (makale ~0.61). Rayleigh 3 dB: baseline 0.637 → per-H 0.702 (makale ~0.74).",
              color=MUTED, fontsize=7.5)
    fig4.tight_layout()
    out4 = FIG / "fig4_faz15_final.png"
    fig4.savefig(out4, bbox_inches="tight", facecolor=SURFACE)
    print(f"yazildi: {out4}")

    # Fig 5: FAZ 2 — DeepSC vs LLM (Qwen-0.8B): kesişim figürü
    C_LLM = "#eb6834"  # slot2 turuncu = LLM
    champs = {"AWGN": ("deepsc-AWGN-widesnr", "DeepSC (geniş-SNR)"),
              "Rayleigh": ("deepsc-Rayleigh-perH", "DeepSC (per-H)")}
    fig5, axes5 = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=300)
    fig5.patch.set_facecolor(SURFACE)
    for ax, ch in zip(axes5, ("AWGN", "Rayleigh")):
        style_ax(ax, f"({'a' if ch == 'AWGN' else 'b'}) {ch}")
        base = read_eval(f"deepsc-{ch}")
        champ = read_eval(champs[ch][0])
        qwen = read_eval(f"qwen08b-{ch}")
        if base:
            ax.plot(base["snr"], base["bleu1"], linewidth=2, marker="o", markersize=6.5,
                    color=C_BIZ, markerfacecolor=C_BIZ, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, label="DeepSC (baseline)", zorder=3)
        if champ:
            ax.plot(champ["snr"], champ["bleu1"], linestyle=(0, (1, 1)), linewidth=2,
                    marker="^", markersize=6.5, color="#1baf7a", markerfacecolor="#1baf7a",
                    markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=champs[ch][1], zorder=3)
        if qwen:
            ax.plot(qwen["snr"], qwen["bleu1"], linestyle=(0, (4, 2)), linewidth=2.2,
                    marker="s", markersize=6.5, color=C_LLM, markerfacecolor=C_LLM,
                    markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label="LLM-SC (Qwen3.5-0.8B)", zorder=4)
        ax.legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=INK2)
    axes5[0].set_ylabel("BLEU (1-gram)", color=INK2)
    fig5.suptitle("Faz 2 — Klasik semantik kodek ile LLM tabanlı çözümün kesişimi (BLEU-1)",
                  color=INK, fontsize=13, y=1.02)
    fig5.text(0.01, -0.06,
              "Aynı 300 test cümlesi, aynı kanal protokolü. Kesişim: AWGN ~5-6 dB, Rayleigh ~7-8 dB. "
              "Yüksek SNR'da Qwen AWGN'de BLEU=1.0'a ulaşır (hatasız iletim); düşük SNR'da klasik kodek üstündür\n"
              "— kanal-farkında adaptif seçim politikasının ampirik temeli.",
              color=MUTED, fontsize=7.5)
    fig5.tight_layout()
    out5 = FIG / "fig5_deepsc_vs_llm_kesisim.png"
    fig5.savefig(out5, bbox_inches="tight", facecolor=SURFACE)
    print(f"yazildi: {out5}")

    # Fig 6: YÜK UZUNLUĞU etkisi — kesişim kayması (kısa vs uzun cümle)
    LSNR = [0, 6, 12, 18]
    short_ds = read_eval("deepsc-AWGN")
    short_q = read_eval("qwen08b-AWGN")
    long_ds = read_eval("LONG-deepsc-base")
    long_ds_re = read_eval("LONG-deepsc-retrained")   # B kolu: rejime uygun eğitim
    long_q = read_eval("LONG-qwen08b")

    def pick(d, snrs=LSNR):
        """snrs=None → dosyadaki bütün noktaları al (eksik ızgarayı gizleme)."""
        if not d:
            return None, None
        idx = [i for i, s in enumerate(d["snr"]) if snrs is None or s in snrs]
        return [d["snr"][i] for i in idx], [d["bleu1"][i] for i in idx]

    # ÖNEMLİ: bu figürdeki TÜM eğriler AYNI uzun-cümle test kümesinde ölçüldü.
    # Kısa rejim eğrileri buraya konmaz — BLEU uzunluğa duyarlı olduğu için farklı
    # test kümelerinin değerleri aynı eksende kıyaslanamaz (kısa rejim: fig5).
    fig6, ax6 = plt.subplots(figsize=(6.4, 4.6), dpi=300)
    fig6.patch.set_facecolor(SURFACE)
    style_ax(ax6, "Uzun yükte doygunluk kapasite değil, EĞİTİM REJİMİ sorunu (AWGN)")
    ax6.set_xticks(LSNR)
    for d, lbl, color, ls, mk in (
            (long_ds, "DeepSC · kısa cümlelerle eğitilmiş (yayındaki kurgu)",
             C_BIZ, (0, (4, 2)), "v"),
            (long_ds_re, "DeepSC · UZUN cümlelerle eğitilmiş (bizim B kolu)",
             "#1f7a3d", "-", "P"),
            (long_q, "LLM (Qwen-0.8B) · eğitim yok",
             C_LLM, (0, (1, 1.4)), "D")):
        x, y = pick(d, snrs=None)
        if x:
            ax6.plot(x, y, linestyle=ls, linewidth=2, marker=mk, markersize=6.5,
                     color=color, markerfacecolor=color, markeredgecolor=SURFACE,
                     markeredgewidth=1.2, label=lbl, zorder=3)
    ax6.set_ylabel("BLEU (1-gram)", color=INK2)
    ax6.legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=INK2)
    ax6.set_xticks(sorted({0, 1, 3, 6, 12, 18}))
    fig6.text(0.01, -0.11,
              "Yayındaki gibi kurulduğunda (kısa cümlelerle eğitilmiş kodek, uzun yükle test) klasik kodek\n"
              "0.578'de doyuyor ve bu 'kapasite sınırı' gibi görünüyor. Aynı kodek uzun cümlelerle yeniden\n"
              "eğitildiğinde 0.962'ye çıkıyor: doygunluk KAPASİTE DEĞİL, eğitim rejimi uyuşmazlığıymış.\n"
              "Sonuç: kesişim eşiği baseline'ın eğitim rejimine göre birkaç dB yanlı → adil kıyas için\n"
              "her iki sistem de aynı yük rejiminde kurulmalı. (Not: BLEU uzunluğa göre değişir; uzun\n"
              "rejimdeki 0.962 ile kısa rejimdeki 0.933 doğrudan kıyaslanamaz.)",
              color=MUTED, fontsize=7.5)
    fig6.tight_layout()
    out6 = FIG / "fig6_yuk_uzunlugu.png"
    fig6.savefig(out6, bbox_inches="tight", facecolor=SURFACE)
    print(f"yazildi: {out6}")

    # Fig 7: BOYUT MERDİVENİ — DeepSC + Vicuna-7B + Qwen 0.8B/2B/4B (AWGN)
    # Kimlik yalnız renkte değil: her seri farklı çizgi stili + marker.
    ladder = [
        ("deepsc-AWGN", "DeepSC (klasik kodek)", "#2a78d6", "-", "o"),
        ("vicuna7b-AWGN", "Vicuna-7B (2023)", "#eb6834", (0, (4, 2)), "s"),
        ("qwen08b-AWGN", "Qwen3.5-0.8B (2026)", "#1baf7a", (0, (1, 1)), "^"),
        ("qwen2b-AWGN", "Qwen3.5-2B (2026)", "#008300", (0, (3, 1, 1, 1)), "D"),
        ("qwen4b-fp16-AWGN", "Qwen3.5-4B (2026)", "#4a3aa7", (0, (5, 1, 1, 1, 1, 1)), "*"),
    ]
    fig7, ax7 = plt.subplots(figsize=(6.6, 4.8), dpi=300)
    fig7.patch.set_facecolor(SURFACE)
    style_ax(ax7, "Boyut merdiveni: klasik kodek ile LLM çözücüler (AWGN)")
    for prefix, lbl, color, ls, mk in ladder:
        d = read_eval(prefix)
        if not d:
            continue
        ms = 9 if mk == "*" else 6.5
        ax7.plot(d["snr"], d["bleu1"], linestyle=ls, linewidth=2, marker=mk,
                 markersize=ms, color=color, markerfacecolor=color,
                 markeredgecolor=SURFACE, markeredgewidth=1.2, label=lbl, zorder=3)
    ax7.set_ylabel("BLEU (1-gram)", color=INK2)
    ax7.set_ylim(0.15, 1.02)
    ax7.legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=INK2)
    fig7.text(0.01, -0.10,
              "Düşük SNR'da klasik kodek üstün; ≥6-9 dB'de LLM'ler öne geçip 15-18 dB'de birebir kurtarma yapıyor\n"
              "(DeepSC ~0.93'te doyuyor). 2026 çıkışlı 0.8B, 2023 çıkışlı 7B'yi tüm noktalarda geçiyor; 2B→4B\n"
              "geçişinde ölçülebilir kazanç yok (ölçek doygunluğu). 4B ölçümü sürüyor (0-12 dB).",
              color=MUTED, fontsize=7.5)
    fig7.tight_layout()
    out7 = FIG / "fig7_boyut_merdiveni.png"
    fig7.savefig(out7, bbox_inches="tight", facecolor=SURFACE)
    print(f"yazildi: {out7}")


if __name__ == "__main__":
    main()
