# -*- coding: utf-8 -*-
"""
Semantik benzerlik skorları — TÜM koşular için (hoca şartı: "sadece BLEU değil").

Kaydedilmiş metin çiftlerinden (results/*_texts/snrX_{target,decoded}.txt) benzerlik
hesaplar. Metrik, DeepSC değerlendirmesinde kullandığımızla BİREBİR aynıdır
(karşılaştırmanın geçerli olması için): bert-base-uncased, katman-11 çıktısı,
padding-maskeli token toplamı, kosinüs benzerliği (DeepSC makalesi Eq. 13).

CPU'da koşar (--device cpu, varsayılan) → GPU'daki ölçüm kuyruğunu aksatmaz.
Çıktı: experiments/results/similarity_all.csv  (run, snr, similarity, n)
"""
import argparse
import csv
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="bert-base-uncased")
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--out", default=str(RES / "similarity_all.csv"))


@torch.no_grad()
def embed(texts, tok, model, device, bs):
    """DeepSC eval'deki HFSimilarity ile aynı: katman-11, pad-maskeli token toplamı."""
    vecs = []
    for i in range(0, len(texts), bs):
        # max_length=512: BERT'in kendi sınırı; en uzun cümlemiz ~100 token →
        # pratikte KESME YOK. (padding=True olduğu için batch'in en uzununa dolgu
        # yapılır; 512 vermek ek maliyet getirmez.)
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        h = model(**enc).hidden_states[11]                 # [B, T, 768]
        mask = enc["attention_mask"].unsqueeze(-1)
        vecs.append((h * mask).sum(dim=1).cpu())
    return torch.cat(vecs)


def main():
    args = parser.parse_args()
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, output_hidden_states=True).to(args.device).eval()

    rows = []
    for d in sorted(RES.glob("*_texts")):
        run = d.name.replace("_texts", "")
        if "smoke" in run:
            continue
        for dec in sorted(d.glob("snr*_decoded.txt"), key=lambda p: int(p.stem.split("_")[0][3:])):
            snr = int(dec.stem.split("_")[0][3:])
            tgt_p = d / f"snr{snr}_target.txt"
            if not tgt_p.exists():
                continue
            tgt = tgt_p.read_text(encoding="utf-8").strip().splitlines()
            hyp = dec.read_text(encoding="utf-8").strip().splitlines()
            n = min(len(tgt), len(hyp))
            if n == 0:
                continue
            v1 = embed(tgt[:n], tok, model, args.device, args.batch_size)
            v2 = embed(hyp[:n], tok, model, args.device, args.batch_size)
            sim = float(torch.nn.functional.cosine_similarity(v1, v2, dim=-1).mean())
            rows.append({"run": run, "snr": snr, "similarity": round(sim, 4), "n": n})
            print(f"{run:32s} SNR {snr:2d} → benzerlik {sim:.4f} (n={n})", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "snr", "similarity", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} nokta → {args.out}")


if __name__ == "__main__":
    main()
