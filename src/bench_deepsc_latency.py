# -*- coding: utf-8 -*-
"""
DeepSC için temiz gecikme/bellek ölçümü (politikanın a0 aksiyonu).

LLM tarafındaki bench_latency.py ile AYNI protokol: tek süreç, ısınma turu,
cümle başına süre, medyan + çeyrekler, tepe VRAM. Fark yalnız çözücüde
(DeepSC greedy decode kullanıyor, beam yok).
Çıktı: experiments/results/latency_bench.csv dosyasına satır ekler.
"""
import argparse
import csv
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "DeepSC"))
RES = ROOT / "experiments" / "results"

import torch  # noqa: E402
from models.transceiver import DeepSC  # noqa: E402
from utils import SNR_to_noise, greedy_decode  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "deepsc-AWGN"))
parser.add_argument("--data-dir", default=str(ROOT / "data" / "processed" / "europarl"))
parser.add_argument("--samples", type=int, default=20, dest="n")
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--snr", type=int, default=9)
parser.add_argument("--label", default="deepsc")


def main():
    args = parser.parse_args()
    dev = "cuda"
    vocab = json.load(open(Path(args.data_dir) / "vocab.json"))
    tok2idx = vocab["token_to_idx"]
    V = len(tok2idx)
    pad, start = tok2idx["<PAD>"], tok2idx["<START>"]

    net = DeepSC(4, V, V, V, V, 128, 8, 512, 0.1).to(dev).eval()
    ck = Path(args.checkpoint)
    f = ck if ck.is_file() else sorted(ck.glob("checkpoint_*.pth"))[-1]
    net.load_state_dict(torch.load(f, map_location=dev))

    data = pickle.load(open(Path(args.data_dir) / "test_data.pkl", "rb"))[: args.warmup + args.n]
    noise = SNR_to_noise(args.snr)
    torch.cuda.reset_peak_memory_stats()
    times, per_tok, lens = [], [], []
    with torch.no_grad():
        for i, seq in enumerate(data):
            x = torch.tensor([seq], device=dev)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            greedy_decode(net, x, noise, 30, pad, start, "AWGN")
            torch.cuda.synchronize()
            if i >= args.warmup:
                dt = time.perf_counter() - t0
                times.append(dt)
                lens.append(len(seq))
                per_tok.append(dt / max(len(seq), 1))

    t = sorted(times)
    pt = sorted(per_tok)
    # NOT: token başına değer YALNIZ bilgi amaçlı. DeepSC kelime düzeyinde,
    # LLM hattı alt-kelime düzeyinde tokenleştirdiği için iki aile arasında
    # token başına kıyas yapılamaz. Politikanın maliyet terimi cümle başına
    # süredir; tüm konfigürasyonlar AYNI test cümlelerinde ölçüldüğü için
    # cümle başına değer aileler arasında karşılaştırılabilir.
    row = {"config": args.label, "quant": "fp32-small",
           "median_s": round(statistics.median(t), 3),
           "p25_s": round(t[len(t) // 4], 3),
           "p75_s": round(t[3 * len(t) // 4], 3),
           "median_s_per_token": round(statistics.median(pt), 4),
           "ort_token": round(sum(lens) / max(len(lens), 1), 1),
           "kanal_adim_ms": "", "llm_adim_ms": "",   # DeepSC'de bu ayrım yok
           "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
           "n": len(t)}
    print(f"{row['config']}: medyan {row['median_s']} sn/cümle "
          f"(IQR {row['p25_s']}-{row['p75_s']}), tepe VRAM {row['vram_gb']} GB")

    out = RES / "latency_bench.csv"
    rows = list(csv.DictReader(open(out))) if out.exists() else []
    rows = [r for r in rows if r.get("config") != args.label] + [row]
    # Alan adları TÜM satırların birleşimi olmalı: yalnız bu satırın anahtarları
    # kullanılırsa v2 kolonlarını taşıyan LLM satırları ya düşer ya da
    # DictWriter hata verir.
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)
    print(f"eklendi → {out}")


if __name__ == "__main__":
    main()
