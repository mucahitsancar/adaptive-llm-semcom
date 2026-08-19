# -*- coding: utf-8 -*-
"""
TEMİZ gecikme/bellek ölçümü (politika maliyet terimleri için).

Neden ayrı script: ana ölçüm koşularındaki sn/cümle değerleri, aynı anda başka
işler (CPU'da benzerlik hesabı, model indirme, arka plan zinciri) çalışırken
alındığı için dalgalanabiliyor — 4B-INT4'te 7.3 ile 33.2 sn/cümle arası fark
görüldü. Politikanın maliyet terimi bu gürültüyle raporlanamaz.

Bu script her konfigürasyonu TEK BAŞINA, sabit koşulda ölçer:
  - tek süreç, başka GPU işi yok (çalıştırmadan önce GPU boş olmalı)
  - ısınma turu (warm-up) sonrası N cümle
  - medyan + çeyrekler arası aralık (ortalama tek uç değerden bozulmasın)
  - tepe VRAM (torch.cuda.max_memory_allocated)
Çıktı: experiments/results/latency_bench.csv
"""
import argparse
import csv
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"

# (etiket, model, quant)
CONFIGS = [
    ("qwen0.8b-fp16", "Qwen/Qwen3.5-0.8B", "none"),
    ("qwen2b-fp16", "Qwen/Qwen3.5-2B", "none"),
    ("qwen2b-int8", "Qwen/Qwen3.5-2B", "int8"),
    ("qwen2b-int4", "Qwen/Qwen3.5-2B", "int4"),
    ("qwen4b-int4", "Qwen/Qwen3.5-4B", "int4"),
]
# NOT: DeepSC gecikmesi ayrı ölçülür (farklı çözücü: greedy, LLM'siz) —
# bkz. bench_deepsc_latency.py. Politikada aksiyon a0 olarak yer alıyor.

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=20, dest="n",
                    help="ölçülecek cümle sayısı (NOT: '--n' kullanılmıyor; "
                         "conda run onu --name ile karıştırıyor)")
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--snr", type=int, default=9, help="sabit SNR (gecikme SNR'dan bağımsız olmalı)")
parser.add_argument("--out", default=str(RES / "latency_bench.csv"))


def bench(model, quant, n, warmup, snr):
    """Alt süreçte tek konfigürasyon ölç: cümle başına süreler + tepe VRAM."""
    code = f'''
import sys, time, json, torch
sys.path.insert(0, r"{ROOT / 'src' / 'llm_sc'}")
from pathlib import Path
from physics import Channel, build_constellation, symbols_per_token
sys.argv = ["x"]
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import importlib.util
spec = importlib.util.spec_from_file_location("ev", r"{ROOT / 'src' / 'llm_sc' / 'eval_llm_sc.py'}")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

dev = "cuda"
tok = AutoTokenizer.from_pretrained("{model}")
q = "{quant}"
if q == "none":
    m = AutoModelForCausalLM.from_pretrained("{model}", torch_dtype=torch.float16).to(dev).eval()
else:
    qc = (BitsAndBytesConfig(load_in_8bit=True) if q == "int8" else
          BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16))
    m = AutoModelForCausalLM.from_pretrained("{model}", quantization_config=qc, device_map="auto").eval()

V = m.config.vocab_size
const = build_constellation(V, dev)
sents = Path(r"{ROOT / 'data' / 'processed' / 'europarl' / 'test_sentences.txt'}").read_text(
    encoding="utf-8").strip().splitlines()[:{warmup + n}]
enc = [torch.tensor(tok(s).input_ids, device=dev) for s in sents]
allowed = torch.zeros(V, dtype=torch.bool, device=dev)
for ids in enc: allowed[ids] = True
na = int(allowed.sum())
ch = Channel({snr}, "AWGN", dev)
from physics import channel_loglik
torch.cuda.reset_peak_memory_stats()
times, per_tok, lens = [], [], []
t_chan = t_lm = 0.0
S = const.shape[1]
for i, ids in enumerate(enc):
    tx = const[ids].reshape(-1)
    y, h, n0 = ch.transmit(tx)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    ev.map_beam_decode(m, tok, y, h, n0, const, len(ids), 10, allowed, n_allowed=na)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if i >= {warmup}:
        times.append(dt); lens.append(len(ids)); per_tok.append(dt / len(ids))
        # bilesen ayristirmasi: kanal-olabilirlik vs LLM ileri gecisi (tek adim)
        torch.cuda.synchronize(); a0 = time.perf_counter()
        for _ in range(5): channel_loglik(y[:S], h[:S], n0, const)
        torch.cuda.synchronize(); t_chan += (time.perf_counter() - a0) / 5
        seq = torch.full((10, 4), tok.eos_token_id or 0, dtype=torch.long, device=dev)
        torch.cuda.synchronize(); a0 = time.perf_counter()
        for _ in range(5):
            with torch.no_grad(): m(seq)
        torch.cuda.synchronize(); t_lm += (time.perf_counter() - a0) / 5
print("RESULT " + json.dumps({{"times": times, "per_tok": per_tok, "lens": lens,
      "chan_step_s": t_chan / max(len(times),1), "lm_step_s": t_lm / max(len(times),1),
      "vram_gb": torch.cuda.max_memory_allocated() / 1e9}}))
'''
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            import json
            return json.loads(line[7:])
    print(r.stdout[-2000:], r.stderr[-2000:])
    return None


def main():
    args = parser.parse_args()
    rows = []
    for label, model, quant in CONFIGS:
        print(f"\n=== {label} ölçülüyor (n={args.n}, warmup={args.warmup}, SNR={args.snr}) ===", flush=True)
        res = bench(model, quant, args.n, args.warmup, args.snr)
        if not res:
            print(f"  {label}: ÖLÇÜLEMEDİ (bellek yetmemiş olabilir)")
            continue
        t = sorted(res["times"])
        pt = sorted(res.get("per_tok", []))
        row = {"config": label, "quant": quant,
               "median_s": round(statistics.median(t), 3),
               "p25_s": round(t[len(t) // 4], 3),
               "p75_s": round(t[3 * len(t) // 4], 3),
               # uzunluk karıştırıcısını kaldıran asıl ölçüt:
               "median_s_per_token": round(statistics.median(pt), 4) if pt else "",
               "ort_token": round(sum(res.get("lens", [0])) / max(len(t), 1), 1),
               # darboğaz ayrıştırması (adım başına):
               "kanal_adim_ms": round(1000 * res.get("chan_step_s", 0), 2),
               "llm_adim_ms": round(1000 * res.get("lm_step_s", 0), 2),
               "vram_gb": round(res["vram_gb"], 3), "n": len(t)}
        rows.append(row)
        print(f"  medyan {row['median_s']} sn/cümle · {row['median_s_per_token']} sn/token "
              f"(ort {row['ort_token']} token) · kanal {row['kanal_adim_ms']} ms vs "
              f"LLM {row['llm_adim_ms']} ms/adım · VRAM {row['vram_gb']} GB")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} konfigürasyon → {args.out}")


if __name__ == "__main__":
    main()
