# -*- coding: utf-8 -*-
"""Yarım kalan Qwen ölçümünü kaldığı yerden sürdürür: her kanal için bitmiş SNR
noktalarını (texts klasöründen) tespit eder, yalnız eksikleri koşar. Sonda tüm
noktalar tamamsa BLEU'ları texts dosyalarından yeniden hesaplayıp final CSV yazar."""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from nltk.translate.bleu_score import sentence_bleu

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
SNRS = [0, 3, 6, 9, 12, 15, 18]
RUNS = [("qwen08b-AWGN", "AWGN"), ("qwen08b-Rayleigh", "Rayleigh")]
LIMIT = 300


def done_snrs(prefix):
    d = RES / f"{prefix}_texts"
    ok = []
    for s in SNRS:
        if (d / f"snr{s}_target.txt").exists() and (d / f"snr{s}_decoded.txt").exists():
            t = (d / f"snr{s}_decoded.txt").read_text(encoding="utf-8").strip().splitlines()
            if len(t) >= LIMIT:
                ok.append(s)
    return ok


def rebuild_csv(prefix):
    d = RES / f"{prefix}_texts"
    rows = []
    for s in SNRS:
        tgt = (d / f"snr{s}_target.txt").read_text(encoding="utf-8").strip().splitlines()
        hyp = (d / f"snr{s}_decoded.txt").read_text(encoding="utf-8").strip().splitlines()
        row = {"snr": s}
        for n in (1, 2, 3, 4):
            w = tuple(1 if i == n - 1 else 0 for i in range(4))
            row[f"bleu{n}"] = float(np.mean([
                sentence_bleu([h.split()], r.split(), weights=w) for r, h in zip(tgt, hyp)]))
        rows.append(row)
    with open(RES / f"{prefix}_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV yeniden kuruldu: {prefix}_eval.csv")


def main():
    for prefix, channel in RUNS:
        done = done_snrs(prefix)
        missing = [s for s in SNRS if s not in done]
        print(f"{prefix}: bitmis={done} eksik={missing}")
        if missing:
            cmd = [sys.executable, str(ROOT / "src" / "llm_sc" / "eval_llm_sc.py"),
                   "--limit", str(LIMIT), "--channel", channel,
                   "--snr-list", ",".join(map(str, missing)),
                   "--out-prefix", prefix]
            r = subprocess.run(cmd)
            if r.returncode != 0:
                sys.exit(f"{prefix} kosusu hata verdi ({r.returncode})")
        if not [s for s in SNRS if s not in done_snrs(prefix)]:
            rebuild_csv(prefix)
    print("=== QWEN OLCUMU KOMPLE ===")


if __name__ == "__main__":
    main()
