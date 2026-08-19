# -*- coding: utf-8 -*-
"""
Europarl preprocess runner — third_party/DeepSC/preprocess_text.py'deki orijinal
fonksiyonları (process, build_vocab, tokenize) BIREBIR import eder; yalnızca
sabit sunucu yollarını (/import/antennas/...) kendi yollarımızla değiştirir.

Orijinalden bilinçli sapmalar (reproduction notu):
  1. os.listdir yerine sorted(os.listdir) — dosya sırası platforma bağlı
     olmasın diye (train/test 90/10 split'i cümle sırasına bağlı).
  2. --max-files parametresi eklendi (smoke test için).
Bunların dışında akış main() ile aynıdır: process → dedup → build_vocab
(punct_to_keep=[';',','], punct_to_remove=['?','.']) → encode → %90/%10 split.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "DeepSC"))

from preprocess_text import SPECIAL_TOKENS, build_vocab, process, tokenize  # noqa: E402
from tqdm import tqdm  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--input-data-dir", default=str(ROOT / "data" / "raw" / "europarl" / "txt" / "en"))
parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed" / "europarl"))
parser.add_argument("--max-files", type=int, default=None, help="Smoke test: yalnızca ilk N dosyayı işle")


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_list = sorted(fn for fn in os.listdir(args.input_data_dir) if fn.endswith(".txt"))
    if args.max_files:
        file_list = file_list[: args.max_files]
    print(f"{len(file_list)} dosya islenecek: {args.input_data_dir}")

    sentences = []
    print("Preprocess Raw Text")
    for fn in tqdm(file_list):
        sentences += process(os.path.join(args.input_data_dir, fn))

    # Orijinal main() ile ayni dedup (dict insertion order korunur)
    a = {}
    for s in sentences:
        if s not in a:
            a[s] = 0
        a[s] += 1
    sentences = list(a.keys())
    print("Number of sentences: {}".format(len(sentences)))

    print("Build Vocab")
    token_to_idx = build_vocab(
        sentences, dict(SPECIAL_TOKENS),
        punct_to_keep=[";", ","], punct_to_remove=["?", "."],
    )
    vocab = {"token_to_idx": token_to_idx}
    print("Number of words in Vocab: {}".format(len(token_to_idx)))

    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab, f)

    print("Start encoding txt")
    results = []
    for seq in tqdm(sentences):
        words = tokenize(seq, punct_to_keep=[";", ","], punct_to_remove=["?", "."])
        tokens = [token_to_idx[word] for word in words]
        results.append(tokens)

    print("Writing Data")
    train_data = results[: round(len(results) * 0.9)]
    test_data = results[round(len(results) * 0.9):]

    with open(out_dir / "train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open(out_dir / "test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)
    print(f"Bitti: train={len(train_data)} test={len(test_data)} → {out_dir}")


if __name__ == "__main__":
    main(parser.parse_args())
