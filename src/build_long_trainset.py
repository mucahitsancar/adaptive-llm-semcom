# -*- coding: utf-8 -*-
"""
UZUN CÜMLE eğitim seti (B kolu: DeepSC'yi kendi rejiminde yeniden eğitmek).

Amaç: "DeepSC uzun cümlede çöküyor" bulgusunun, eğitim eksikliğinden değil
mimari kapasiteden geldiğini göstermek. Bunun için DeepSC uzun cümlelerle
SIFIRDAN eğitilir ve aynı uzun test setinde ölçülür.

Tasarım kararları (adil karşılaştırma için):
  - Sözlük AYNI (mevcut 22.234 kelimelik vocab) → çıktı uzayı değişmiyor,
    tek değişken cümle uzunluğu rejimi. Sözlük dışı kelime → <UNK>.
  - Eğitim cümleleri, uzun TEST setiyle KESİŞMEZ (sızıntı yok).
  - Aynı uzunluk filtresi: 30 < len < 70 kelime.
Çıktı: data/processed/europarl_long/train_data.pkl (test_data.pkl zaten var)
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "DeepSC"))

from preprocess_text import normalize_string, tokenize  # noqa: E402
from tqdm import tqdm  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", default=str(ROOT / "data" / "raw" / "europarl" / "txt" / "en"))
parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed" / "europarl_long"))
parser.add_argument("--min-len", type=int, default=30)
parser.add_argument("--max-len", type=int, default=70)
parser.add_argument("--limit", type=int, default=30000, help="Eğitim cümlesi sayısı")


def main():
    args = parser.parse_args()
    out = Path(args.out_dir)
    vocab = json.load(open(out / "vocab.json"))
    tok2idx = vocab["token_to_idx"]
    unk = tok2idx["<UNK>"]

    test_sents = set((out / "test_sentences.txt").read_text(encoding="utf-8").strip().splitlines())
    print(f"test setinde {len(test_sents)} cümle var (bunlar eğitime GİRMEYECEK)")

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".txt"))
    seen, sents = set(), []
    for fn in tqdm(files, desc="dosyalar"):
        with open(os.path.join(args.input_dir, fn), encoding="utf8") as f:
            for line in f.read().strip().split("\n"):
                s = normalize_string(line)
                n = len(s.split())
                if args.min_len < n < args.max_len and s not in seen and s not in test_sents:
                    seen.add(s)
                    sents.append(s)
        if len(sents) >= args.limit:
            break
    sents = sents[: args.limit]

    encoded, unk_tot, tok_tot = [], 0, 0
    for s in tqdm(sents, desc="encode"):
        words = tokenize(s, punct_to_keep=[";", ","], punct_to_remove=["?", "."])
        ids = [tok2idx.get(w, unk) for w in words]
        unk_tot += sum(1 for i in ids if i == unk)
        tok_tot += len(ids)
        encoded.append(ids)

    with open(out / "train_data.pkl", "wb") as f:
        pickle.dump(encoded, f)
    lens = [len(s.split()) for s in sents]
    msg = (f"eğitim cümlesi   : {len(sents)}\n"
           f"uzunluk          : {min(lens)}-{max(lens)} (ort {sum(lens)/len(lens):.1f})\n"
           f"<UNK> oranı      : %{100*unk_tot/tok_tot:.2f}\n")
    print(msg)
    (out / "stats_train.txt").write_text(msg, encoding="utf-8")


if __name__ == "__main__":
    main()
