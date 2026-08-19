# -*- coding: utf-8 -*-
"""
UZUN CÜMLE test seti (payload-length rejim deneyi).

Gerekçe: LLM-SC (Wang vd. 2025) Europarl'dan 60-70 KELİMELİK cümleler kullanıyor;
DeepSC ise MAX_LENGTH=30 ile eğitiliyor. Kesişim SNR'ındaki farkın kaynağını test
etmek için 30<len<70 kelimelik bir test seti kuruyoruz.

Üretilenler (data/processed/europarl_long/):
  - test_data.pkl        : DeepSC formatı (mevcut vocab ile, bilinmeyen → <UNK>)
  - vocab.json           : mevcut vocab'ın kopyası (DeepSC checkpoint'leri bununla eğitildi)
  - test_sentences.txt   : LLM tarafı için düz metin (aynı cümleler, aynı sıra)
  - stats.txt            : cümle sayısı, uzunluk dağılımı, <UNK> oranı (dürüstlük kaydı)

Not: <UNK> oranı raporlanır — DeepSC'nin uzun cümledeki dezavantajının ne kadarı
uzunluktan, ne kadarı sözlük dışı kelimelerden geldiğini ayırt edebilmek için.
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
parser.add_argument("--vocab", default=str(ROOT / "data" / "processed" / "europarl" / "vocab.json"))
parser.add_argument("--min-len", type=int, default=30)
parser.add_argument("--max-len", type=int, default=70)
parser.add_argument("--max-files", type=int, default=400)
parser.add_argument("--limit", type=int, default=400, help="Kaç uzun cümle tutulacak")


def main():
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vocab = json.load(open(args.vocab))
    tok2idx = vocab["token_to_idx"]
    unk = tok2idx["<UNK>"]

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".txt"))[: args.max_files]
    seen, sents = set(), []
    for fn in tqdm(files, desc="dosyalar"):
        with open(os.path.join(args.input_dir, fn), encoding="utf8") as f:
            for line in f.read().strip().split("\n"):
                s = normalize_string(line)
                n = len(s.split())
                if args.min_len < n < args.max_len and s not in seen:
                    seen.add(s)
                    sents.append(s)
        if len(sents) >= args.limit:
            break
    sents = sents[: args.limit]

    encoded, unk_tot, tok_tot = [], 0, 0
    for s in sents:
        words = tokenize(s, punct_to_keep=[";", ","], punct_to_remove=["?", "."])
        ids = []
        for w in words:
            i = tok2idx.get(w, unk)
            if i == unk and w != "<UNK>":
                unk_tot += 1
            ids.append(i)
        tok_tot += len(ids)
        encoded.append(ids)

    with open(out / "test_data.pkl", "wb") as f:
        pickle.dump(encoded, f)
    with open(out / "vocab.json", "w") as f:
        json.dump(vocab, f)
    with open(out / "test_sentences.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sents))

    lens = [len(s.split()) for s in sents]
    stats = (f"cümle sayısı      : {len(sents)}\n"
             f"uzunluk aralığı   : {min(lens)}-{max(lens)} kelime\n"
             f"ortalama uzunluk  : {sum(lens)/len(lens):.1f} kelime\n"
             f"token/cümle (ort) : {tok_tot/len(sents):.1f}\n"
             f"<UNK> oranı       : %{100*unk_tot/tok_tot:.2f}\n")
    (out / "stats.txt").write_text(stats, encoding="utf-8")
    print(stats)
    print(f"yazıldı: {out}")


if __name__ == "__main__":
    main()
