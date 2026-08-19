# -*- coding: utf-8 -*-
"""
TÜRKÇE veri seti (ikincil katkı) — PARALEL kurgu.

Tasarım kararı: Türkçe ve İngilizce test cümleleri AYNI İÇERİĞİN çevirileri
(OPUS TED2020 en-tr paralel korpusu). Böylece dil etkisi izole olur: içerik
sabit, yalnız dil değişir. Europarl'da Türkçe olmadığı için paralel korpusa
geçildi; register farkı (parlamento → konferans konuşması) makalede belirtilir.

Üretilenler (data/processed/turkish/):
  test_sentences.txt      Türkçe test cümleleri (LLM hattı için)
  test_sentences_en.txt   Aynı cümlelerin İngilizcesi (kontrol kolu)
  train_data.pkl          DeepSC-TR eğitimi için (Türkçe, kelime düzeyi)
  test_data.pkl           DeepSC-TR değerlendirmesi (aynı test cümleleri)
  vocab.json              Türkçe sözlük (DeepSC pipeline formatı)
  stats.txt               sayılar + <UNK> oranı

Not: Türkçe'de küçük harfe çevirirken 'I'→'ı' kuralı gerekir (str.lower() yanlış
sonuç verir); bu yüzden özel dönüşüm kullanılıyor.
"""
import argparse
import json
import pickle
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SPECIAL = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}

parser = argparse.ArgumentParser()
parser.add_argument("--en", default=str(RAW / "TED2020.en-tr.en"))
parser.add_argument("--tr", default=str(RAW / "TED2020.en-tr.tr"))
parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed" / "turkish"))
parser.add_argument("--min-len", type=int, default=4)
parser.add_argument("--max-len", type=int, default=30)
parser.add_argument("--test-size", type=int, default=400)
parser.add_argument("--train-size", type=int, default=60000)


def tr_lower(s):
    """Türkçe'ye uygun küçük harf: I→ı, İ→i."""
    return s.replace("I", "ı").replace("İ", "i").lower()


def normalize_tr(s):
    s = tr_lower(s.strip())
    s = re.sub(r"([!.?])", r" \1", s)
    s = re.sub(r"[^a-zçğıöşü0-9.!?,;']+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_en(s):
    s = "".join(c for c in unicodedata.normalize("NFD", s.strip())
                if unicodedata.category(c) != "Mn").lower()
    s = re.sub(r"([!.?])", r" \1", s)
    s = re.sub(r"[^a-z0-9.!?,;']+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(s):
    return ["<START>"] + s.replace(".", "").replace("?", "").split() + ["<END>"]


def main():
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("paralel korpus okunuyor...")
    with open(args.en, encoding="utf-8") as f_en, open(args.tr, encoding="utf-8") as f_tr:
        pairs, seen = [], set()
        for le, lt in zip(f_en, f_tr):
            e, t = normalize_en(le), normalize_tr(lt)
            ne, nt = len(e.split()), len(t.split())
            if not (args.min_len < ne < args.max_len and args.min_len < nt < args.max_len):
                continue
            if t in seen:
                continue
            seen.add(t)
            pairs.append((e, t))
            if len(pairs) >= args.test_size + args.train_size:
                break
    print(f"{len(pairs)} paralel cümle çifti (her iki dilde de {args.min_len}-{args.max_len} kelime)")

    test = pairs[: args.test_size]
    train = pairs[args.test_size:]

    (out / "test_sentences.txt").write_text("\n".join(t for _, t in test), encoding="utf-8")
    (out / "test_sentences_en.txt").write_text("\n".join(e for e, _ in test), encoding="utf-8")

    # DeepSC-TR sözlüğü: yalnız EĞİTİM cümlelerinden (test sızıntısı olmasın)
    counts = {}
    for _, t in train:
        for w in tokenize(t):
            if w not in SPECIAL:
                counts[w] = counts.get(w, 0) + 1
    tok2idx = dict(SPECIAL)
    for w, _ in sorted(counts.items()):
        tok2idx[w] = len(tok2idx)
    json.dump({"token_to_idx": tok2idx}, open(out / "vocab.json", "w"))

    unk = tok2idx["<UNK>"]
    def encode(s):
        return [tok2idx.get(w, unk) for w in tokenize(s)]
    tr_train = [encode(t) for _, t in train]
    tr_test = [encode(t) for _, t in test]
    pickle.dump(tr_train, open(out / "train_data.pkl", "wb"))
    pickle.dump(tr_test, open(out / "test_data.pkl", "wb"))

    n_tok = sum(len(x) for x in tr_test)
    n_unk = sum(sum(1 for i in x if i == unk) for x in tr_test)
    stats = (f"paralel çift        : {len(pairs)}\n"
             f"eğitim / test       : {len(train)} / {len(test)}\n"
             f"Türkçe sözlük       : {len(tok2idx)}\n"
             f"test <UNK> oranı    : %{100*n_unk/n_tok:.2f}\n"
             f"kaynak              : OPUS TED2020 en-tr\n")
    (out / "stats.txt").write_text(stats, encoding="utf-8")
    print(stats)
    print("örnek çift:")
    print("  EN:", test[0][0])
    print("  TR:", test[0][1])


if __name__ == "__main__":
    main()
