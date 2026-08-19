# -*- coding: utf-8 -*-
"""DeepSC test pkl'ından düz metin cümle listesi üretir (LLM-SC ile ORTAK test seti).
`tez` ortamında çalıştırılır: conda run -n tez python src/llm_sc/dump_test_sentences.py"""
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "europarl"

vocab = json.load(open(DATA / "vocab.json"))
idx2tok = {v: k for k, v in vocab["token_to_idx"].items()}
test = pickle.load(open(DATA / "test_data.pkl", "rb"))

out = DATA / "test_sentences.txt"
with open(out, "w", encoding="utf-8") as f:
    for seq in test:
        words = [idx2tok[i] for i in seq]
        words = [w for w in words if w not in ("<START>", "<END>", "<PAD>")]
        f.write(" ".join(words) + "\n")
print(f"{len(test)} cümle → {out}")
