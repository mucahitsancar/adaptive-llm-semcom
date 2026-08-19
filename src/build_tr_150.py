# -*- coding: utf-8 -*-
"""
Türkçe test setinin İLK 150 CÜMLESİNİ içeren ayrı bir DeepSC veri seti üret.

Neden gerekli: DeepSC hattı `test_data.pkl`'i uzunluğa göre gruplayarak işliyor,
bu yüzden `--limit 150` verildiğinde dosya düzeninin ilk 150'si değil, sıralanmış
düzenin ilk 150'si alınıyor. Sonuç: DeepSC-TR ile LLM-TR farklı cümle
altkümelerinde ölçüldü (kesişim 3/150) ve sayıları kıyaslanamaz hale geldi.

Çözüm: test setini baştan 150 cümleye indirip ayrı klasöre yazmak. Böylece
gruplama ne yaparsa yapsın küme LLM'in kümesiyle birebir aynı olur.
"""
import json
import pickle
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "turkish"
DST = ROOT / "data" / "processed" / "turkish_150"
N = 150

DST.mkdir(parents=True, exist_ok=True)

# Sözlük ve eğitim seti aynen: model bu sözlükle eğitildi, değiştirilemez.
shutil.copy(SRC / "vocab.json", DST / "vocab.json")
shutil.copy(SRC / "train_data.pkl", DST / "train_data.pkl")

test = pickle.load(open(SRC / "test_data.pkl", "rb"))[:N]
pickle.dump(test, open(DST / "test_data.pkl", "wb"))

cumleler = (SRC / "test_sentences.txt").read_text(encoding="utf-8").splitlines()[:N]
(DST / "test_sentences.txt").write_text("\n".join(cumleler), encoding="utf-8")

vocab = json.load(open(SRC / "vocab.json"))["token_to_idx"]
idx2tok = {v: k for k, v in vocab.items()}
unk = vocab["<UNK>"]
n_tok = sum(len(x) for x in test)
n_unk = sum(sum(1 for i in x if i == unk) for x in test)

print(f"test: {len(test)} cümle (kaynak dosyanın ilk {N}'i)")
print(f"token: {n_tok} | <UNK>: {n_unk} (%{100*n_unk/n_tok:.2f})")
print(f"ilk cümle (pkl'den çözülmüş): "
      f"{' '.join(idx2tok[i] for i in test[0][:12])}")
print(f"ilk cümle (txt'den)         : {cumleler[0][:70]}")
print(f"\nyazıldı → {DST}")
