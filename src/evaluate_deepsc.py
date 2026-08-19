# -*- coding: utf-8 -*-
"""
DeepSC değerlendirme runner'ı — third_party/DeepSC/performance.py'nin
parametreleştirilmiş kopyası. Akış birebir aynı: test seti üzerinde her SNR için
greedy_decode → SeqtoText → BLEU; farklar:
  - sabit yollar parametrik,
  - BLEU 1/2/3/4-gram tek geçişte hesaplanır (orijinal yalnız 1-gram),
  - sentence similarity HF transformers ile implemente edildi (orijinalde
    bert4keras'lı kod tamamen comment-out'tu). Makale Eq.13'teki saf kosinüs
    kullanılır: B_Φ = BERT 11. katman çıktısının token toplamı (padding hariç).
    Orijinal comment-out koddaki sklearn normalize(axis=0) adımı batch'e bağımlı
    olduğu için (metrik diğer cümlelere göre değişir) bilinçli olarak atlandı;
    orijinalden belgelenmiş bir sapmadır.
  - sonuçlar CSV'ye + örnek çözümlemeler txt'ye yazılır.
"""
import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "DeepSC"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dataset import collate_data  # noqa: E402  (third_party/DeepSC)
from models.transceiver import DeepSC  # noqa: E402
from utils import (BleuScore, Channels, PowerNormalize, SNR_to_noise,  # noqa: E402
                   SeqtoText, greedy_decode, subsequent_mask)

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=str(ROOT / "data" / "processed" / "europarl"))
parser.add_argument("--checkpoint-path", required=True, help=".pth dosyası veya checkpoint klasörü (en yenisi seçilir)")
parser.add_argument("--channel", default="AWGN", choices=["AWGN", "Rayleigh", "Rician"])
parser.add_argument("--MAX-LENGTH", default=30, type=int)
parser.add_argument("--d-model", default=128, type=int)
parser.add_argument("--dff", default=512, type=int)
parser.add_argument("--num-layers", default=4, type=int)
parser.add_argument("--num-heads", default=8, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--rounds", default=2, type=int, help="Değerlendirme tur sayısı (orijinal: 2)")
parser.add_argument("--snr-list", default="0,3,6,9,12,15,18")
parser.add_argument("--similarity", action="store_true", help="BERT sentence similarity da hesapla (HF transformers)")
parser.add_argument("--sim-model", default="bert-base-uncased")
parser.add_argument("--limit", type=int, default=None, help="Test setini ilk N cümleyle sınırla (smoke)")
parser.add_argument("--out-prefix", default=None, help="Varsayılan: deepsc-<kanal>")
parser.add_argument("--beam", type=int, default=0,
                    help="Beam genişliği (0=greedy, orijinal). utils.py:343'te yazarın "
                         "önerip yazmadığı beam search; kanal geçişi greedy_decode ile birebir aynı.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EurDataset(Dataset):
    def __init__(self, data_dir, split="test", limit=None):
        with open(Path(data_dir) / f"{split}_data.pkl", "rb") as f:
            self.data = pickle.load(f)
        if limit:
            self.data = self.data[:limit]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class HFSimilarity:
    """Makale Eq.13: match(s,ŝ)=cos(B_Φ(s),B_Φ(ŝ)). B_Φ: BERT katman-11 çıktısı,
    padding-maskeli token toplamı. (bert4keras'taki Encoder-11-FeedForward-Norm
    karşılığı = HF hidden_states[11].)"""

    def __init__(self, model_name):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()

    @torch.no_grad()
    def embed(self, sentences, batch_size=64):
        vecs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            # max_length 32 → 512 (2026-08-17 düzeltmesi): 32'de kesmek uzun
            # cümleleri bozuyordu ve LLM hattındaki ayardan farklıydı; tek protokol
            # için hepsi 512 (pratikte kesme yok).
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=512,
                                 return_tensors="pt").to(device)
            out = self.model(**enc)
            h = out.hidden_states[11]                      # [B, T, 768]
            mask = enc["attention_mask"].unsqueeze(-1)     # [B, T, 1]
            vecs.append((h * mask).sum(dim=1).cpu())       # token toplamı (pad hariç)
        return torch.cat(vecs)

    def compute(self, real, predicted):
        v1, v2 = self.embed(real), self.embed(predicted)
        cos = torch.nn.functional.cosine_similarity(v1, v2, dim=-1)
        return cos.numpy().tolist()


@torch.no_grad()
def beam_decode(model, src, n_var, max_len, padding_idx, start_symbol, end_idx,
                channel, beam_width=4):
    """greedy_decode (utils.py:341) ile AYNI kodlama+kanal+bellek akışı; tek fark
    çözümleme: K aday cümle paralel yürütülür, uzunluk-normalize log-olasılık
    toplamı en yüksek olan seçilir (Wu vd. tarzı, alpha=1). Bitmiş beam'ler
    <PAD> ile skor değişmeden taşınır (PAD pozisyonları maskede zaten kapalı)."""
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
    enc_output = model.encoder(src, src_mask)
    Tx_sig = PowerNormalize(model.channel_encoder(enc_output))
    Rx_sig = getattr(channels, channel)(Tx_sig, n_var)
    memory = model.channel_decoder(Rx_sig)

    B, K = src.size(0), beam_width
    mem = memory.unsqueeze(1).expand(B, K, *memory.shape[1:]).reshape(B * K, *memory.shape[1:])
    seqs = torch.full((B, K, 1), start_symbol, dtype=src.dtype, device=src.device)
    scores = torch.full((B, K), float("-inf"), device=src.device)
    scores[:, 0] = 0.0                      # ilk adımda beam'ler özdeş; teki canlı
    lens = torch.ones(B, K, device=src.device)
    finished = torch.zeros(B, K, dtype=torch.bool, device=src.device)

    for _ in range(max_len - 1):
        flat = seqs.view(B * K, -1)
        trg_mask = (flat == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
        la = subsequent_mask(flat.size(1)).type(torch.FloatTensor)
        comb = torch.max(trg_mask, la).to(device)
        dec = model.decoder(flat, mem, comb, None)
        logp = torch.log_softmax(model.dense(dec)[:, -1, :], dim=-1).view(B, K, -1)
        V = logp.size(-1)
        logp = logp.masked_fill(finished.unsqueeze(-1), float("-inf"))
        pad_col = logp[..., padding_idx]
        logp[..., padding_idx] = torch.where(finished, torch.zeros_like(pad_col), pad_col)
        cand = (scores.unsqueeze(-1) + logp).view(B, K * V)
        scores, idx = torch.topk(cand, K, dim=-1)
        beam_idx, tok_idx = idx // V, idx % V
        gather3 = beam_idx.unsqueeze(-1).expand(B, K, seqs.size(-1))
        seqs = torch.cat([torch.gather(seqs, 1, gather3), tok_idx.unsqueeze(-1)], dim=-1)
        finished = torch.gather(finished, 1, beam_idx)
        lens = torch.gather(lens, 1, beam_idx) + (~finished).float()
        finished = finished | (tok_idx == end_idx)
        if bool(finished.all()):
            break

    best = (scores / lens.clamp(min=1)).argmax(dim=-1)
    return seqs[torch.arange(B, device=src.device), best]


def resolve_checkpoint(path_str):
    p = Path(path_str)
    if p.is_file():
        return p
    cands = [(fn, int(fn.stem.split("_")[-1])) for fn in p.glob("checkpoint_*.pth")
             if fn.stem.split("_")[-1].isdigit()]
    if cands:
        return sorted(cands, key=lambda x: x[1])[-1][0]
    last = p / "checkpoint_last.pth"
    if last.exists():
        return last
    raise FileNotFoundError(f"Checkpoint bulunamadı: {p}")


def main():
    args = parser.parse_args()
    snr_list = [int(s) for s in args.snr_list.split(",")]
    out_prefix = args.out_prefix or f"deepsc-{args.channel}"

    vocab = json.load(open(Path(args.data_dir) / "vocab.json", "rb"))
    token_to_idx = vocab["token_to_idx"]
    idx_to = None  # SeqtoText kendi ters haritasını kurar
    num_vocab = len(token_to_idx)
    pad_idx, start_idx, end_idx = token_to_idx["<PAD>"], token_to_idx["<START>"], token_to_idx["<END>"]

    net = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab,
                 args.d_model, args.num_heads, args.dff, 0.1).to(device)
    ckpt = resolve_checkpoint(args.checkpoint_path)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    print(f"model yüklendi: {ckpt}")

    test_ds = EurDataset(args.data_dir, "test", limit=args.limit)
    test_iterator = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    StoT = SeqtoText(token_to_idx, end_idx)
    sim = HFSimilarity(args.sim_model) if args.similarity else None

    # BLEU hesaplayıcılar: bireysel n-gram (makale figürlerindeki eksen)
    bleu_fns = {f"bleu{n}": BleuScore(*[1 if i == n - 1 else 0 for i in range(4)]) for n in (1, 2, 3, 4)}

    rows = []          # (round, snr, bleu1..4, sim)
    samples_path = ROOT / "experiments" / "results" / f"{out_prefix}_samples.txt"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    sf = open(samples_path, "w", encoding="utf-8")

    with torch.no_grad():
        for rnd in range(args.rounds):
            for snr in tqdm(snr_list, desc=f"tur {rnd + 1}/{args.rounds}"):
                noise_std = SNR_to_noise(snr)
                decoded_all, target_all = [], []
                for sents in test_iterator:
                    sents = sents.to(device)
                    if args.beam and args.beam > 1:
                        out = beam_decode(net, sents, noise_std, args.MAX_LENGTH,
                                          pad_idx, start_idx, end_idx, args.channel,
                                          beam_width=args.beam)
                    else:
                        out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH,
                                            pad_idx, start_idx, args.channel)
                    decoded_all += list(map(StoT.sequence_to_text, out.cpu().numpy().tolist()))
                    target_all += list(map(StoT.sequence_to_text, sents.cpu().numpy().tolist()))

                row = {"round": rnd + 1, "snr": snr}
                for name, fn in bleu_fns.items():
                    row[name] = float(np.mean(fn.compute_blue_score(decoded_all, target_all)))
                if sim is not None:
                    row["similarity"] = float(np.mean(sim.compute(target_all, decoded_all)))
                rows.append(row)

                if rnd == 0:
                    sf.write(f"\n===== SNR {snr} dB =====\n")
                    for t, d in list(zip(target_all, decoded_all))[:5]:
                        sf.write(f"HEDEF : {t}\nÇÖZÜM : {d}\n---\n")
                    # Tam metin kaydı (LLM hattıyla aynı format) → benzerlik skorları
                    # tek ve ortak scriptle yeniden hesaplanabilsin diye.
                    txt_dir = samples_path.parent / f"{out_prefix}_texts"
                    txt_dir.mkdir(parents=True, exist_ok=True)
                    (txt_dir / f"snr{snr}_target.txt").write_text(
                        "\n".join(target_all), encoding="utf-8")
                    (txt_dir / f"snr{snr}_decoded.txt").write_text(
                        "\n".join(decoded_all), encoding="utf-8")
    sf.close()

    # Tur ortalamaları
    csv_path = ROOT / "experiments" / "results" / f"{out_prefix}_eval.csv"
    keys = [k for k in rows[0] if k != "round"]

    # SNR bazinda BIRLESTIR, uzerine yazma. Ayni on adla farkli bir SNR listesiyle
    # ikinci kez kosmak onceki noktalari siliyordu: LONG-deepsc-retrained'in
    # 0/6/12/18 noktalari 18 Agu'da tam boyle kaybedildi (eval_llm_sc.py'de ayni
    # duzeltme yapilmisti, bu betikte eksik kalmisti). Ayni SNR tekrar olculurse
    # YENI deger gecerli; farkli SNR'lar korunur.
    birlesik = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    birlesik[int(float(r["snr"]))] = r
                except (TypeError, ValueError, KeyError):
                    continue
        korunan = sorted(set(birlesik) - set(snr_list))
        if korunan:
            print(f"Mevcut dosyadaki {korunan} dB noktalari korunuyor.")
    for snr in snr_list:
        snr_rows = [r for r in rows if r["snr"] == snr]
        birlesik[int(snr)] = {k: (str(snr) if k == "snr" else
                                  f"{np.mean([r[k] for r in snr_rows]):.4f}")
                              for k in keys}
    alanlar = list(keys)
    for r in birlesik.values():
        for k in r:
            if k not in alanlar:
                alanlar.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=alanlar, restval="")
        w.writeheader()
        for snr in sorted(birlesik):
            w.writerow(birlesik[snr])

    print(f"Sonuçlar: {csv_path}\nÖrnek çözümlemeler: {samples_path}")
    for snr in snr_list:
        snr_rows = [r for r in rows if r["snr"] == snr]
        msg = f"SNR {snr:2d} dB → " + "  ".join(
            f"{k}={np.mean([r[k] for r in snr_rows]):.4f}" for k in keys if k != "snr")
        print(msg)


if __name__ == "__main__":
    main()
