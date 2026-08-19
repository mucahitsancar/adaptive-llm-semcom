# -*- coding: utf-8 -*-
"""
LLM-SC değerlendirme hattı (temiz implementasyon, modern transformers).

Akış (LLM_com/test.ipynb protokolünün portu):
  cümle -> tokenizer -> sembol dizisi -> kanal -> alıcıda MAP beam search:
  skor(token) = log P_LLM(token | geçmiş) + log p(y | token)      [MAP füzyonu]
Alıcı, sembol sayısından token sayısını bilir (S sembol/token) -> tam L adım çözer.
Orijinaldeki "imkânsız token" maskesi korunur (--restrict-vocab, varsayılan açık):
test korpusunda hiç geçmeyen token id'leri aday dışı bırakılır.

Çıktılar: <out>_eval.csv (snr, bleu1-4, gecikme, vram) + <out>_samples.txt +
<out>_texts/ (SNR başına hedef/çözüm metinleri — benzerlik skoru sonradan
mevcut HFSimilarity hattıyla hesaplanır).
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu

from physics import Channel, build_constellation, channel_loglik, symbols_per_token

ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
parser.add_argument("--sentences", default=str(ROOT / "data" / "processed" / "europarl" / "test_sentences.txt"))
parser.add_argument("--limit", type=int, default=500)
parser.add_argument("--snr-list", default="0,3,6,9,12,15,18")
parser.add_argument("--channel", default="AWGN", choices=["AWGN", "Rayleigh"])
parser.add_argument("--beam", type=int, default=10, help="test.ipynb: num_beams=10")
parser.add_argument("--restrict-vocab", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--out-prefix", required=True)
parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=False,
                    help="KV/state onbellegi + beam yeniden siralama (transformers>=5 gerekir). "
                         "VARSAYILAN KAPALI: eski hucreler tam-ileri-gecis motoruyla olculdu. "
                         "Acmadan once --cache-check ile cikti esdegerligini kanitla.")
parser.add_argument("--cache-check", type=int, default=0, metavar="N",
                    help="N cumlede iki yolu AYNI kanal gerceklesmesiyle kosar, cozulen "
                         "token dizilerini birebir karsilastirir, hizi raporlar ve cikar.")
parser.add_argument("--quant", default="none", choices=["none", "int8", "int4"],
                    help="Nicemleme (bitsandbytes). NOT: bnb bellek optimize eder, "
                         "gecikmeyi düşürmez (ağırlıklar kullanımda geri açılır) — ölçülüp raporlanır.")

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def map_beam_decode(model, tok, y, h, n0, constellation, n_tokens, beam, allowed_mask,
                    n_allowed=None):
    """Tek cümle MAP beam çözümü. y,h: [n_tokens*S]; dönen: [n_tokens] token id.
    Not: Qwen3.5'in hibrit (linear attention) cache'i beam yeniden sıralamayı
    desteklemediğinden her adımda tam ileri geçiş yapılır (L≤30 için kabul edilebilir)."""
    S = constellation.shape[1]
    V = constellation.shape[0]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    seqs = torch.full((1, 1), bos, dtype=torch.long, device=device)
    scores = torch.zeros(1, device=device)
    k_cap = beam if n_allowed is None else min(beam, n_allowed)

    for t in range(n_tokens):
        y_step = y[t * S:(t + 1) * S]
        h_step = h[t * S:(t + 1) * S]
        ch_ll = channel_loglik(y_step, h_step, n0, constellation)      # [V]
        if allowed_mask is not None:
            ch_ll = ch_ll.masked_fill(~allowed_mask, float("-inf"))

        out = model(seqs)                                              # [B, t+1, V]
        lm_ll = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)  # [B, V]
        total = scores.unsqueeze(1) + lm_ll + ch_ll.unsqueeze(0)       # [B, V]
        scores, flat = torch.topk(total.view(-1), k_cap)
        beam_idx, tok_idx = flat // V, flat % V
        seqs = torch.cat([seqs[beam_idx], tok_idx.unsqueeze(1)], dim=1)

    return seqs[scores.argmax(), 1:]


@torch.no_grad()
def map_beam_decode_cached(model, tok, y, h, n0, constellation, n_tokens, beam,
                           allowed_mask, n_allowed=None):
    """map_beam_decode ile MATEMATIKSEL OLARAK AYNI cozum; farki yalniz hesabin
    tekrarlanmamasi: onek her adimda yeniden ileri gecirilmez, KV/state onbellegi
    tutulur ve topk sonrasi beam_idx ile yeniden siralanir.
    transformers 5.x'te hibrit (linear attention + full attention) katmanlar da
    reorder_cache destekliyor (LinearAttentionAndFullAttentionLayer) — orijinal
    kodun 'desteklenmiyor' notu bu surumde gecerli degil, ama kanit sart:
    --cache-check ile iki yolun ciktisi birebir karsilastirilmali."""
    S = constellation.shape[1]
    V = constellation.shape[0]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    seqs = torch.full((1, 1), bos, dtype=torch.long, device=device)
    scores = torch.zeros(1, device=device)
    k_cap = beam if n_allowed is None else min(beam, n_allowed)
    past, girdi = None, seqs

    for t in range(n_tokens):
        y_step = y[t * S:(t + 1) * S]
        h_step = h[t * S:(t + 1) * S]
        ch_ll = channel_loglik(y_step, h_step, n0, constellation)
        if allowed_mask is not None:
            ch_ll = ch_ll.masked_fill(~allowed_mask, float("-inf"))

        out = model(girdi, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        lm_ll = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        total = scores.unsqueeze(1) + lm_ll + ch_ll.unsqueeze(0)
        scores, flat = torch.topk(total.view(-1), k_cap)
        beam_idx, tok_idx = flat // V, flat % V
        seqs = torch.cat([seqs[beam_idx], tok_idx.unsqueeze(1)], dim=1)
        past.reorder_cache(beam_idx)      # ilk adimda 1 satir -> k satira da buyutur
        girdi = tok_idx.unsqueeze(1)

    return seqs[scores.argmax(), 1:]


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    snr_list = [int(s) for s in args.snr_list.split(",")]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.quant == "none":
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    else:
        from transformers import BitsAndBytesConfig
        qc = (BitsAndBytesConfig(load_in_8bit=True) if args.quant == "int8" else
              BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=dtype))
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=qc,
                                                    device_map="auto").eval()
    V = model.config.vocab_size
    S = symbols_per_token(V)
    print(f"model={args.model} V={V} bit/token={S*3} sembol/token={S} cihaz={device}")

    all_lines = [l.strip() for l in open(args.sentences, encoding="utf-8") if l.strip()]
    sents = all_lines[: args.limit]
    enc = [torch.tensor(tok(s).input_ids, device=device) for s in sents]

    allowed = None
    if args.restrict_vocab:
        # LLM_com protokolü: aday küme TÜM korpustan kurulur (test edilen altkümeden değil)
        allowed = torch.zeros(V, dtype=torch.bool, device=device)
        for s in all_lines:
            allowed[torch.tensor(tok(s).input_ids, device=device)] = True
        print(f"restrict-vocab: {int(allowed.sum())}/{V} token aday (tam dosyadan)")

    constellation = build_constellation(V, device)
    n_allowed = int(allowed.sum()) if allowed is not None else None
    cozucu = map_beam_decode_cached if args.use_cache else map_beam_decode
    print(f"cozucu: {'ONBELLEKLI (--use-cache)' if args.use_cache else 'tam ileri gecis (varsayilan)'}")

    if args.cache_check:
        # Iki yolu AYNI kanal gerceklesmesiyle kosar: tx/y/h/n0 bir kez uretilir.
        ch = Channel(snr_list[0], args.channel, device, gen)
        esit, ayrik, sure = 0, [], {"tam": 0.0, "onbellek": 0.0}
        for i, ids in enumerate(enc[: args.cache_check], 1):
            tx = constellation[ids].reshape(-1)
            y, h, n0 = ch.transmit(tx)
            ciktilar = {}
            for ad, fn in (("tam", map_beam_decode), ("onbellek", map_beam_decode_cached)):
                torch.cuda.synchronize() if device == "cuda" else None
                t0 = time.time()
                ciktilar[ad] = fn(model, tok, y, h, n0, constellation, len(ids),
                                  args.beam, allowed, n_allowed=n_allowed)
                torch.cuda.synchronize() if device == "cuda" else None
                sure[ad] += time.time() - t0
            ayni = torch.equal(ciktilar["tam"], ciktilar["onbellek"])
            esit += int(ayni)
            if not ayni:
                ayrik.append((i, tok.decode(ciktilar["tam"], skip_special_tokens=True),
                              tok.decode(ciktilar["onbellek"], skip_special_tokens=True)))
            print(f"  cumle {i}: {'AYNI' if ayni else 'FARKLI'}", flush=True)
        n = args.cache_check
        print("")
        print(f"=== CACHE-CHECK ({args.channel} {snr_list[0]} dB, n={n}) ===")
        print(f"birebir ayni token dizisi: {esit}/{n}")
        print(f"tam ileri gecis : {sure['tam']/n:.2f} sn/cumle")
        print(f"onbellekli      : {sure['onbellek']/n:.2f} sn/cumle "
              f"(hizlanma x{sure['tam']/max(sure['onbellek'], 1e-9):.1f})")
        for i, a, b in ayrik[:3]:
            print(f"  --- fark, cumle {i} ---")
            print(f"  tam      : {a}")
            print(f"  onbellek : {b}")
        print("SONUC: " + ("esdeger, --use-cache guvenle acilabilir." if esit == n else
                           "ESDEGER DEGIL — --use-cache ACMA, olcumler karsilastirilamaz."))
        return

    res_dir = ROOT / "experiments" / "results"
    txt_dir = res_dir / f"{args.out_prefix}_texts"
    txt_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # "a" ile ac: ayni on adla ikinci kez kosuldugunda (or. baska SNR listesi)
    # onceki orneklerin uzerine yazilmasin. Ayni SNR tekrar olculurse
    # dosyada iki bolum olur; asil delil zaten <prefix>_texts/ altindadir.
    sf = open(res_dir / f"{args.out_prefix}_samples.txt", "a", encoding="utf-8")

    for snr in snr_list:
        ch = Channel(snr, args.channel, device, gen)
        hyps, t0 = [], time.time()
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        prog = res_dir / "PROGRESS.txt"
        for i, ids in enumerate(enc, 1):
            tx = constellation[ids].reshape(-1)          # [L*S]
            y, h, n0 = ch.transmit(tx)
            out_ids = cozucu(model, tok, y, h, n0, constellation,
                             len(ids), args.beam, allowed, n_allowed=n_allowed)
            hyps.append(tok.decode(out_ids, skip_special_tokens=True))
            if i % 5 == 0 or i == len(enc):
                hız = (time.time() - t0) / i
                kalan = hız * (len(enc) - i) / 60
                satır = (f"{args.out_prefix} ({args.quant}) | {args.channel} {snr} dB | "
                         f"{i}/{len(enc)} cümle | {hız:.1f} sn/cümle | bu nokta ~{kalan:.0f} dk sonra biter")
                prog.write_text(satır + "\n", encoding="utf-8")
                if i % 25 == 0:
                    print(satır, flush=True)
        dt = (time.time() - t0) / len(enc)
        vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

        bleu = {n: float(np.mean([
            sentence_bleu([h.split()], r.split(),
                          weights=tuple(1 if i == n - 1 else 0 for i in range(4)))
            for r, h in zip(sents, hyps)])) for n in (1, 2, 3, 4)}
        rows.append({"snr": snr, **{f"bleu{n}": bleu[n] for n in (1, 2, 3, 4)},
                     "saniye_per_cumle": dt, "vram_gb": vram})
        print(f"SNR {snr:2d} -> B1={bleu[1]:.4f} B4={bleu[4]:.4f}  {dt:.2f} sn/cümle  {vram:.2f} GB")

        # SATIR SONU KACIRMA - SART: aday maskesi kapali kosularda model satir
        # sonu tokeni uretebiliyor ve metin dosyasinin satir hizasi bozuluyor
        # (olculdu: 0 dB'de 149 hedef satira karsi 413 cozum satiri). Hiza
        # bozulunca cumle bazli analiz (bootstrap, geri hesaplama) gecersiz olur.
        def _tek_satir(x):
            return x.replace("\r", " ").replace("\n", " \\n ")
        with open(txt_dir / f"snr{snr}_target.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(_tek_satir(x) for x in sents))
        with open(txt_dir / f"snr{snr}_decoded.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(_tek_satir(x) for x in hyps))
        sf.write(f"\n===== SNR {snr} dB =====\n")
        for r, h in list(zip(sents, hyps))[:5]:
            sf.write(f"HEDEF : {r}\nÇÖZÜM : {h}\n---\n")
    sf.close()

    # SNR bazında BİRLEŞTİR, üzerine yazma. Neden: aynı ön ada farklı SNR
    # listesiyle ikinci kez koşmak (ör. ince kesişim ölçümü) önceki noktaları
    # siliyordu — LONG-qwen08b'de 4 nokta bu yüzden kayboldu, ancak cümle bazlı
    # metinlerden geri getirilebildi (LOG 2026-08-18). Aynı SNR tekrar ölçülürse
    # YENİ değer geçerli; farklı SNR'lar korunur.
    out_csv = res_dir / f"{args.out_prefix}_eval.csv"
    birlesik = {}
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    birlesik[int(float(r["snr"]))] = r
                except (ValueError, KeyError, TypeError):
                    continue
        korunan = sorted(set(birlesik) - {int(r["snr"]) for r in rows})
        if korunan:
            print(f"Mevcut dosyadaki {korunan} dB noktaları korunuyor.")
    for r in rows:
        birlesik[int(r["snr"])] = r
    alanlar = list(rows[0].keys())
    for r in birlesik.values():                     # eski satırların ek kolonları
        for k in r:
            if k not in alanlar:
                alanlar.append(k)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=alanlar, restval="")
        w.writeheader()
        for snr in sorted(birlesik):
            w.writerow(birlesik[snr])
    print(f"Sonuç: {out_csv} ({len(birlesik)} nokta)")


if __name__ == "__main__":
    main()
