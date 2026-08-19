# -*- coding: utf-8 -*-
"""
DeepSC eğitim runner'ı — third_party/DeepSC/main.py'nin eğitim döngüsünün
parametreleştirilmiş kopyası. Model/kayıp/optimizasyon mantığının TAMAMI
orijinal repodan import edilir (models.transceiver.DeepSC, utils.train_step,
utils.val_step, utils.train_mi, models.mutual_info.Mine); burada yalnızca:
  - sabit veri yolu kendi yolumuzla değiştirildi (EurDataset yeniden yazıldı,
    dataset.py:17'deki sınıfla birebir aynı davranış),
  - checkpoint kaydı "her epoch" yerine "en iyi val loss + son epoch" olarak
    düzenlendi (orijinal mantık record_acc bug'ı yüzünden her epoch kaydediyor),
  - epoch başına val loss CSV'ye loglanıyor,
  - --smoke ve --mine anahtarları eklendi.
Hiperparametreler main.py varsayılanlarıyla aynı: d_model=128, dff=512,
num_layers=4, heads=8, batch=128, epochs=80, Adam(lr=1e-4, betas=(0.9,0.98),
eps=1e-8, wd=5e-4), MI optimizasyonu Adam(lr=1e-3), epoch başına gürültü
U(SNR_to_noise(5), SNR_to_noise(10)); MINE modunda n_var=0.1 (main.py ile aynı).
"""
import argparse
import csv
import json
import math
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "DeepSC"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dataset import collate_data  # noqa: E402  (third_party/DeepSC)
from models.mutual_info import Mine, sample_batch  # noqa: E402
from models.transceiver import DeepSC  # noqa: E402
from utils import (Channels, PowerNormalize, SNR_to_noise, create_masks,  # noqa: E402
                   initNetParams, loss_function, train_mi, train_step, val_step)

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=str(ROOT / "data" / "processed" / "europarl"))
parser.add_argument("--checkpoint-dir", default=None, help="Varsayılan: checkpoints/deepsc-<kanal>[-mine]")
parser.add_argument("--channel", default="AWGN", choices=["AWGN", "Rayleigh", "Rician"])
parser.add_argument("--MAX-LENGTH", default=30, type=int)
parser.add_argument("--MIN-LENGTH", default=4, type=int)
parser.add_argument("--d-model", default=128, type=int)
parser.add_argument("--dff", default=512, type=int)
parser.add_argument("--num-layers", default=4, type=int)
parser.add_argument("--num-heads", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--epochs", default=80, type=int)
parser.add_argument("--mine", action="store_true", help="MINE ile karşılıklı bilgi maksimizasyonunu aç")
parser.add_argument("--run-name", default=None,
                    help="Koşu adı (checkpoint klasörü + CSV öneki). BASELINE KORUMASI: "
                         "deney koşularında mutlaka özgün ad ver; baseline adları "
                         "(deepsc-AWGN, deepsc-Rayleigh) ile çakışma engellenir.")
parser.add_argument("--smoke", action="store_true", help="Mini subset (2000 train/500 test) + 1 epoch")
parser.add_argument("--stable-mi", action="store_true",
                    help="MINE exp() taşmasına karşı kırpma+sonluluk koruması (belgeli sapma)")
parser.add_argument("--wide-snr", action="store_true",
                    help="Eğitim gürültüsünü epoch başına U(5,10dB) yerine BATCH başına U(0,18dB) örnekle (belgeli sapma)")
parser.add_argument("--per-sentence-h", action="store_true",
                    help="Rayleigh'de H'yi batch başına değil cümle başına çek (belgeli sapma)")
parser.add_argument("--seed", type=int, default=None, help="Verilirse tohum sabitlenir (orijinalde kapalı)")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EurDataset(Dataset):
    """dataset.py:15'teki sınıfla aynı; yalnızca yol parametrik."""

    def __init__(self, data_dir, split="train", limit=None):
        with open(Path(data_dir) / f"{split}_data.pkl", "rb") as f:
            self.data = pickle.load(f)
        if limit:
            self.data = self.data[:limit]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------------------
# STABILIZE MI (--stable-mi): utils.py:280 train_mi ve utils.py:228 train_step'in
# birebir kopyaları; TEK fark MINE'ın DV alt sınırındaki exp() taşmasına karşı
# iki koruma (belgelenmiş sapma — eğitimde gözlenen NaN bulgusu):
#   (1) mine_net çıktısı exp'ten önce [-20, 20] aralığına kırpılır (standart
#       MINE stabilizasyonu; exp(20)≈4.8e8 taşmaz),
#   (2) MI terimi sonlu değilse o batch'te kayba eklenmez / adım atlanır.
# Orijinal davranış --stable-mi verilmezse AYNEN korunur (utils fonksiyonları).
# ---------------------------------------------------------------------------
class PerSentenceChannels:
    """utils.Channels ile aynı arayüz; TEK fark: Rayleigh'de H, batch başına tek
    değil CÜMLE BAŞINA bağımsız çekilir (belgeli sapma — Faz 1.5 deneyi).
    Gerekçe: orijinal kodda 128 cümle aynı sönümlemeyi görür → epoch başına ~517
    kanal örneği; cümle başına H ile ~66k örnek → kanal çeşitliliği artar."""

    def AWGN(self, Tx_sig, n_var):
        return Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        B = shape[0]
        H_real = torch.normal(0, math.sqrt(1 / 2), size=(B,)).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=(B,)).to(device)
        H = torch.stack([torch.stack([H_real, -H_imag], dim=-1),
                         torch.stack([H_imag, H_real], dim=-1)], dim=-2)  # [B,2,2]
        x = Tx_sig.view(B, -1, 2)
        x = torch.matmul(x, H)                                   # kanal
        x = x + torch.normal(0, n_var, size=x.shape).to(device)  # gürültü
        x = torch.matmul(x, torch.inverse(H))                    # perfect CSI
        return x.view(shape)

    def Rician(self, Tx_sig, n_var, K=1):
        raise NotImplementedError("per-sentence Rician bu deneyde kapsam dışı")


def stable_mutual_information(joint, marginal, mine_net, clamp=20.0):
    t = torch.clamp(mine_net(joint), -clamp, clamp)
    et = torch.exp(torch.clamp(mine_net(marginal), -clamp, clamp))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def stable_train_mi(model, mi_net, src, n_var, padding_idx, opt, channel, channels=None):
    mi_net.train()
    opt.zero_grad()
    channels = channels or Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
    joint, marginal = sample_batch(Tx_sig, Rx_sig)
    mi_lb, _, _ = stable_mutual_information(joint, marginal, mi_net)
    loss_mine = -mi_lb
    if not torch.isfinite(loss_mine):
        return float("nan")  # adım atla; çağıran taraf sayaç tutar
    loss_mine.backward()
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)
    opt.step()
    return loss_mine.item()


def stable_train_step(model, src, trg, n_var, pad, opt, criterion, channel, mi_net=None, channels=None):
    model.train()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    channels = channels or Channels()
    opt.zero_grad()
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1), pad, criterion)
    if mi_net is not None:
        mi_net.eval()
        joint, marginal = sample_batch(Tx_sig, Rx_sig)
        mi_lb, _, _ = stable_mutual_information(joint, marginal, mi_net)
        if torch.isfinite(mi_lb):
            loss = loss + 0.0009 * (-mi_lb)  # λ orijinaldeki gibi (utils.py:271)
    loss.backward()
    opt.step()
    return loss.item()


def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_validate(epoch, args, net, test_ds, pad_idx, criterion):
    test_iterator = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, 0.1, pad_idx, criterion, args.channel)
            total += loss
            pbar.set_description(f"Epoch: {epoch + 1}; Type: VAL; Loss: {loss:.5f}")
    return total / len(test_iterator)


def run_train_epoch(epoch, args, net, train_ds, pad_idx, criterion, optimizer, mi_net=None, mi_opt=None):
    train_iterator = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1,))
    channels_obj = PerSentenceChannels() if args.per_sentence_h else None
    use_custom = args.stable_mi or args.wide_snr or args.per_sentence_h
    fn_train_mi = stable_train_mi if use_custom else train_mi
    fn_train_step = stable_train_step if use_custom else train_step

    for sents in pbar:
        sents = sents.to(device)
        # wide-snr: batch başına 0-18 dB aralığından örnekle (SNR_to_noise ters orantılı)
        nv = float(np.random.uniform(SNR_to_noise(18), SNR_to_noise(0))) if args.wide_snr else noise_std[0]
        if mi_net is not None:
            mi_nv = nv if args.wide_snr else 0.1
            if use_custom:
                mi = fn_train_mi(net, mi_net, sents, mi_nv, pad_idx, mi_opt, args.channel, channels=channels_obj)
                loss = fn_train_step(net, sents, sents, mi_nv, pad_idx, optimizer, criterion,
                                     args.channel, mi_net, channels=channels_obj)
            else:
                mi = fn_train_mi(net, mi_net, sents, mi_nv, pad_idx, mi_opt, args.channel)
                loss = fn_train_step(net, sents, sents, mi_nv, pad_idx, optimizer, criterion,
                                     args.channel, mi_net)
            pbar.set_description(f"Epoch: {epoch + 1}; Type: Train; Loss: {loss:.5f}; MI {mi:.5f}")
        else:
            if use_custom:
                loss = stable_train_step(net, sents, sents, nv, pad_idx, optimizer,
                                         criterion, args.channel, channels=channels_obj)
            else:
                loss = train_step(net, sents, sents, nv, pad_idx, optimizer,
                                  criterion, args.channel)
            pbar.set_description(f"Epoch: {epoch + 1}; Type: Train; Loss: {loss:.5f}")


def main():
    args = parser.parse_args()
    if args.seed is not None:
        setup_seed(args.seed)
    if args.smoke:
        args.epochs = min(args.epochs, 1)

    run_name = args.run_name or (
        f"deepsc-{args.channel}{'-mine' if args.mine else ''}"
        f"{'-widesnr' if args.wide_snr else ''}{'-perH' if args.per_sentence_h else ''}"
        f"{'-smoke' if args.smoke else ''}")
    # Baseline koruması: tamamlanmış baseline koşularının üzerine yazmayı reddet
    PROTECTED = {"deepsc-AWGN", "deepsc-Rayleigh"}
    if run_name in PROTECTED and (ROOT / "checkpoints" / run_name / "checkpoint_last.pth").exists():
        raise SystemExit(f"HATA: '{run_name}' korunan baseline. Yeni deney için --run-name ver.")
    ckpt_dir = Path(args.checkpoint_dir or (ROOT / "checkpoints" / run_name))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_csv = ROOT / "experiments" / "results" / f"{run_name}_train.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    vocab = json.load(open(Path(args.data_dir) / "vocab.json", "rb"))
    token_to_idx = vocab["token_to_idx"]
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]

    train_ds = EurDataset(args.data_dir, "train", limit=2000 if args.smoke else None)
    test_ds = EurDataset(args.data_dir, "test", limit=500 if args.smoke else None)
    print(f"cihaz={device}  vocab={num_vocab}  train={len(train_ds)}  test={len(test_ds)}  "
          f"kanal={args.channel}  MINE={args.mine}  epochs={args.epochs}")

    # main.py:113-122 ile birebir ayni kurulum
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab,
                    args.d_model, args.num_heads, args.dff, 0.1).to(device)
    mi_net = Mine().to(device) if args.mine else None
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(deepsc.parameters(), lr=1e-4, betas=(0.9, 0.98),
                                 eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3) if args.mine else None
    initNetParams(deepsc)

    best_val = float("inf")
    with open(results_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "val_loss", "epoch_sure_sn"])

    for epoch in range(args.epochs):
        start = time.time()
        run_train_epoch(epoch, args, deepsc, train_ds, pad_idx, criterion, optimizer, mi_net, mi_opt)
        avg_val = run_validate(epoch, args, deepsc, test_ds, pad_idx, criterion)
        elapsed = time.time() - start

        with open(results_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_val:.6f}", f"{elapsed:.1f}"])
        print(f"epoch {epoch + 1}: val_loss={avg_val:.5f}  sure={elapsed:.1f}s")

        torch.save(deepsc.state_dict(), ckpt_dir / "checkpoint_last.pth")
        if avg_val < best_val:
            best_val = avg_val
            torch.save(deepsc.state_dict(), ckpt_dir / f"checkpoint_{str(epoch + 1).zfill(2)}.pth")

    print(f"Bitti. En iyi val loss={best_val:.5f}  checkpoint'ler: {ckpt_dir}")


if __name__ == "__main__":
    main()
