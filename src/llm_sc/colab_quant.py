# -*- coding: utf-8 -*-
"""Colab (T4) için dayanıklı nicemleme koşusu sürücüsü.

Neden ayrı script: Colab oturumu kopunca /content siliniyor. Bu sürücü
eval_llm_sc.py'yi SNR noktası başına AYRI süreçte çağırır ve her nokta biter
bitmez sonuçları Drive'a kopyalar. Oturum koparsa aynı komut tekrar çalıştırılır,
bitmiş noktaları atlar (texts/ klasöründeki satır sayısına bakarak).

Kural (§2.2): Colab T4 gecikme/VRAM sayıları YEREL GPU tablosuna karıştırılmaz.
Bu yüzden her koşuda <prefix>_HARDWARE.txt yazılır (GPU adı, torch, tarih).

Örnek:
  python src/llm_sc/colab_quant.py --model Qwen/Qwen3.5-4B --quant int8 \
      --channel AWGN --snr-list 12,15,18 --out-prefix qwen4b-int8-AWGN \
      --limit 150 --backup-dir /content/drive/MyDrive/tez_sonuclar
"""
import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--quant", default="int8", choices=["none", "int8", "int4"])
p.add_argument("--channel", default="AWGN", choices=["AWGN", "Rayleigh"])
p.add_argument("--snr-list", default="0,6,12,18")
p.add_argument("--out-prefix", required=True)
p.add_argument("--limit", type=int, default=150)
p.add_argument("--beam", type=int, default=10)
p.add_argument("--sentences", default=None, help="test_sentences.txt yolu (varsayılan: repo içi)")
p.add_argument("--backup-dir", default=None, help="Her nokta sonrası kopyalanacak Drive klasörü")
p.add_argument("--force", action="store_true", help="Bitmiş noktaları da yeniden koş")


def done_snrs(prefix, snrs, limit):
    """Bitmiş SNR noktaları: hem hedef hem çözüm metni var ve >= limit satır."""
    d = RES / f"{prefix}_texts"
    ok = []
    for s in snrs:
        dec, tgt = d / f"snr{s}_decoded.txt", d / f"snr{s}_target.txt"
        if dec.exists() and tgt.exists():
            n = len(dec.read_text(encoding="utf-8").strip().splitlines())
            if n >= limit:
                ok.append(s)
    return ok


def read_samples(prefix):
    f = RES / f"{prefix}_samples.txt"
    return f.read_text(encoding="utf-8") if f.exists() else ""


def merge_samples(prefix, snr, prev):
    """eval_llm_sc.py samples dosyasını 'w' ile açıyor → nokta başına koşarken
    eski örnekler siliniyordu. Eski bölümleri (bu SNR hariç) geri ekle."""
    yeni = read_samples(prefix)
    tutulan = []
    for blok in prev.split("\n===== SNR "):
        blok = blok.strip("\n")
        if not blok:
            continue
        if blok.startswith(f"{snr} dB ====="):     # bu nokta yeniden ölçüldü
            continue
        tutulan.append("\n===== SNR " + blok if "dB =====" in blok.splitlines()[0] else blok)
    (RES / f"{prefix}_samples.txt").write_text(
        ("\n".join(tutulan) + "\n" if tutulan else "") + yeni, encoding="utf-8")


def hardware_note(prefix, args):
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        tv, cu = torch.__version__, torch.version.cuda
    except Exception:
        gpu, tv, cu = "?", "?", "?"
    (RES / f"{prefix}_HARDWARE.txt").write_text(
        f"donanim: {gpu}\ntorch: {tv} (cuda {cu})\ntarih: {datetime.now():%Y-%m-%d %H:%M}\n"
        f"model: {args.model} quant={args.quant} kanal={args.channel} "
        f"beam={args.beam} n={args.limit}\n"
        "NOT: Bu koşunun gecikme/VRAM sayilari YEREL GPU tablosuna karistirilmaz "
        "(NOTLAR §2.2); ayri raporlanir.\n", encoding="utf-8")


def backup(prefix, dest):
    if not dest:
        return
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for ad in (f"{prefix}_eval.csv", f"{prefix}_samples.txt", f"{prefix}_HARDWARE.txt"):
        src = RES / ad
        if src.exists():
            shutil.copy2(src, dest / ad)
    td = RES / f"{prefix}_texts"
    if td.exists():
        shutil.copytree(td, dest / f"{prefix}_texts", dirs_exist_ok=True)
    print(f"  -> Drive yedegi guncellendi: {dest}", flush=True)


def main():
    args = p.parse_args()
    snrs = [int(s) for s in args.snr_list.split(",")]
    prefix = args.out_prefix
    hardware_note(prefix, args)

    done = [] if args.force else done_snrs(prefix, snrs, args.limit)
    missing = [s for s in snrs if s not in done]
    print(f"{prefix}: bitmis={done} kosulacak={missing}", flush=True)
    if not missing:
        print("Yapilacak nokta yok."); backup(prefix, args.backup_dir); return

    for i, snr in enumerate(missing, 1):
        prev = read_samples(prefix)
        cmd = [sys.executable, str(ROOT / "src" / "llm_sc" / "eval_llm_sc.py"),
               "--model", args.model, "--quant", args.quant, "--channel", args.channel,
               "--limit", str(args.limit), "--beam", str(args.beam),
               "--snr-list", str(snr), "--out-prefix", prefix]
        if args.sentences:
            cmd += ["--sentences", args.sentences]
        print(f"\n=== [{i}/{len(missing)}] {args.channel} {snr} dB basliyor ===", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd)
        if r.returncode != 0:
            backup(prefix, args.backup_dir)
            sys.exit(f"{snr} dB hata verdi ({r.returncode}) — yedek alindi, komutu tekrar calistir.")
        merge_samples(prefix, snr, prev)
        print(f"=== {snr} dB bitti ({(time.time()-t0)/60:.1f} dk) ===", flush=True)
        backup(prefix, args.backup_dir)

    kalan = [s for s in snrs if s not in done_snrs(prefix, snrs, args.limit)]
    print(f"\n=== {prefix} TAMAM (eksik: {kalan or 'yok'}) ===")


if __name__ == "__main__":
    main()
