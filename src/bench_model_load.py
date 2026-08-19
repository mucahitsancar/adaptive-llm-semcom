# -*- coding: utf-8 -*-
"""TEMIZ MODEL YUKLEME SURESI.

Neden gerekli: makalede "gecisin bir maliyeti var" diyoruz (sec:timescale).
O maliyetin buyuk kismi karar suresi degil, YENI MODELI YUKLEME suresi.
Elimizdeki tek sayi kirli olcumden geliyordu (INT8 icin 1800 sn gibi
inanilmaz bir deger) ve makaleye giremez.

Protokol:
  - her konfigurasyon icin ayri SUREC yok; tek surecte sirayla yuklenir ve
    aradan model silinip VRAM bosaltilir (gercek gecis senaryosu bu)
  - iki yukleme yapilir: ilki disk onbellegi soguk olabilir, ikincisi
    dagitimda gorulecek kararli hal. Ikisi de yazilir, yorum okuyucuya birakilmaz.
  - INT8 tek yukleme (kirli olcum cok yavas oldugunu gosterdi, iki tur
    kuyrugun kalanini yer)
  - GPU baska is yapiyorsa olcum kirlenir: basta bos oldugu dogrulanir
"""
import argparse
import csv
import gc
import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CIKTI = ROOT / "experiments" / "results" / "model_load_time.csv"

KONFIG = [
    ("qwen0.8b-fp16", "Qwen/Qwen3.5-0.8B", "none", 2),
    ("qwen2b-fp16", "Qwen/Qwen3.5-2B", "none", 2),
    ("qwen2b-int4", "Qwen/Qwen3.5-2B", "int4", 2),
    ("qwen4b-int4", "Qwen/Qwen3.5-4B", "int4", 2),
    ("qwen2b-int8", "Qwen/Qwen3.5-2B", "int8", 1),
]


def vram_mb():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return -1


def yukle(model_id, quant):
    """Modeli yukle, gecen sureyi dondur. Import icerde: olcum disi kalsin."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"dtype": torch.float16, "device_map": "cuda:0"}
    if quant in ("int4", "int8"):
        from transformers import BitsAndBytesConfig
        if quant == "int4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kw.pop("dtype")

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    torch.cuda.synchronize()
    sure = time.perf_counter() - t0

    tepe = torch.cuda.max_memory_allocated() / 1e9
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return sure, tepe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vram-esik-mb", type=int, default=800,
                    help="Bunun uzerinde kullanim varsa GPU mesgul sayilir")
    a = ap.parse_args()

    bos = vram_mb()
    print(f"baslangic VRAM: {bos} MB")
    if bos > a.vram_esik_mb:
        print(f"IPTAL: GPU mesgul ({bos} MB > {a.vram_esik_mb} MB). "
              "Baska is varken olculen yukleme suresi kirli olur.")
        return 1

    satirlar = []
    for ad, model_id, quant, tur in KONFIG:
        for i in range(tur):
            etiket = "soguk" if i == 0 else "kararli"
            try:
                sure, tepe = yukle(model_id, quant)
                print(f"{ad:16s} {etiket:8s} {sure:8.1f} sn  tepe {tepe:5.2f} GB",
                      flush=True)
                satirlar.append({"konfig": ad, "quant": quant, "tur": etiket,
                                 "yukleme_sn": round(sure, 1),
                                 "tepe_bellek_gb": round(tepe, 3)})
            except Exception as e:  # olcum surse de kuyruk durmasin
                print(f"{ad:16s} {etiket:8s} HATA: {type(e).__name__}: {e}",
                      flush=True)
                satirlar.append({"konfig": ad, "quant": quant, "tur": etiket,
                                 "yukleme_sn": "", "tepe_bellek_gb": "",
                                 "hata": f"{type(e).__name__}: {e}"})

    alanlar = sorted({k for s in satirlar for k in s})
    with open(CIKTI, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=alanlar)
        w.writeheader()
        w.writerows(satirlar)
    print(f"\n-> {CIKTI.name} ({len(satirlar)} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
