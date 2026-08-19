# -*- coding: utf-8 -*-
"""
Tüm *_eval.csv dosyalarında ÜZERİNE YAZMA kaynaklı veri kaybı taraması.

Her dosyanın şu andaki SNR kümesini, git geçmişindeki TÜM sürümlerinin SNR
kümeleriyle karşılaştırır. Geçmişte olup şimdi olmayan bir SNR varsa nokta
kaybedilmiş demektir (bkz. LONG-qwen08b olayı, LOG 2026-08-18).
"""
import csv
import io
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"


def git(*args):
    r = subprocess.run(["git"] + list(args), cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def snr_set(text):
    if not text.strip():
        return set()
    out = set()
    for r in csv.DictReader(io.StringIO(text)):
        try:
            out.add(int(float(r["snr"])))
        except (ValueError, KeyError, TypeError):
            pass
    return out


kayip_var = False
for p in sorted(RES.glob("*_eval.csv")):
    rel = f"experiments/results/{p.name}"
    simdi = snr_set(p.read_text(encoding="utf-8"))
    commits = [c for c in git("log", "--format=%h", "--all", "--", rel).split() if c]
    gecmis = set()
    for c in commits:
        gecmis |= snr_set(git("show", f"{c}:{rel}"))
    kayip = gecmis - simdi
    if kayip:
        kayip_var = True
        print(f"KAYIP  {p.name}: simdi {sorted(simdi)} | gecmiste ayrica {sorted(kayip)}")

if not kayip_var:
    print("Tarama tamam: git gecmisinde olup su an eksik olan SNR noktasi YOK.")
print(f"({len(list(RES.glob('*_eval.csv')))} eval dosyasi tarandi)")
