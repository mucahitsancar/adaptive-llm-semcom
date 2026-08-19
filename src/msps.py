# -*- coding: utf-8 -*-
"""
MSPS — Morfolojik Anlam Koruma Skoru (ikincil katkı, hoca onaylı).

Sorun: BLEU kelime örtüşmesi sayar; embedding benzerliğinin ise ölçtüğümüz şans
tabanı 0.70 (bkz. NOTLAR §6.6.1). İkisi de "gelmeyeceğim → geleceğim" gibi
ANLAMI TERSİNE ÇEVİREN küçük bir ek kaybını yakalayamıyor. MSPS bunu doğrudan ölçer.

Tanım (makale §IV.D):
  Φ(s) : cümledeki anlam-taşıyıcı morfolojik özellikler kümesi (kelime konumuyla
         birlikte), yalnız kritik kategoriler:
           Neg (olumsuzluk), Zaman (Past/Fut/Prog/Aor/Narr), Kişi (A1sg..A3pl),
           Soru (Ques), Hâl (Loc/Abl/Dat/Acc/Gen), Çoğul (A3pl), Yeterlik (Able)
  MSPS(s,ŝ) = |Φ(s) ∩ Φ(ŝ)| / |Φ(s)|        (kategori bazında da raporlanır)

Uygulama notları:
  - Analizör: zeyrek (Zemberek'in Python portu). Belirsizlikte ilk çözümleme alınır;
    analiz edilemeyen kelime, o kelimenin özellikleri boş sayılır (dürüstlük: bu
    oran raporlanır — `analiz_edilemeyen_oran`).
  - Eşleştirme çoklu-küme (multiset) üzerinden yapılır, konum duyarlı DEĞİLDİR:
    çözülen cümlede kelime sırası kayabilir.
  - Özellikler LEMMA'dan bağımsızdır (varsayılan). Gerekçe: sözcük seçimini BLEU
    zaten ölçüyor; MSPS'in işi dilbilgisel anlam taşıyıcılarının korunumu. Örnek:
    "katılamayacağım → gelemeyeceğim" sözcük değişse de olumsuzluk/yeterlik/zaman/
    kişi korunduğu için yüksek MSPS alır; "katılamayacağım → katılacağım" ise
    yeterlik/olumsuzluk kaybı nedeniyle düşer. Lemma duyarlı katı varyant
    `--lemma-sensitive` ile ölçülebilir (ablasyon için).
  - Ek olarak KRİTİK TERSİNME ORANI raporlanır: en az bir olumsuzluk/yeterlik/zaman
    özelliğini kaybeden cümlelerin payı — yorumlanması en kolay sayı.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"

# kritik kategoriler: morfem etiketi → (kategori adı)
CRITICAL = {
    "Neg": "olumsuzluk",
    "Past": "zaman", "Fut": "zaman", "Prog1": "zaman", "Prog2": "zaman",
    "Aor": "zaman", "Narr": "zaman", "Pres": "zaman",
    "A1sg": "kisi", "A2sg": "kisi", "A3sg": "kisi",
    "A1pl": "kisi", "A2pl": "kisi", "A3pl": "kisi",
    "Ques": "soru",
    "Loc": "hal", "Abl": "hal", "Dat": "hal", "Acc": "hal", "Gen": "hal", "Ins": "hal",
    "Able": "yeterlik", "Unable": "yeterlik",
    "Cond": "kip", "Necess": "kip", "Opt": "kip", "Imp": "kip",
}

parser = argparse.ArgumentParser()
parser.add_argument("--target", required=True, help="hedef cümleler dosyası")
parser.add_argument("--decoded", required=True, help="çözülen cümleler dosyası")
parser.add_argument("--label", default="", help="çıktı satırı etiketi")
parser.add_argument("--out", default=str(RES / "msps_all.csv"))
parser.add_argument("--lemma-sensitive", action="store_true", help="katı varyant (ablasyon)")


CRITICAL_FOR_INVERSION = {"olumsuzluk", "yeterlik", "zaman"}


class MSPS:
    def __init__(self, lemma_sensitive=False):
        import zeyrek
        self.an = zeyrek.MorphAnalyzer()
        self.cache = {}
        self.n_words = 0
        self.n_unparsed = 0
        self.lemma_sensitive = lemma_sensitive

    def features(self, sentence):
        """Kritik morfolojik özellik çoklu-kümesi.
        Varsayılan anahtar: (kategori, morfem) — lemmadan bağımsız."""
        feats = Counter()
        for w in sentence.split():
            w = w.strip(".,;:!?()\"'").lower()
            if not w:
                continue
            self.n_words += 1
            if w in self.cache:
                parsed = self.cache[w]
            else:
                try:
                    r = self.an.analyze(w)
                    parsed = (r[0][0].lemma, list(r[0][0].morphemes)) if r and r[0] else None
                except Exception:
                    parsed = None
                self.cache[w] = parsed
            if parsed is None:
                self.n_unparsed += 1
                continue
            lemma, morphemes = parsed
            for m in morphemes:
                if m in CRITICAL:
                    key = ((lemma, CRITICAL[m], m) if self.lemma_sensitive
                           else (CRITICAL[m], m))
                    feats[key] += 1
        return feats

    def score(self, targets, decodeds):
        """Genel MSPS + kategori oranları + kritik tersinme oranı."""
        tot_t = tot_hit = 0
        per_cat = {}
        inverted = n_eval = 0
        for t, d in zip(targets, decodeds):
            ft, fd = self.features(t), self.features(d)
            hit = sum(min(c, fd[k]) for k, c in ft.items())
            n = sum(ft.values())
            tot_t += n
            tot_hit += hit
            crit_lost = False
            for key, c in ft.items():
                cat = key[1] if self.lemma_sensitive else key[0]
                a, b = per_cat.get(cat, (0, 0))
                got = min(c, fd[key])
                per_cat[cat] = (a + c, b + got)
                if cat in CRITICAL_FOR_INVERSION and got < c:
                    crit_lost = True
            def _cat(k):
                return k[1] if self.lemma_sensitive else k[0]
            if any(_cat(k) in CRITICAL_FOR_INVERSION for k in ft):
                n_eval += 1
                inverted += int(crit_lost)
        return {
            "msps": round(tot_hit / tot_t, 4) if tot_t else None,
            "kategoriler": {k: round(v[1] / v[0], 4) for k, v in sorted(per_cat.items()) if v[0]},
            "kritik_tersinme_orani": round(inverted / n_eval, 4) if n_eval else None,
            "ozellik_sayisi": tot_t,
            "analiz_edilemeyen_oran": round(self.n_unparsed / max(self.n_words, 1), 4),
        }


def main():
    args = parser.parse_args()
    tgt = Path(args.target).read_text(encoding="utf-8").strip().splitlines()
    hyp = Path(args.decoded).read_text(encoding="utf-8").strip().splitlines()
    n = min(len(tgt), len(hyp))
    m = MSPS(lemma_sensitive=args.lemma_sensitive)
    r = m.score(tgt[:n], hyp[:n])
    print(f"{args.label or Path(args.decoded).name}: MSPS={r['msps']} "
          f"(özellik={r['ozellik_sayisi']}, analiz edilemeyen kelime "
          f"%{100*r['analiz_edilemeyen_oran']:.1f})")
    for k, v in r["kategoriler"].items():
        print(f"    {k:12s} {v}")

    out = Path(args.out)
    rows = list(csv.DictReader(open(out, encoding="utf-8"))) if out.exists() else []
    row = {"label": args.label, "msps": r["msps"], "n": n,
           "ozellik_sayisi": r["ozellik_sayisi"],
           "analiz_edilemeyen_oran": r["analiz_edilemeyen_oran"],
           "kritik_tersinme_orani": r["kritik_tersinme_orani"],
           **{f"kat_{k}": v for k, v in r["kategoriler"].items()}}
    rows = [x for x in rows if x.get("label") != args.label] + [row]
    keys = sorted({k for x in rows for k in x})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
