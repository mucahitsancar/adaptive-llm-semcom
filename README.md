# Resource- and Channel-Adaptive Large Language Model Selection for Text Semantic Communication

Artifacts for the paper of the same title, by **Mücahit Sancar** and
**A. F. M. Shahen Shah** (Department of Electronics and Communications
Engineering, Yıldız Technical University, İstanbul).

This repository holds the measurement code, the run configurations and the
**per-sentence decoder outputs** behind every number reported in the paper. Each
table cell in the paper traces to a CSV file here; `configs/RUNS.md` gives the
run-label → configuration mapping and a glossary for the Turkish column names.

---

## Layout

```
src/                measurement and analysis pipeline
  llm_sc/           LLM-SC decoder: MAP beam fusion, channel physics, benchmarks
  train_deepsc.py   DeepSC baseline training (imports the official implementation)
  evaluate_deepsc.py
  derive_policy.py      fits the adaptive selection rule
  compare_policies.py   evaluates it against fixed / oracle alternatives
  msps.py               Morphological Semantic Preservation Score (Turkish)
  bootstrap_ci.py       paired bootstrap confidence intervals
  bench_*.py            latency, VRAM and model-load cost
  preprocess_europarl.py  Europarl download and preprocessing
  build_*.py            long-sentence, Turkish and matched test sets
  make_fig*.py          figures
experiments/
  results/          all eval CSVs, sampled decodings, per-sentence texts
  msps_sanity/      fixed-input sanity test for the MSPS metric
configs/
  RUNS.md           run-label → configuration map + CSV column glossary
  UPSTREAM_SOURCES.md   upstream repositories the runners import
figures/            generated figures
```

## Evaluation protocol

Identical across every system compared:

| Item | Value |
|---|---|
| Test sentences | held-out Europarl English, 4–30 words (DeepSC training regime) |
| SNR grid | {0, 3, 6, 9, 12, 15, 18} dB |
| Beam width | B = 10 |
| Channel | AWGN (h = 1) and Rayleigh (per-symbol h ~ CN(0,1), perfect CSI) |
| Noise power | N₀ = P_x · 10^(−γ/10), with P_x measured from the transmitted symbols |
| Symbol budget | codec 8 symbols/token; LLM S = 6 (smallest budget carrying a token index at 3 bits/symbol) |
| DeepSC training | 80 epochs; LLM decoders receive no training |

## Models

| Configuration | Identifier |
|---|---|
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` |
| Qwen3.5-2B (FP16 / INT8 / INT4) | `Qwen/Qwen3.5-2B` |
| Qwen3.5-4B (FP16 / INT8 / INT4) | `Qwen/Qwen3.5-4B` |
| Vicuna-7B v1.5 | `lmsys/vicuna-7b-v1.5` |
| DeepSC | trained here; see `src/train_deepsc.py` |

Quantisation uses `bitsandbytes` and requires that package plus a CUDA device.

## Hardware provenance — read before comparing costs

Fidelity is deterministic given the sentence set and the channel realisation, so
it is **device independent**. Latency is not.

- **All cost figures** (latency, VRAM, load time) come from one **RTX 4050 Laptop
  (6 GB)**, single process, otherwise idle machine.
- **Three configurations exceed that memory** and were run on Colab
  (Vicuna-7B, Qwen3.5-4B FP16, Qwen3.5-4B INT8). They carry **fidelity but no
  comparable cost**, and the policy excludes them.
- Timings are **never** compared across devices.

## Policy action space

The table below is the full picture: `policy_actions.csv` holds the candidate set
and the one candidate that was dropped, and the paper states one further
exclusion that never became a candidate. Both are listed here.

| Action | Status |
|---|---|
| DeepSC, DeepSC-improved | included |
| Qwen-0.8B FP16 | included |
| Qwen-2B FP16 / INT8 / INT4 | included |
| Qwen-4B INT4 | included |
| Qwen-4B FP16 | **excluded** — no comparable latency/memory measurement (run on Colab), so its penalty cannot be treated as zero |
| Qwen-4B INT8 | **excluded** — 6.31 GB peak memory exceeds the 6.0 GB budget, so it fails the memory constraint before the action space is formed; fidelity was measured on Colab, cost was not |

The two exclusions differ in kind. Qwen-4B FP16 *was* a candidate and was removed
because its cost is unmeasured: an action whose cost was never measured must not be
assigned a free one. Qwen-4B INT8 never entered the candidate set at all, because
6.31 GB violates the memory constraint of the optimisation directly — which is why
`policy_actions.csv`, whose scope is the candidate set, lists only the first.
Both are stated as excluded in the paper.

### Which λ values the paper reports

`derive_policy.py` sweeps λ ∈ {0, 0.01, 0.03, 0.08, 0.20}. The paper reports
λ ∈ {0.01, 0.03, 0.08}, the range where the decision is actually contested. The
two endpoints are degenerate and carry no information about the rule: at λ = 0
cost is free, and at λ = 0.20 the policy selects a codec at **every** operating
point (5 × DeepSC, 3 × DeepSC-improved, no LLM anywhere), so every rule that can
pick a codec ties. Both endpoints are present in `policy_table.csv` if you want
to check this.

## Reproducing

```bash
conda create -n semcom python=3.10 -y
conda activate semcom
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "numpy<2" nltk w3lib tqdm scikit-learn transformers bitsandbytes zeyrek

# 1) data: Europarl v7 -> data/raw -> data/processed
python src/preprocess_europarl.py

# 2) DeepSC baseline
python src/train_deepsc.py --channel AWGN
python src/evaluate_deepsc.py --channel AWGN --out-prefix deepsc-AWGN

# 3) LLM-SC decoder
python src/llm_sc/eval_llm_sc.py \
    --model Qwen/Qwen3.5-2B --channel AWGN --quant none \
    --snr-list 0,3,6,9,12,15,18 --beam 10 --limit 300 \
    --out-prefix qwen2b-AWGN

# 4) cost, then the policy
python src/llm_sc/bench_latency.py     # LLM decoders
python src/bench_deepsc_latency.py      # codec baseline
python src/bench_model_load.py
python src/derive_policy.py --sweep 0.00,0.01,0.03,0.08,0.20 --tie-lambdas --mem-max 6.0
python src/compare_policies.py
```

`--restrict-vocab` (on by default) limits the decoder to tokens that occur in the
corpus; `qwen08b-nomask-AWGN` is the ablation with it disabled.

## Not included

| Excluded | Why | How to obtain |
|---|---|---|
| `data/` | Europarl archive + preprocessed pickles | `src/preprocess_europarl.py` |
| `checkpoints/` | ~46 GB of weights | retrain, or download the HF models |
| `experiments/results/trainlogs/` | 561 MB of raw stdout | the same information is in `*_train.csv` |
| `third_party/` | upstream clones, kept unmodified | `configs/UPSTREAM_SOURCES.md` |

## Superseded and diagnostic artifacts, kept on purpose

These are **not** used in the paper. They are published because discarding a
measurement silently is worse than keeping it labelled:

- `model_load_time_KIRLI.csv` — *kirli* = contaminated. The host machine slept
  mid-run; wall-clock kept counting, so the load times are inflated. Superseded
  by `model_load_time.csv`.
- `deepsc-Rayleigh-mine_train_deneme1-nan.csv` — a training attempt that diverged
  to NaN. It is the record that motivated the numerical guards documented in
  `src/train_deepsc.py` (`--stable-mi`).
- `*smoke*`, `awgn-preview` — short pipeline checks, not measurements.

Real decoder outputs are published verbatim, including their errors: some
decodings contain garbled or hallucinated words, which is the phenomenon under
study.

## Language note

The thesis language is Turkish, so code comments, some run labels and several CSV
column names are Turkish. `configs/RUNS.md` translates every column name and run
label used in the paper.

## Citation

```bibtex
@inproceedings{sancar2026adaptive,
  author    = {Sancar, M{\"u}cahit and Shah, A. F. M. Shahen},
  title     = {Resource- and Channel-Adaptive Large Language Model Selection
               for Text Semantic Communication},
  booktitle = {Submitted to IEEE GLOBECOM Workshops},
  year      = {2026},
  note      = {Under review; not yet accepted}
}
```
