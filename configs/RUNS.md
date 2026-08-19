# Run configurations and column glossary

Every result file in `experiments/results/` is named `<label>_eval.csv`, with the
sampled decodings in `<label>_samples.txt` and the full per-sentence decoder
output in `<label>_texts/snr<γ>_decoded.txt`. The sentence count `n` is recorded
in the `n` column of each CSV rather than repeated here, so it cannot go stale.

## Label grammar

A label is built from `<system>[-<variant>...]-<channel>`, with these parts:

| Part | Meaning |
|---|---|
| `deepsc` | DeepSC codec baseline (Xie et al., IEEE TSP 2021) |
| `qwen08b`, `qwen2b`, `qwen4b` | Qwen3.5 at 0.8B / 2B / 4B parameters |
| `vicuna7b` | Vicuna-7B v1.5 |
| `AWGN`, `Rayleigh` | channel; a trailing `R` in `M300-*` labels also means Rayleigh |
| `fp16`, `int8`, `int4` | decoder precision (absent = FP16) |
| `widesnr` | DeepSC trained with per-batch noise drawn U(0, 18) dB instead of a fixed level |
| `perH` | Rayleigh training with an independent fading realisation per sentence |
| `mine` | MINE mutual-information regularisation enabled |
| `beam4`, `beam1` | beam-width ablation (`beam1` = greedy); default is B = 10 |
| `nomask` | corpus vocabulary restriction disabled (`--no-restrict-vocab`) |
| `-fine` | extra SNR points (1, 3 dB) to locate the crossover on a finer grid |
| `smoke`, `preview` | pipeline checks, not measurements |

## Run families

### Short-sentence English (main results)

| Label | Configuration |
|---|---|
| `deepsc-AWGN`, `deepsc-Rayleigh` | DeepSC baselines, 80 epochs |
| `deepsc-AWGN-widesnr`, `deepsc-Rayleigh-widesnr` | wide-SNR training variant |
| `deepsc-Rayleigh-perH`, `deepsc-Rayleigh-widesnr-perH` | per-sentence fading; the second combines both variants |
| `deepsc-AWGN-mine`, `deepsc-Rayleigh-mine` | MINE-regularised variants |
| `deepsc-*-beam4` | beam-width ablation of the codec decoder |
| `qwen08b-AWGN`, `qwen08b-Rayleigh` | Qwen3.5-0.8B, FP16 |
| `qwen2b-{AWGN,Rayleigh}`, `qwen2b-int8-*`, `qwen2b-int4-*` | Qwen3.5-2B at three precisions |
| `qwen4b-fp16-*`, `qwen4b-int8-*`, `qwen4b-int4-*` | Qwen3.5-4B at three precisions (FP16/INT8 on Colab) |
| `vicuna7b-AWGN`, `vicuna7b-Rayleigh` | Vicuna-7B v1.5 (Colab) |
| `qwen08b-beam1-AWGN` | greedy decoding ablation |
| `qwen08b-nomask-AWGN` | vocabulary-restriction ablation |

### Matched sentence set (`M300-*`)

DeepSC re-evaluated on the same 300 sentences the LLM decoders use, so paired
comparisons rest on a common set: `M300-base300` (AWGN), `M300-base300R`
(Rayleigh), `M300-perH300R`, `M300-widesnr300`.

### Long-sentence regime (`LONG-*`)

The payload-length regime: `LONG-deepsc-base`, `LONG-deepsc-retrained`,
`LONG-deepsc-widesnr`, `LONG-qwen08b`, plus `-fine` companions carrying the extra
1 dB and 3 dB points. `retrained` means DeepSC retrained on the long-sentence
distribution rather than applied out-of-distribution.

### Turkish and its English control (secondary contribution)

| Label | Configuration |
|---|---|
| `TR-deepsc` | DeepSC on Turkish |
| `TR-deepsc150` | the same 150 sentences the LLM runs use, so the two are comparable |
| `TR-qwen08b`, `TR-qwen2b` | Qwen3.5 decoders on Turkish |
| `ENpar-qwen08b`, `ENpar-qwen2b` | the **English side of the same parallel sentences**, isolating the language effect from content |

Turkish data comes from the OPUS TED2020 en-tr parallel corpus, so the English
control and the Turkish run carry identical content.

## Analysis files

| File | Contents |
|---|---|
| `latency_bench.csv` | per-configuration latency, VRAM, channel/LLM step split |
| `model_load_time.csv` | cold/warm load time and peak memory |
| `policy_actions.csv` | the action space, with exclusions and their reasons |
| `policy_table.csv` | fitted policy: selection and utility per (λ, channel, SNR) |
| `policy_validation.csv` | held-out evaluation with per-point regret |
| `policy_comparison.csv`, `policy_cmp_<λ>.csv` | the rule against fixed and oracle alternatives |
| `msps_all.csv` | MSPS and per-category preservation for the Turkish runs |
| `similarity_all.csv` | sentence-similarity scores |
| `bootstrap_ci.csv` | paired bootstrap confidence intervals per claim |
| `*_train.csv` | DeepSC training curves |

## Column glossary (Turkish → English)

**`policy_table.csv`** — `lam_d` λ_D (accuracy/latency trade-off), `lam_m` λ_M
(memory weight), `kanal` channel, `secim` selected action, `fayda` utility,
`gecikme_sn_cumle` latency s/sentence, `gecikme_sn_token` latency s/token,
`bellek_gb` memory GB.

**`policy_validation.csv`** — `politika_secimi` policy's choice,
`politika_fayda` its utility, `en_iyi_fayda_secimi` / `en_iyi_fayda` the best
achievable choice and utility, `regret` the gap, `en_dogru_secim` /
`en_yuksek_bleu1` the most accurate choice and its BLEU-1, `dogruluk_farki`
accuracy gap.

**`policy_comparison.csv`** — `kural` is the decision rule being compared:
`BIZIM-POLITIKA-v1` / `-v2` our fitted policy, `SABIT-EN-IYI` fixed
best-on-average, `ESIK-SEZGISI` an SNR-threshold heuristic, `KAHIN` the
utility oracle, `KAHIN-DOGRULUK` the accuracy oracle. Other columns as above.

**`policy_actions.csv`** — `aksiyon` action, `durum` status (`dahil` included,
`dislandi` excluded), `gerekce` reason for exclusion.

**`latency_bench.csv`** — `median_s` median seconds/sentence, `ort_token` mean
tokens/sentence, `kanal_adim_ms` channel step ms, `llm_adim_ms` LLM step ms.

**`model_load_time.csv`** — `konfig` configuration, `tepe_bellek_gb` peak memory
GB, `tur` run type (`soguk` cold, `sicak` warm), `yukleme_sn` load seconds.

**`msps_all.csv`** — `msps` the score, `kritik_tersinme_orani` critical inversion
rate over {negation, potentiality, tense}, `analiz_edilemeyen_oran` share of
tokens the morphological analyser could not parse, `ozellik_sayisi` feature
count, and the seven critical categories: `kat_olumsuzluk` negation,
`kat_zaman` tense, `kat_kisi` person, `kat_soru` question, `kat_hal` case,
`kat_yeterlik` potentiality, `kat_kip` mood.

**`bootstrap_ci.csv`** — `iddia` the claim under test, `A`/`B` the two runs
compared, `beklenen` the expected direction, `fark` the paired difference,
`ci_alt`/`ci_ust` CI bounds, `p` the p-value, `eslestirme` the pairing basis,
`sonuc` verdict (`ayirt_edilemez` = indistinguishable).

**`*_train.csv`** — `epoch_sure_sn` epoch duration in seconds.

**`*_samples.txt`** — `HEDEF` target sentence, `ÇÖZÜM` decoded sentence.
