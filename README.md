# FedAlphaEdit: Null-Space-Aligned Merging for Collaborative Knowledge Editing

Code for the paper

> **FedAlphaEdit: Null-Space-Aligned Merging for Collaborative Knowledge Editing**
> Sota Sugawara, Yukihiko Okada (University of Tsukuba)
> arXiv:2610.11033 (https://arxiv.org/abs/2610.11033)

Contact: Sota Sugawara (s2620520@u.tsukuba.ac.jp)

This repository contains the minimal code needed to reproduce the proposed
method (null-space-aligned merging of per-client AlphaEdit updates) together
with the centralized and naive baselines and the merge ablations.

## Citation

```bibtex
@article{sugawara2026fedalphaedit,
  title   = {FedAlphaEdit: Null-Space-Aligned Merging for Collaborative Knowledge Editing},
  author  = {Sugawara, Sota and Okada, Yukihiko},
  journal = {arXiv preprint arXiv:2610.11033},
  year    = {2026}
}
```

## Contents

```
README.md                  this file
LICENSE                    MIT license for our code
THIRD_PARTY_NOTICES.md     upstream projects, their licenses, and our changes
licenses/                  full license texts of upstream projects
CollabEdit/                code base (a reduced copy of CollabEdit)
  experiments/g2_singleRound.py      proposed method + merge ablations
  experiments/g1_singleRound.py      centralized and naive baselines
  experiments/g2_unittest_oneshot.py Theorem 1 one-shot exactness test
  experiments/CollabEdit_summarize.py summary statistics
  memit/nsmerge.py                   null-space-aligned merging rule
  memit/alphaedit_collab.py          per-client AlphaEdit editor, P0, leak metrics
```

## 1. Environment

* Conda environment: `CollabEdit/scripts/collabEdit.yml` (Python 3.9,
  PyTorch 2.6.0+cu118). All experiments ran on a single 8 GB-class GPU in bf16.
* Recommended env vars: `PYTHONUTF8=1`,
  `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`,
  `G0_CUDA_MEM_CAP_GB=7.4` (VRAM cap), `G0_COMPUTE_Z_SPLIT=2`.
* Working directory for all commands: `CollabEdit/`. Paths in
  `CollabEdit/globals.yml` are relative to it (caches live in `../cache/`).

## 2. Data, models, and caches

No model weights or datasets are included; everything is fetched or built on
first use.

* **Datasets** (MultiCounterFact, zsRE): auto-downloaded from the public
  ROME/MEMIT data host (`REMOTE_ROOT_URL` in `globals.yml`).
* **Models**: `gpt2-xl` and `Qwen/Qwen2.5-1.5B` from the HuggingFace Hub.
* **Second-moment statistics**: computed from `wikipedia` (snapshot
  `20220301.en`, HuggingFace `datasets`) by `rome/layer_stats.py` and cached
  under `data/stats/`.
* **Null-space projection P0** (threshold 0.02): built from the statistics and
  cached under `../cache/P/` (next to the z-cache).
* **z-cache** (target vectors at the base model): computed per case on the
  first run with `--use_cache` and reused by every method. The z optimization
  uses randomly sampled context templates (unseeded, as in MEMIT/CollabEdit),
  so a freshly built cache differs from ours and numbers reproduce up to this
  run-to-run variation. Build the cache once and reuse it for all methods so
  that comparisons stay internally consistent.

## 3. Running the experiments

GPT-2 XL, MultiCounterFact, 10 clients x 500 edits, single round:

```bash
# Centralized AlphaEdit (all 5000 edits in one place; deterministic)
python -m experiments.g1_singleRound --merging_method GLOBAL \
    --num_clients 10 --num_edits_per_client_per_round 500 \
    --dataset_size_limit 5000 --use_cache --skip_glue

# Naive combination: per-client AlphaEdit + CollabEdit merge (seeds 1/2/3)
python -m experiments.g1_singleRound --merging_method CollabEdit \
    --num_clients 10 --num_edits_per_client_per_round 500 \
    --dataset_size_limit 5000 --use_cache --skip_glue --shard_seed 1

# Proposed: FedAlphaEdit (null-space-aligned merging; seeds 1/2/3)
python -m experiments.g2_singleRound --merging_method NS \
    --num_clients 10 --num_edits_per_client_per_round 500 \
    --dataset_size_limit 5000 --use_cache --shard_seed 1
```

Variants (same drivers):

* Merge ablations: merge-only (`g2_singleRound --merging_method S2`:
  per-client MEMIT + merge-time right projection) and simple average
  (`--merging_method S3`).
* Per-client MEMIT control: `g1_singleRound --merging_method CollabEdit --client_editor memit`.
* Qwen arm: add `--model_name Qwen/Qwen2.5-1.5B --hparams_fname qwen2.5-1.5b.json`
  (10 x 200 edits in the paper).
* zsRE arm: add `--ds_name zsre` and set `Z_EVAL_FP32_HEAD=1 Z_EVAL_CHUNK=8`
  (fp32 output head, chunked evaluation, as in the paper).

`--shard_seed` controls the IID assignment of cases to clients. Each run writes
to the next free `results/<G1|G2>/run_XXX`. Summarize a run with

```bash
python -m experiments.CollabEdit_summarize --dir_name G2 \
    --edit_category collaborative_edit --runs run_000
```

Use `--dir_name G1` for the baselines; the centralized (GLOBAL) run stores its
results under `--edit_category global_edit`, all other runs under
`collaborative_edit`.

### Theorem 1 unit test

```bash
python -m experiments.g2_unittest_oneshot
```

The script prints `OVERALL PASS` (exit code 0) when both checks hold:
Test A, for one client and one batch, null-space-aligned merging reproduces
centralized AlphaEdit to machine precision (`test_A_pass: true`); and the
negative control, a mis-specified regularizer (beta added per client), breaks
the identity (`negctrl_detects_bug: true`). Test B (two clients) is reported as
a diagnostic and does not enter the verdict: because AlphaEdit edits layers
sequentially, the keys of upper layers differ slightly between the per-client
and centralized runs, so Test B shows a small non-zero deviation by design.

## 4. Code-to-paper mapping

| Paper object | Implementation |
|---|---|
| P0 (null-space projection, threshold 0.02) | `memit/alphaedit_collab.py::load_or_build_projection` |
| Per-client AlphaEdit solve (operator P0 Ai P0 + beta I) | `memit/alphaedit_collab.py::execute_alphaedit_collab` |
| Null-space-aligned merging (Delta_proposed) | `memit/nsmerge.py::ns_aligned_merge_layer` |
| Merge-only (S2) / simple-average (S3) ablations | `memit/nsmerge.py::s2_rightproject_layer` / `s3_average_layer` |
| Naive covariance-weighted merge (Delta_naive) | CollabEdit branch of `experiments/g1_singleRound.py` |
| Centralized AlphaEdit | GLOBAL branch of `experiments/g1_singleRound.py` |
| oons and \|\|Delta K0\|\|_F leak measurements | `memit/alphaedit_collab.py::kl_leak` |
| Theorem 1 verification | `experiments/g2_unittest_oneshot.py` |
| ES/PS/NS evaluation (MCF) | `experiments/py/eval_utils_counterfact.py` |
| zsRE evaluation | `experiments/py/eval_utils_zsre.py` |

## 5. Not included

The following are not part of this minimal release:

* the general-ability (GLUE-style) harness, so `g1_singleRound.py` has to be
  run with `--skip_glue`;
* the FedLEKE reimplementation, including its unit tests and its
  retrieval-gate records;
* the scalar-weight baseline scripts (gamma * sum of client updates);
* the multi-round drivers, the diagnostic scripts, and the per-track analysis
  scripts.

Run outputs and caches are regenerated by the drivers.

## 6. License

Our code is released under the MIT License (`LICENSE`,
Copyright (c) 2026 Sota Sugawara, Yukihiko Okada). The `CollabEdit/` code base
is a reduced copy of the public CollabEdit repository (MIT, `CollabEdit/LICENSE`),
which builds on MEMIT/ROME (MIT); parts of the editor are ported from AlphaEdit
(MIT). See `THIRD_PARTY_NOTICES.md` and `licenses/`.
