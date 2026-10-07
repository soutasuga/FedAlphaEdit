# Third-party notices

This repository builds on the following MIT-licensed projects. Their full
license texts are reproduced in `licenses/` (CollabEdit's is kept unmodified at
`CollabEdit/LICENSE`).

| Project | License | Copyright | License text |
|---|---|---|---|
| CollabEdit (github.com/LINs-lab/CollabEdit) | MIT | (c) 2024 Zheng Jiamu | `CollabEdit/LICENSE` |
| MEMIT (github.com/kmeng01/memit) | MIT | (c) 2022 Kevin Meng | `licenses/MEMIT_LICENSE.txt` |
| ROME (github.com/kmeng01/rome) | MIT | (c) 2022 Kevin Meng | `licenses/ROME_LICENSE.txt` |
| AlphaEdit (github.com/jianghoucheng/AlphaEdit) | MIT | (c) 2025 jianghc | `licenses/AlphaEdit_LICENSE.txt` |
| PitfallsKnowledgeEditing (github.com/zjunlp/PitfallsKnowledgeEditing), acknowledged by CollabEdit | MIT | (c) 2023 ZJUNLP | `licenses/PitfallsKnowledgeEditing_LICENSE.txt` |

## Provenance of files

This is a reduced copy of CollabEdit that keeps only the modules needed by the
FedAlphaEdit drivers; unused upstream experiments, hyper-parameter files and
scripts were removed.

**From CollabEdit (which itself builds on MEMIT/ROME), unmodified in content**:
every file under `CollabEdit/` that is not listed below.

**From CollabEdit, modified by us** (content changes relative to
LINs-lab/CollabEdit `main`):

- `CollabEdit/dsets/tfidf_stats.py`
- `CollabEdit/experiments/py/eval_utils_zsre.py`
- `CollabEdit/globals.yml` (cache path)
- `CollabEdit/memit/compute_z.py`
- `CollabEdit/memit/memit_main.py`
- `CollabEdit/rome/layer_stats.py`

**Ported from AlphaEdit** (MIT, (c) 2025 jianghc):

- `CollabEdit/memit/alphaedit_hparams.py` (verbatim copy of AlphaEdit's hyper-parameter class)
- `CollabEdit/hparams/AlphaEdit/gpt2-xl.json` (AlphaEdit recipe; our
  `qwen2.5-1.5b.json` is derived from AlphaEdit's Llama3-8B recipe)

**Written by us** (MIT, (c) 2026 Sota Sugawara, Yukihiko Okada; each file
carries an SPDX header): `experiments/g1_singleRound.py`,
`experiments/g2_singleRound.py`, `experiments/g2_unittest_oneshot.py`,
`memit/alphaedit_collab.py` (implements AlphaEdit's editing equation inside the
CollabEdit pipeline), `memit/nsmerge.py`, `util/g0_mem.py`.
