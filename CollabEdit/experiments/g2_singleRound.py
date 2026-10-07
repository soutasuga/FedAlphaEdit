# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sota Sugawara, Yukihiko Okada
# Builds on CollabEdit / MEMIT / AlphaEdit (MIT); see THIRD_PARTY_NOTICES.md.
"""Single-round driver for the proposed method (FedAlphaEdit) and the merge ablations.

merging_method:
  NS -- per-client AlphaEdit + null-space-aligned merging (proposed)
  S2 -- per-client MEMIT (no null-space) + CollabEdit merge, then right-
        project the merged update by P0 ("merge-only" ablation). oons=0
        by construction; discriminant is NS/KL.
  S3 -- per-client AlphaEdit + simple average (1/N) sum Delta_i.

Reuses the editor in memit.alphaedit_collab and the standard config. Each merge
is exact (nsmerge.py); no clipping/rescaling. Per-layer merged oons/KL logged
to g2_kl.jsonl. The centralized and naive baselines are run by
g1_singleRound.py. Anything else fails loud.
"""
import json
import os
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import (
    AttributeSnippets, CounterFactDataset, MENDQADataset,
    MultiCounterFactDataset, get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import upd_matrix_match_shape
from memit.memit_main import execute_memit
from memit.memit_hparams import MEMITHyperParams
from memit.alphaedit_collab import execute_alphaedit_collab, kl_leak, load_or_build_projection
from memit.alphaedit_hparams import AlphaEditHyperParams
from memit.nsmerge import ns_aligned_merge_layer, s2_rightproject_layer, s3_average_layer
from util import nethook
from util.globals import *
from util.g0_mem import G0MemTracker
from copy import deepcopy

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def shard(n, num_clients, per_client, seed):
    idx = list(range(n))
    if seed is not None:
        idx = np.random.RandomState(seed).permutation(n).tolist()
    return [idx[per_client * i: per_client * (i + 1)] for i in range(num_clients)]


def main(merging_method, model_name, ds_name, dataset_size_limit,
         skip_generation_tests, generation_test_interval, device,
         num_edits_pC_pR=20, num_clients=3, use_cache=True,
         shard_seed=None, hparams_fname="gpt2-xl.json"):
    if merging_method not in ("NS", "S2", "S3"):
        raise ValueError(f"[G2] merging_method={merging_method!r} not supported "
                         "(NS/S2/S3 only; GLOBAL/CollabEdit are run by g1_singleRound)")
    _cap = os.environ.get("G0_CUDA_MEM_CAP_GB")
    if _cap:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(float(_cap) * 2**30 / tot, 0)
        print(f"[G0] cap {_cap}GB", flush=True)

    num_edits = num_edits_pC_pR * num_clients
    is_alpha = merging_method in ("NS", "S3")

    alg_dir = RESULTS_DIR / "G2"
    run_id = 0
    if alg_dir.exists():
        ids = [int(str(x).split("_")[-1]) for x in alg_dir.iterdir()
               if str(x).split("_")[-1].isnumeric()]
        run_id = max(ids) + 1 if ids else 0
    run_dir = RESULTS_DIR / "G2" / f"run_{str(run_id).zfill(3)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results at {run_dir} (method={merging_method})")
    g0mem = G0MemTracker(run_dir)

    # AlphaEdit hparams drive P + z; MEMIT hparams (S2 clients) share the same
    # editing constants (layers, lambda, clamp) -- MEMIT lacks null-space keys.
    a_hp = AlphaEditHyperParams.from_json(HPARAMS_DIR / "AlphaEdit" / hparams_fname)
    # MEMIT hparams drive the S2 MEMIT clients only (is_alpha=NS/S3 never use m_hp).
    # S2 is a GPT2-XL ablation, so this stays gpt2-xl.json; dead load under NS/S3.
    m_hp = MEMITHyperParams.from_json(HPARAMS_DIR / "MEMIT" / "gpt2-xl.json")
    shutil.copyfile(HPARAMS_DIR / "AlphaEdit" / hparams_fname, run_dir / "params.json")
    L2 = a_hp.L2

    kl_path = run_dir / "g2_kl.jsonl"
    def kl_log(**kw):
        with open(kl_path, "a") as f:
            f.write(json.dumps(kw) + "\n")

    g0mem.phase_start("model_load")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    g0mem.phase_end("model_load")

    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None
    ds_class, ds_eval = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    cache_template = (KV_DIR / f"{model_name.replace('/', '_')}_MEMIT"
                      / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz") if use_cache else None

    g0mem.phase_start("projection")
    P = load_or_build_projection(model, tok, a_hp, device)
    g0mem.phase_end("projection")
    layers = a_hp.layers
    wnames = [f"{a_hp.rewrite_module_tmp.format(l)}.weight" for l in layers]
    layer_of = {w: l for w, l in zip(wnames, layers)}

    record_chunks = list(islice(iter(ds), num_edits))
    case_ids = [r["case_id"] for r in record_chunks]
    alloc = shard(num_edits, num_clients, num_edits_pC_pR, shard_seed)

    # Per-client edits (base model each time; restore=True keeps clients independent)
    per_client = []   # list over clients: dict w -> (Delta_paper d0-form used by merge)
    covs_ref = None
    all_kkts = []
    for i in range(num_clients):
        g0mem.phase_start("client_edit")
        recs = [record_chunks[j] for j in alloc[i]]
        reqs = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in recs]
        if is_alpha:
            cache_c = torch.zeros((len(layers), P.shape[1], P.shape[1]))
            deltas, covs, kkts, ks = execute_alphaedit_collab(
                model, tok, reqs, a_hp, device, cache_template=cache_template,
                cache_c=cache_c, P=P, restore=True, accumulate_kp=False)
        else:  # MEMIT clients for S2
            deltas, covs, kkts, ks = execute_memit(
                model, tok, reqs, m_hp, device, cache_template=cache_template)
        covs_ref = covs
        all_kkts.append(kkts)
        client = {}
        for li, w in enumerate(wnames):
            adj_k, resid = deltas[w]
            client[w] = resid.double() @ adj_k.double().T   # Delta_paper (d x d0)
        per_client.append(client)
        g0mem.phase_end("client_edit", detail=f"c{i}")
        torch.cuda.empty_cache()

    # Merge
    g0mem.phase_start("merge")
    with torch.no_grad():
        for li, w in enumerate(wnames):
            Pl = P[li].double()
            deltas_paper = [per_client[i][w] for i in range(num_clients)]
            if merging_method == "NS":
                PAP = [Pl @ all_kkts[i][w].double() @ Pl for i in range(num_clients)]
                merged = ns_aligned_merge_layer(deltas_paper, PAP, L2)
                # Save N_all per layer so the operator deviation
                # ||N_all - N_G||/||N_G|| can be computed against the centralized
                # operator N_G (diagnostic).
                I = torch.eye(Pl.shape[0], dtype=torch.float64)
                N_all = sum(PAP) + L2 * I
                torch.save(N_all, run_dir / f"N_all_layer_{layer_of[w]}.pt")
            elif merging_method == "S3":
                merged = s3_average_layer(deltas_paper)
            elif merging_method == "S2":
                # naive CollabEdit nondestructive merge on MEMIT clients, then
                # right-project by P0
                cov = covs_ref[w].cpu().double()
                acc = torch.zeros_like(deltas_paper[0])
                A = deepcopy(cov)
                for i in range(num_clients):
                    kkt = all_kkts[i][w].cpu().double()
                    acc = acc + deltas_paper[i] @ (kkt + cov)
                    A = A + kkt
                merged_naive = acc @ torch.linalg.inv(A)
                merged = s2_rightproject_layer(merged_naive, Pl)

            # Save the merged delta so the sequential-drift metric
            # ||Delta^NS - Delta_GLOBAL||/||Delta_GLOBAL|| is computable against
            # the centralized update (diagnostic).
            torch.save(merged.detach().cpu(), run_dir / f"Delta_merged_layer_{layer_of[w]}.pt")

            mom2 = covs_ref[w].double() / a_hp.mom2_update_weight
            kl, fro, kl_norm, oons = kl_leak(merged, mom2, Pl)
            kl_log(phase="merged", method=merging_method, layer=layer_of[w],
                   n_edits=num_edits, kl=kl, fro=fro, kl_norm=kl_norm, oons=oons)

            w2 = nethook.get_parameter(model, w)
            w2[...] += upd_matrix_match_shape(merged, w2.shape).to(w2.device, w2.dtype)
    g0mem.phase_end("merge")

    # Evaluate
    torch.cuda.empty_cache()
    g0mem.phase_start("evaluation")
    tmpl = str(run_dir / "{}_edits-case_{}.json")
    for ri, record in enumerate(record_chunks):
        out_file = Path(tmpl.format(num_edits, record["case_id"]))
        if out_file.exists():
            continue
        post = ds_eval(model, tok, record,
                       *([snips, vec] if record["case_id"] % generation_test_interval == 0
                         else [None, None]))
        entry = {"case_id": record["case_id"], "grouped_case_ids": case_ids,
                 "num_edits": num_edits, "requested_rewrite": record["requested_rewrite"],
                 "time": 0.0, "post": post}
        with open(out_file, "w") as f:
            json.dump({"global_edit": {}, "collaborative_edit": entry}, f, indent=1)
        # Per-case allocator hygiene to prevent eval-phase fragmentation OOM.
        # Numerically inert (frees cached memory only; no formula/batch-shape
        # change).
        torch.cuda.empty_cache()
    g0mem.phase_end("evaluation")
    print(f"[G2] {merging_method} done -> {run_dir}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--merging_method", required=True, choices=["NS", "S2", "S3"])
    p.add_argument("--model_name", default="gpt2-xl")
    p.add_argument("--hparams_fname", default="gpt2-xl.json")
    p.add_argument("--ds_name", default="mcf", choices=["mcf", "zsre"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset_size_limit", type=int, default=None)
    p.add_argument("--skip_generation_tests", action="store_true")
    p.add_argument("--generation_test_interval", type=int, default=1)
    p.add_argument("--num_clients", type=int, default=3)
    p.add_argument("--num_edits_per_client_per_round", type=int, default=20)
    p.add_argument("--use_cache", action="store_true")
    p.add_argument("--shard_seed", type=int, default=None)
    a = p.parse_args()
    main(a.merging_method, a.model_name, a.ds_name, a.dataset_size_limit,
         a.skip_generation_tests, a.generation_test_interval, a.device,
         num_edits_pC_pR=a.num_edits_per_client_per_round,
         num_clients=a.num_clients, use_cache=a.use_cache, shard_seed=a.shard_seed,
         hparams_fname=a.hparams_fname)
