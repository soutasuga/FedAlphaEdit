# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sota Sugawara, Yukihiko Okada
# Builds on CollabEdit / MEMIT / AlphaEdit (MIT); see THIRD_PARTY_NOTICES.md.
"""Single-round baselines: naive combination (per-client AlphaEdit +
CollabEdit merge) and centralized AlphaEdit (all edits in one batch).

Derived from experiments/merging_singleRound.py; the editor is
memit.alphaedit_collab (AlphaEdit equations, factor-form returns) and the
CollabEdit merge equation is kept verbatim. Additions are measurement only:
leak-KL jsonl, GLUE eval (ported AlphaEdit harness; not shipped in this
release), and seeded client sharding (seed == allocation RNG).

Supported merging_method: "GLOBAL" (centralized) and "CollabEdit" (naive).
Anything else fails loudly.
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
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import upd_matrix_match_shape
from memit.alphaedit_collab import (
    execute_alphaedit_collab,
    kl_leak,
    load_or_build_projection,
    projection_path,
)
from memit.alphaedit_hparams import AlphaEditHyperParams
# Per-client MEMIT control: verbatim ports from g2_singleRound.py;
# inert unless --client_editor memit is passed.
from memit.memit_hparams import MEMITHyperParams
from memit.memit_main import execute_memit
from util import nethook
from util.globals import *
from util.g0_mem import G0MemTracker
from copy import deepcopy

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def shard_clients(record_chunks, num_clients, per_client, shard_seed, shard_by):
    """Assign the chunk's records to clients.

    Default (shard_seed None, shard_by none): upstream-contiguous slices.
    shard_seed k: uniform random permutation from RandomState(k) -- the
    "seed" of every multi-seed experiment. shard_by relation: group by
    relation_id, greedy-balance whole groups onto clients (exploratory
    non-IID probe; client sizes may then differ slightly).
    """
    n = len(record_chunks)
    if shard_by == "relation":
        groups = {}
        for idx, record in enumerate(record_chunks):
            rid = record["requested_rewrite"]["relation_id"]
            groups.setdefault(rid, []).append(idx)
        order = sorted(groups.values(), key=len, reverse=True)
        alloc = [[] for _ in range(num_clients)]
        for group in order:
            target = min(range(num_clients), key=lambda c: len(alloc[c]))
            alloc[target].extend(group)
        return alloc
    idx = list(range(n))
    if shard_seed is not None:
        rng = np.random.RandomState(shard_seed)
        idx = rng.permutation(n).tolist()
    return [
        idx[per_client * i : per_client * (i + 1)] for i in range(num_clients)
    ]


def run_glue(model, tok, g0mem, save_location, tag, edit_num, case_ids, n_tests):
    try:
        from glue_eval.glue_eval import GLUEEval
    except ImportError as e:  # GLUE harness is not part of the public release
        raise SystemExit("GLUE evaluation is not included in this release; "
                         "run with --skip_glue") from e
    g0mem.phase_start("glue_eval")
    glue_results = {"edit_num": edit_num}
    if case_ids is not None:
        glue_results["case_id"] = case_ids
    out_file = save_location + f"{tag}.json"
    glue_eval = GLUEEval(model, tok, number_of_tests=n_tests)
    glue_results = glue_eval.evaluate(
        glue_results,
        out_file,
        nli_flag=True,
        sst_flag=True,
        cola_flag=True,
        rte_flag=True,
        mmlu_flag=True,
        mrpc_flag=True,
    )
    output_filename = out_file.replace(".json", "_glue.json")
    with open(output_filename, "w") as f:
        json.dump(glue_results, f, indent=4)
    g0mem.phase_end("glue_eval", detail=tag)


def main(
    merging_method: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    skip_generation_tests: bool,
    generation_test_interval: int,
    device: str,
    num_edits_pC_pR: int = 1,
    num_clients: int = 10,
    use_cache: bool = False,
    shard_seed: int = None,
    shard_by: str = "none",
    glue_tests: int = 100,
    skip_glue: bool = False,
    client_editor: str = "alphaedit",
):
    if client_editor not in ("alphaedit", "memit"):
        raise ValueError(f"[G1] unknown client_editor {client_editor!r}")
    if client_editor == "memit" and merging_method not in ("CollabEdit", "GLOBAL"):
        raise ValueError(
            "[G1] client_editor='memit' is wired only for the CollabEdit merge "
            "(per-client MEMIT control) and GLOBAL (centralized MEMIT); "
            "refusing to run"
        )
    if merging_method not in ("GLOBAL", "CollabEdit"):
        raise ValueError(
            f"[G1] merging_method={merging_method!r} is not supported by this driver "
            "(only GLOBAL / CollabEdit); refusing to run"
        )
    # Standard config: hard per-process VRAM cap (Linux-like allocator
    # discipline on Windows/WDDM); see merging_multiRound.py.
    _cap_gb = os.environ.get("G0_CUDA_MEM_CAP_GB")
    if _cap_gb:
        _total = torch.cuda.get_device_properties(0).total_memory
        _frac = float(_cap_gb) * 2**30 / _total
        torch.cuda.set_per_process_memory_fraction(_frac, 0)
        print(
            f"[G0] cuda_mem_cap_gb={float(_cap_gb):.2f} "
            f"total_gb={_total / 2**30:.2f} fraction={_frac:.4f}",
            flush=True,
        )

    num_edits = num_edits_pC_pR * num_clients

    dir_name = "G1"
    alg_dir = RESULTS_DIR / dir_name
    if alg_dir.exists():
        id_list = [
            int(str(x).split("_")[-1])
            for x in alg_dir.iterdir()
            if str(x).split("_")[-1].isnumeric()
        ]
        run_id = 0 if not id_list else max(id_list) + 1
    else:
        run_id = 0
    run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    g0mem = G0MemTracker(run_dir)

    params_path = HPARAMS_DIR / "AlphaEdit" / hparams_fname
    hparams = AlphaEditHyperParams.from_json(params_path)
    # MEMIT hparams drive the MEMIT control clients only (verbatim from
    # g2_singleRound; None unless the MEMIT control is requested).
    m_hp = (
        MEMITHyperParams.from_json(HPARAMS_DIR / "MEMIT" / "gpt2-xl.json")
        if client_editor == "memit"
        else None
    )
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing G1 AlphaEdit junction with parameters {hparams}")

    kl_path = run_dir / "g1_kl.jsonl"

    def kl_log(**kw):
        with open(kl_path, "a") as f:
            f.write(json.dumps(kw) + "\n")

    g0mem.phase_start("model_load")
    print("Instantiating model")
    _g0_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[
        os.environ.get("G0_MODEL_DTYPE", "bf16")
    ]
    print(f"[G0] model_dtype={_g0_dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=_g0_dtype
    ).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    g0mem.phase_end("model_load")

    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    # Shared method-independent z cache: the files live under <model>_MEMIT
    # regardless of editor (z is editor-agnostic).
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{str(model_name).replace('/', '_')}_MEMIT"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # Common null-space projector P: one shared object for all runs.
    g0mem.phase_start("projection")
    P = load_or_build_projection(model, tok, hparams, device)
    g0mem.phase_end("projection")

    layer_of_wname = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": layer
        for layer in hparams.layers
    }

    glue_save_location = str(run_dir) + "/glue_eval/"
    os.makedirs(glue_save_location, exist_ok=True)

    record_chunks = list(islice(iter(ds), num_edits))
    if len(record_chunks) < num_edits:
        raise ValueError(
            f"[G1] dataset yielded {len(record_chunks)} < num_edits {num_edits}"
        )
    case_ids = [record["case_id"] for record in record_chunks]
    case_result_template = str(run_dir / "{}_edits-case_{}.json")
    etc_args = dict(cache_template=cache_template)

    # Base GLUE (pre-edit reference)
    if not skip_glue:
        run_glue(model, tok, g0mem, glue_save_location, "base", -1, None, glue_tests)

    exec_time = 0.0
    if merging_method == "GLOBAL":
        # Centralized: one AlphaEdit batch over all cases; edit stays in the model.
        cache_c = torch.zeros(
            (len(hparams.layers), P.shape[1], P.shape[1]), device="cpu"
        )
        g0mem.phase_start("client_edit")
        start = time()
        if client_editor == "memit":
            # Centralized MEMIT: one central
            # MEMIT batch. execute_memit computes deltas and RESTORES the
            # original weights (unlike the AlphaEdit restore=False path), so
            # the update is applied explicitly below with the same application
            # form as the merge path (upd_matrix_match_shape + dtype cast).
            deltas, covs, kkts, ks = execute_memit(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **record["requested_rewrite"]}
                    for record in record_chunks
                ],
                m_hp,
                device,
                cache_template=cache_template,
            )
            with torch.no_grad():
                for w_name, (adj_k, resid) in deltas.items():
                    delta_paper = resid.double() @ adj_k.double().T
                    w2 = nethook.get_parameter(model, w_name)
                    w2[...] += upd_matrix_match_shape(delta_paper, w2.shape).to(
                        w2.device, w2.dtype
                    )
        else:
            deltas, covs, kkts, ks = execute_alphaedit_collab(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **record["requested_rewrite"]}
                    for record in record_chunks
                ],
                hparams,
                device,
                cache_c=cache_c,
                P=P,
                restore=False,
                accumulate_kp=False,
                **etc_args,
            )
        exec_time = time() - start
        print("Execution took", exec_time)
        g0mem.phase_end("client_edit", detail="global")
        torch.cuda.empty_cache()

        g0mem.phase_start("kl_measure")
        for w_name, (adj_k, resid) in deltas.items():
            delta_paper = resid @ adj_k.T
            mom2 = covs[w_name] / hparams.mom2_update_weight
            li = hparams.layers.index(layer_of_wname[w_name])
            kl, fro, kl_norm, oons = kl_leak(delta_paper, mom2, P[li])
            kl_log(
                phase="global",
                round=0,
                client=None,
                layer=layer_of_wname[w_name],
                n_edits=len(record_chunks),
                kl=kl,
                fro=fro,
                kl_norm=kl_norm,
                oons=oons,
            )
        g0mem.phase_end("kl_measure")
        alloc = None
    else:
        # Naive: per-client AlphaEdit on the base model, CollabEdit merge.
        alloc = shard_clients(
            record_chunks, num_clients, num_edits_pC_pR, shard_seed, shard_by
        )
        with open(run_dir / "g1_client_allocation.json", "w") as f:
            json.dump(
                {
                    "shard_seed": shard_seed,
                    "shard_by": shard_by,
                    "clients": {
                        str(i): [record_chunks[j]["case_id"] for j in alloc[i]]
                        for i in range(num_clients)
                    },
                },
                f,
                indent=1,
            )

        deltas_list = []
        all_kkts = []
        covs = None
        for i in range(num_clients):
            g0mem.phase_start("client_edit")
            start = time()
            records = [record_chunks[j] for j in alloc[i]]
            # K_p accumulator: zero at round start, client-local. In a
            # single round it is only read (as zero) by the solve;
            # accumulation is skipped because no later round reads it.
            cache_c = torch.zeros(
                (len(hparams.layers), P.shape[1], P.shape[1]), device="cpu"
            )
            if client_editor == "memit":
                # verbatim port of g2_singleRound's S2 MEMIT client call;
                # the CollabEdit merge below is unchanged either way.
                delta_i, covs, kkts, ks = execute_memit(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **record["requested_rewrite"]}
                        for record in records
                    ],
                    m_hp,
                    device,
                    cache_template=cache_template,
                )
            else:
                delta_i, covs, kkts, ks = execute_alphaedit_collab(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **record["requested_rewrite"]}
                        for record in records
                    ],
                    hparams,
                    device,
                    cache_c=cache_c,
                    P=P,
                    restore=True,
                    accumulate_kp=False,
                    **etc_args,
                )
            all_kkts.append(kkts)
            deltas_list.append(delta_i)
            exec_time = time() - start
            print("client:", i, " Execution took ", exec_time)
            g0mem.phase_end("client_edit", detail=f"c{i}")
            del cache_c
            torch.cuda.empty_cache()

        # Materialize per-client updates (paper orientation d x d0), exactly
        # what the CollabEdit merge consumes.
        upd_matrix_dict_list = []
        for delta in deltas_list:
            upd_matrix_dict = {}
            for w_name, (key_mat, val_mat) in delta.items():
                upd_matrix_dict[w_name] = val_mat @ key_mat.T
            upd_matrix_dict_list.append(upd_matrix_dict)
        del deltas_list
        torch.cuda.empty_cache()

        # Per-client leak KL (mean_i KL(Delta_i) vs merged)
        g0mem.phase_start("kl_measure")
        for i in range(num_clients):
            for w_name, upd in upd_matrix_dict_list[i].items():
                mom2 = covs[w_name] / hparams.mom2_update_weight
                li = hparams.layers.index(layer_of_wname[w_name])
                kl, fro, kl_norm, oons = kl_leak(upd, mom2, P[li])
                kl_log(
                    phase="client",
                    round=0,
                    client=i,
                    layer=layer_of_wname[w_name],
                    n_edits=len(alloc[i]),
                    kl=kl,
                    fro=fro,
                    kl_norm=kl_norm,
                    oons=oons,
                )
        g0mem.phase_end("kl_measure")

        # CollabEdit merge, equation verbatim from merging_singleRound.py:
        # delta = [sum_i upd_i (kkt_i + cov)] (cov + sum kkt_i)^-1
        # on CPU fp64. Only the final delta moves to the GPU.
        g0mem.phase_start("merge")
        with torch.no_grad():
            for w_name, delta_demo in upd_matrix_dict_list[0].items():
                delta_Nx10 = torch.zeros_like(delta_demo)
                print("Using Nondestructive Merge")
                cov = covs[w_name].cpu()
                A = deepcopy(cov)
                for i in range(num_clients):
                    all_kkt = all_kkts[i][w_name].cpu()
                    delta_Nx10 = delta_Nx10 + upd_matrix_dict_list[i][w_name] @ (
                        all_kkt + cov
                    )
                    A = A + all_kkt
                delta_Nx10 = delta_Nx10 @ A.inverse()

                # measurement only: leak KL of the merged update
                mom2 = covs[w_name] / hparams.mom2_update_weight
                li = hparams.layers.index(layer_of_wname[w_name])
                kl, fro, kl_norm, oons = kl_leak(delta_Nx10, mom2, P[li])
                kl_log(
                    phase="merged",
                    round=0,
                    client=None,
                    layer=layer_of_wname[w_name],
                    n_edits=num_edits,
                    kl=kl,
                    fro=fro,
                    kl_norm=kl_norm,
                    oons=oons,
                )

                w2 = nethook.get_parameter(model, w_name)
                delta_Nx10 = upd_matrix_match_shape(delta_Nx10, w2.shape)
                w2[...] += delta_Nx10.to(w2.device, w2.dtype)
                del delta_demo, delta_Nx10
        g0mem.phase_end("merge")
        print("models generated")

    # Final GLUE on the edited/merged model
    if not skip_glue:
        run_glue(
            model, tok, g0mem, glue_save_location, "final", num_edits, None,
            glue_tests,
        )

    # Evaluate new model (upstream protocol; key layout kept so
    # CollabEdit_summarize --edit_category works unchanged)
    torch.cuda.empty_cache()
    g0mem.phase_start("evaluation")
    gen_test_vars = [snips, vec]
    for _g1_ri, record in enumerate(record_chunks):
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))
        if out_file.exists():
            print(f"Skipping {out_file}; already exists")
            continue
        post = ds_eval_method(
            model,
            tok,
            record,
            *(
                gen_test_vars
                if record["case_id"] % generation_test_interval == 0
                else [None, None]
            ),
        )
        entry = {
            "case_id": record["case_id"],
            "grouped_case_ids": case_ids,
            "num_edits": num_edits,
            "requested_rewrite": record["requested_rewrite"],
            "time": exec_time,
            "post": post,
        }
        metrics = {
            "global_edit": entry if merging_method == "GLOBAL" else {},
            "collaborative_edit": {} if merging_method == "GLOBAL" else entry,
        }
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)
        # Per-case allocator hygiene to prevent eval-phase fragmentation OOM.
        # Numerically inert (frees cached memory only; no formula/batch-shape
        # change).
        torch.cuda.empty_cache()
    g0mem.phase_end("evaluation")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--merging_method", type=str, required=True,
                        choices=["GLOBAL", "CollabEdit"])
    parser.add_argument("--model_name", default="gpt2-xl",
                        choices=["gpt2-medium", "gpt2-large", "gpt2-xl",
                                 "EleutherAI/gpt-j-6B", "Qwen/Qwen2.5-1.5B"])
    parser.add_argument("--hparams_fname", type=str, default="gpt2-xl.json")
    parser.add_argument("--ds_name", choices=["mcf", "cf", "zsre"],
                        default="mcf")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dataset_size_limit", type=int, default=None)
    parser.add_argument("--skip_generation_tests", action="store_true")
    parser.add_argument("--generation_test_interval", type=int, default=1)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_edits_per_client_per_round", type=int, default=1)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--shard_seed", type=int, default=None,
                        help="seed: RNG for case->client allocation")
    parser.add_argument("--shard_by", choices=["none", "relation"],
                        default="none", help="exploratory non-IID probe: relation sharding")
    parser.add_argument("--glue_tests", type=int, default=100)
    parser.add_argument("--skip_glue", action="store_true")
    parser.add_argument("--client_editor", choices=["alphaedit", "memit"],
                        default="alphaedit")
    args = parser.parse_args()

    main(
        args.merging_method,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.device,
        num_edits_pC_pR=args.num_edits_per_client_per_round,
        num_clients=args.num_clients,
        use_cache=args.use_cache,
        shard_seed=args.shard_seed,
        shard_by=args.shard_by,
        glue_tests=args.glue_tests,
        skip_glue=args.skip_glue,
        client_editor=args.client_editor,
    )
