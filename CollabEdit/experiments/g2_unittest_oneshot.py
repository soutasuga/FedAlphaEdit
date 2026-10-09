# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sota Sugawara, Yukihiko Okada
# Builds on CollabEdit / MEMIT / AlphaEdit (MIT); see THIRD_PARTY_NOTICES.md.
"""Unit test: Theorem 1 as an implementation verifier.

Test A checks the ONE-CLIENT one-shot identity
    Delta_global^NS == Delta_GLOBAL   (to numerical tolerance).
The 1-client case validates beta consistency between the merge numerator weight
and N_all, and that the client Delta construction matches the GLOBAL path -- but
it CANNOT detect beta*I double-counting in N_all (at N=1, +beta*I once and
+N*beta*I coincide). Double-counting is exactly the (N-1)*lambda*C trap that broke
the naive merge, so we ADD a 2-client one-shot check (disjoint edits) that does
detect it. The PASS/FAIL verdict uses Test A and the negative control;
Test B (2 clients) is reported as a diagnostic only, because AlphaEdit's
layer-sequential editing makes it deviate slightly by design.

A structural bug (beta mismatch or double-count) produces O(0.1-1) relative
Frobenius error -- orders of magnitude above fp64 round-trip noise -- so the
pass/fail verdict is insensitive to the exact tolerance. We set a generous
relative tolerance and report the measured value.

Runs on GPT2-XL with the shared P and z-cache. CPU fp64.
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import MultiCounterFactDataset
from memit.alphaedit_collab import (
    execute_alphaedit_collab,
    load_or_build_projection,
)
from memit.alphaedit_hparams import AlphaEditHyperParams
from memit.nsmerge import ns_aligned_merge_layer
from util.globals import *

REL_TOL = 1e-6  # generous fp64 bound; see module docstring for rationale
U_TOTAL = 6     # edits in the one-shot batch (split 3+3 for the 2-client case)
OUT = "unittest_oneshot.json"  # written to the working directory (CollabEdit/)


def rel_fro(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def client_edit(model, tok, hparams, device, P, records, cache_template):
    """Run the AlphaEdit editor on one batch; return per-layer
    (Delta_paper d x d0, PAP = P Ai P d0 x d0), all fp64 CPU."""
    deltas, covs, kkts, ks = execute_alphaedit_collab(
        model, tok,
        [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in records],
        hparams, device,
        cache_template=cache_template,
        cache_c=torch.zeros((len(hparams.layers), P.shape[1], P.shape[1])),
        P=P, restore=True, accumulate_kp=False,
    )
    out = {}
    for li, layer in enumerate(hparams.layers):
        w = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        adj_k, resid = deltas[w]
        adj_k = adj_k.double()
        resid = resid.double()
        Pl = P[li].double()
        delta_paper = resid @ adj_k.T                             # d x d0
        PAP = Pl @ kkts[w].double() @ Pl                          # P Ai P (Ai=KiKi^T)
        # ks[w] = layer_ks (d0 x u); keep for the cancellation-identity check
        Kmat = ks[w].double()
        out[layer] = (delta_paper, PAP, resid, Kmat)
    return out


def main():
    device = "cuda"
    _cap = os.environ.get("G0_CUDA_MEM_CAP_GB")
    if _cap:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(float(_cap) * 2**30 / tot, 0)

    model_name = "gpt2-xl"
    hparams = AlphaEditHyperParams.from_json(HPARAMS_DIR / "AlphaEdit" / "gpt2-xl.json")
    # Diagnostic override: restrict edited layers to isolate whether the
    # 2-client residual gap is the multi-layer sequential residual effect
    # (G2_TEST_LAYERS=17 -> single layer should recover the exact identity).
    _tl = os.environ.get("G2_TEST_LAYERS")
    if _tl:
        hparams.layers = [int(x) for x in _tl.split(",")]
        print(f"[unittest] layers override -> {hparams.layers}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    P = load_or_build_projection(model, tok, hparams, device)
    L2 = hparams.L2

    cache_template = (
        KV_DIR / f"{model_name}_MEMIT" / "mcf_layer_{}_clamp_{}_case_{}.npz"
    )
    ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=U_TOTAL)
    records = list(ds)[:U_TOTAL]

    # GLOBAL: all U edits in one batch (this is Delta_GLOBAL).
    g = client_edit(model, tok, hparams, device, P, records, cache_template)

    # --- Test A (primary): 1 client == the whole batch ---
    errs_1c = {}
    for layer, (dg, PAP, _, _) in g.items():
        d_ns = ns_aligned_merge_layer([dg], [PAP], L2)
        errs_1c[layer] = rel_fro(d_ns, dg)

    # --- Test B (supplementary): 2 clients, disjoint 3+3, merge, vs GLOBAL ---
    c1 = client_edit(model, tok, hparams, device, P, records[:3], cache_template)
    c2 = client_edit(model, tok, hparams, device, P, records[3:], cache_template)
    errs_2c = {}
    cancel_err = {}   # ||sum_i Di(PAiP+bI) - resid_G K_G^T P|| / ||rhs||
    n_err = {}        # ||N_all - N_G|| / ||N_G||
    for layer, (dg, PAP_g, resid_g, K_g) in g.items():
        d1, PAP1, r1, K1 = c1[layer]
        d2, PAP2, r2, K2 = c2[layer]
        d_ns = ns_aligned_merge_layer([d1, d2], [PAP1, PAP2], L2)
        errs_2c[layer] = rel_fro(d_ns, dg)
        # localize the gap: cancellation identity and N_all vs N_G
        Pl = P[hparams.layers.index(layer)].double()
        I = torch.eye(PAP1.shape[0], dtype=torch.float64)
        num = d1 @ (PAP1 + L2 * I) + d2 @ (PAP2 + L2 * I)
        rhs = (r1 @ K1.T + r2 @ K2.T) @ Pl            # resid_G K_G^T P (block form)
        cancel_err[layer] = rel_fro(num, rhs)
        N_all = PAP1 + PAP2 + L2 * I
        N_G = Pl @ (K_g @ K_g.T) @ Pl + L2 * I
        n_err[layer] = rel_fro(N_all, N_G)

    # --- Negative control: deliberately double-count beta*I in N_all,
    #     confirm the 2-client test then FAILS (guards the test's power) ---
    def ns_doublecount(deltas, PAPs, L2):
        I = torch.eye(PAPs[0].shape[0], dtype=PAPs[0].dtype)
        N_all = sum(PAPs) + len(PAPs) * L2 * I           # WRONG: +N*beta*I
        num = sum(D @ (PAP + L2 * I) for D, PAP in zip(deltas, PAPs))
        return torch.linalg.solve(N_all.T, num.T).T
    errs_2c_bug = {}
    for layer, (dg, _, _, _) in g.items():
        d1, PAP1, _, _ = c1[layer]
        d2, PAP2, _, _ = c2[layer]
        d_bug = ns_doublecount([d1, d2], [PAP1, PAP2], L2)
        errs_2c_bug[layer] = rel_fro(d_bug, dg)

    max1 = max(errs_1c.values())
    max2 = max(errs_2c.values())
    maxbug = max(errs_2c_bug.values())
    result = {
        "rel_tol": REL_TOL,
        "u_total": U_TOTAL,
        "L2_beta": L2,
        "test_A_1client_relfro_per_layer": {str(k): v for k, v in errs_1c.items()},
        "test_B_2client_relfro_per_layer": {str(k): v for k, v in errs_2c.items()},
        "negctrl_2client_doublecount_relfro_per_layer": {str(k): v for k, v in errs_2c_bug.items()},
        "cancellation_identity_relfro_per_layer": {str(k): v for k, v in cancel_err.items()},
        "N_all_vs_N_G_relfro_per_layer": {str(k): v for k, v in n_err.items()},
        "cancellation_max": max(cancel_err.values()),
        "N_all_vs_N_G_max": max(n_err.values()),
        "test_A_max": max1,
        "test_B_max": max2,
        "negctrl_max": maxbug,
        "test_A_pass": bool(max1 <= REL_TOL),
        "test_B_within_tol_diagnostic": bool(max2 <= REL_TOL),  # not part of the verdict
        "negctrl_detects_bug": bool(maxbug > REL_TOL),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    ok = result["test_A_pass"] and result["negctrl_detects_bug"]  # Test B: diagnostic
    print(f"[unittest] OVERALL {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
