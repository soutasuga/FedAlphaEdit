# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sota Sugawara, Yukihiko Okada
# Builds on CollabEdit / MEMIT / AlphaEdit (MIT); see THIRD_PARTY_NOTICES.md.
"""AlphaEdit editor for the CollabEdit collaborative pipelines.

Per-client editor for the naive combination (per-client AlphaEdit +
CollabEdit merge), the proposed method, and the centralized baseline
(GLOBAL AlphaEdit, one batch). The editing equation is AlphaEdit's
(verbatim semantics):

    A     = P (K K^T + C_p) + L2 * I          (CPU, fp64)
    Delta = A^{-1} P K resid^T                 applied layer-by-layer

with P the common null-space projector built from the shared Wikipedia
mom2 statistics and C_p the client's *local* past-
edit key covariance (K_p accumulation stays inside the client -- naive
definition; no cross-client sharing).

The solve is returned in CollabEdit factor form so the unchanged CollabEdit
server merge can consume it:

    deltas[w] = (adj_k, resid)  with  adj_k = A^{-1} P K   (d0 x u, fp64)
    so   upd_paper = resid @ adj_k.T   (d x d0)  ==  Delta^T

(Algebraically identical to AlphaEdit's solve(A, P K resid^T); the factor
form solves the same linear system with K's u columns as RHS instead of
resid^T's d columns -- associativity only, no equation change. The
centralized and naive runs both use this one code path, so any fp64 rounding convention is common
to both arms of the contrast.)

Also returns covs (mom2_update_weight * mom2) and kkts (K K^T) exactly as
memit_main.execute_memit does, because the CollabEdit merge weights the
client updates with them (merge rule unchanged -- naive means naive).

No model mutation leaks out when restore=True: the edit is applied
layer-by-layer during computation (AlphaEdit's sequential-layer semantics,
so layer l+1's keys see layer l's update, exactly as in the AlphaEdit
repo), then the edited weights are reverted before returning.
"""
import os
from copy import deepcopy
from pathlib import Path
from time import time as _g1_time
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words
from .memit_main import get_cov, get_context_templates, upd_matrix_match_shape
from .alphaedit_hparams import AlphaEditHyperParams


def projection_path(hparams) -> Path:
    """Shared on-disk location of the null-space projector P.

    One file reused by every run -- all methods must share the identical P
    object, so it is built once and loaded thereafter. Lives next to the
    shared z cache.
    """
    default = KV_DIR.parent / "P" / f"{hparams.model_name}_nullspace_P.pt"
    return Path(os.environ.get("G1_P_PATH", str(default)))


def load_or_build_projection(model, tok, hparams, device) -> torch.Tensor:
    """Build (or load) P[i] per edited layer: U_0 U_0^T from the SVD of the
    shared Wikipedia mom2 statistic, singular values < nullspace_threshold.
    Identical math to AlphaEdit/experiments/evaluate.py::get_project."""
    path = projection_path(hparams)
    W_out = nethook.get_parameter(
        model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
    )
    # Key-dim orientation ported verbatim from AlphaEdit/experiments/evaluate.py
    # gpt2-xl (Conv1D) -> shape[0]; nn.Linear models
    # (gpt-j / Llama3 / phi / Qwen) -> shape[1] (down_proj input dim, 8960 for Qwen).
    if hparams.model_name == "gpt2-xl":
        d0 = W_out.shape[0]  # gpt2-xl c_proj: (6400, 1600) -> key dim 6400
    elif hparams.model_name in ["EleutherAI_gpt-j-6B", "Llama3-8B", "phi-1.5", "Qwen2.5-1.5B"]:
        d0 = W_out.shape[1]  # nn.Linear weight (out, in): key dim = in
    else:
        raise NotImplementedError(
            f"[G1] unknown model_name {hparams.model_name!r} for d0 orientation"
        )
    del W_out
    if path.exists():
        P = torch.load(path, map_location="cpu")
        if P.shape != (len(hparams.layers), d0, d0):
            raise ValueError(
                f"[G1] cached P shape {tuple(P.shape)} does not match "
                f"({len(hparams.layers)}, {d0}, {d0}) -- refusing to continue"
            )
        print(f"[G1] loaded shared null-space projector from {path}", flush=True)
        return P
    print(f"[G1] building null-space projector (one-time) -> {path}", flush=True)
    P = torch.zeros((len(hparams.layers), d0, d0), device="cpu")
    for i, layer in enumerate(hparams.layers):
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            device,
        ).cpu()
        U, S, _ = torch.linalg.svd(cov, full_matrices=False)
        small_singular_indices = (S < hparams.nullspace_threshold).nonzero(
            as_tuple=True
        )[0]
        print(
            f"[G1] layer {layer}: null-space dim {len(small_singular_indices)}",
            flush=True,
        )
        P[i, :, :] = U[:, small_singular_indices] @ U[:, small_singular_indices].T
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(P, path)
    return P


def execute_alphaedit_collab(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    device: str,
    cache_template: Optional[str] = None,
    cache_c: Optional[torch.Tensor] = None,
    P: Optional[torch.Tensor] = None,
    restore: bool = True,
    accumulate_kp: bool = False,
):
    """AlphaEdit batch edit, CollabEdit-compatible returns.

    restore=True  -> client mode: model is reverted, only factors leave.
    restore=False -> GLOBAL mode (centralized): edit stays applied to the model.
    accumulate_kp -> after the edit, K K^T of the freshly edited keys is
        accumulated into cache_c in place (fp32 CPU, AlphaEdit
        convention), computed on the *locally edited* model exactly as
        AlphaEdit does after each chunk. Used by multiRound (K_p carries
        across rounds inside one client only).
    """
    if P is None:
        raise ValueError("[G1] P projector is required (no fallback)")
    if cache_c is None:
        raise ValueError("[G1] cache_c (K_p accumulator) is required; pass zeros")

    deltas, kkts, covs, ks = {}, {}, {}, {}

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    # z retrieval from the shared method-independent cache.
    # Cache-build time is measured separately.
    _g1_z_t0 = _g1_time()
    _g1_cache_hits = 0
    for request in requests:
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if cache_fname is not None and cache_fname.exists():
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(device))
                data_loaded = True
                _g1_cache_hits += 1
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")
        if not data_loaded:
            cur_z = compute_z(
                model, tok, request, hparams, z_layer, context_templates, device
            )
            z_list.append(cur_z)
            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(cache_fname, **{"v_star": cur_z.detach().cpu().numpy()})
    print(
        f"[G0] chunk_z_seconds={_g1_time() - _g1_z_t0:.2f} "
        f"cache_hits={_g1_cache_hits}/{len(requests)}",
        flush=True,
    )
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(
            model, tok, requests, hparams, layer, context_templates
        ).T
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # spread residual

        # AlphaEdit closed-form solve on CPU fp64 (standard config)
        layer_ks_d = layer_ks.detach().cpu().double()
        resid_d = resid.detach().cpu().double()
        P_d = P[i, :, :].double()
        A = (
            P_d @ (layer_ks_d @ layer_ks_d.T + cache_c[i, :, :].double())
            + hparams.L2
            * torch.eye(layer_ks_d.shape[0], dtype=torch.double, device="cpu")
        )
        adj_k = torch.linalg.solve(A, P_d @ layer_ks_d)  # (d0, u)
        upd_matrix = adj_k @ resid_d.T  # == A^{-1} P K resid^T  (d0, d)

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix.to(
                device=weights[weight_name].device,
                dtype=weights[weight_name].dtype,
            )
            deltas[weight_name] = (adj_k, resid_d)
            kkts[weight_name] = layer_ks_d @ layer_ks_d.T
            covs[weight_name] = (
                hparams.mom2_update_weight
                * get_cov(
                    model,
                    tok,
                    hparams.rewrite_module_tmp.format(layer),
                    hparams.mom2_dataset,
                    hparams.mom2_n_samples,
                    hparams.mom2_dtype,
                    device,
                ).double()
            ).cpu()
            ks[weight_name] = layer_ks_d

        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    if accumulate_kp:
        # K_p accumulation on the locally edited model (AlphaEdit
        # post-chunk convention), before any restore. Client-local only.
        for i, layer in enumerate(hparams.layers):
            layer_ks = compute_ks(
                model, tok, requests, hparams, layer, context_templates
            ).T
            layer_ks_c = layer_ks.detach().cpu().float()
            cache_c[i, :, :] += layer_ks_c @ layer_ks_c.T
            del layer_ks
        torch.cuda.empty_cache()

    if restore:
        with torch.no_grad():
            for k, v in weights.items():
                v[...] = weights_copy[k]
    torch.cuda.empty_cache()

    return deltas, covs, kkts, ks


def kl_leak(delta_paper: torch.Tensor, mom2: torch.Tensor, P_layer=None):
    """Leak metrics for one layer.

    KL = ||Delta K_0||_F computed as sqrt(trace(Delta mom2 Delta^T)) with
    mom2 the *sample-normalized* second moment (moment(), the fixed
    normalization used identically in every run) from the shared
    Wikipedia npz. delta_paper is the (d, d0) "paper orientation" update.
    Returns (kl, fro, kl_norm, oons).

    oons: ||Delta (I - P)||_F / ||Delta||_F,
    the fraction of the update outside the approximate null space. For a
    per-client AlphaEdit update this is ~0 (the solve keeps the update in
    range(P) exactly); a nonzero value after merging is the direct trace
    of the projection-incompatibility mechanism. None
    if P_layer is not given.
    """
    d = delta_paper.double()
    m = mom2.double()
    kl2 = ((d @ m) * d).sum()
    kl = float(kl2.clamp_min(0).sqrt())
    fro = float(torch.linalg.norm(d))
    oons = None
    if P_layer is not None:
        out = d - d @ P_layer.double()
        oons = float(torch.linalg.norm(out)) / fro if fro > 0 else float("nan")
    return kl, fro, (kl / fro if fro > 0 else float("nan")), oons
