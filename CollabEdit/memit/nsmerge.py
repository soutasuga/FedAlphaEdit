# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sota Sugawara, Yukihiko Okada
# Builds on CollabEdit / MEMIT / AlphaEdit (MIT); see THIRD_PARTY_NOTICES.md.
"""Server-side merges: null-space-aligned merging (proposed) and ablation merges (S2/S3).

Notation (as in the paper):
  P0   -- approximate null-space projector (symmetric, idempotent), threshold 0.02
  Ci   -- client i edit-key covariance = Ki Ki^T. **Ci NEVER includes lambda*C.**
          Regularization (beta*I) is written explicitly on the denominator /
          merge-weight side, never folded into Ci.
  Ai   -- inner symmetric matrix Kp,i Kp,i^T + Ki Ki^T (single round: Kp=0 -> Ai=Ci).
          Ai is NOT the solve operator (that is Ni = P0 Ai P0 + beta*I).
  beta -- AlphaEdit L2 coefficient, implementation value L2 = 10 (NOT 1).

Proposed -- null-space-aligned merging (Theorem 1, exact under the one-shot idealization):
    Delta_global^NS = [ sum_i Delta_i (P0 Ai P0 + beta*I) ] . N_all^{-1}
    N_all = sum_i (P0 Ai P0)  +  beta*I        <-- beta*I added ONCE
  The cancellation identity Delta_i (P0 Ai P0 + beta*I) = resid_i Ki^T P0 holds
  because the client solve keeps adj_k_i in range(P0), so (P0 Ai P0 + beta*I)
  acts as the client's own solve operator Ni on adj_k_i. Adding beta*I once (not
  per client) makes N_all equal the GLOBAL operator N_all = P0 (sum_i Ki Ki^T) P0
  + beta*I, giving Delta_global^NS = Delta_GLOBAL exactly. Double-counting beta*I
  (i.e. + N*beta*I) breaks this and is detectable only at N>=2.

S2 -- merge-only constraint (ablation): per-client *MEMIT* (no null-space in the
    editor) + the CollabEdit nondestructive merge, then right-project the merged
    update by P0. oons(S2) = 0 by construction (right projection) -- its discriminant
    is NS/KL, not oons.

S3 -- simple-average control: Delta_avg = (1/N) sum_i Delta_i (AlphaEdit clients).

All merges run on CPU in float64 (standard config); only the final delta returns
to the GPU in the weight dtype. Equations are exact; no clipping / rescaling
or any other "improvement" beyond the theorem.
"""
import torch


def _eye_like(M):
    return torch.eye(M.shape[0], dtype=M.dtype, device=M.device)


def ns_aligned_merge_layer(deltas_paper, PAP_list, L2):
    """Null-space-aligned merging for one layer.

    deltas_paper : list of (d x d0) fp64 client updates Delta_i (paper orientation)
    PAP_list     : list of (d0 x d0) fp64 projected covariances P0 Ai P0
    L2           : beta (= AlphaEdit L2 = 10)
    returns Delta_global^NS  (d x d0) fp64
    """
    assert len(deltas_paper) == len(PAP_list) and len(PAP_list) > 0
    I = _eye_like(PAP_list[0])
    N_all = sum(PAP_list) + L2 * I  # beta*I ONCE
    num = None
    for D, PAP in zip(deltas_paper, PAP_list):
        term = D @ (PAP + L2 * I)
        num = term if num is None else num + term
    # Delta_NS = num @ inv(N_all); solve is more stable than explicit inverse:
    # (num @ N_all^{-1})^T = N_all^{-T} num^T  -> solve(N_all^T, num^T)
    return torch.linalg.solve(N_all.T, num.T).T


def s2_rightproject_layer(delta_global_naive, P):
    """S2 for one layer: right-project the naive (MEMIT-client) merged update.

    delta_global_naive : (d x d0) fp64 output of the CollabEdit merge on MEMIT clients
    P                  : (d0 x d0) fp64 projector
    returns (d x d0) fp64.  oons == 0 by construction.
    """
    return delta_global_naive @ P


def s3_average_layer(deltas_paper):
    """S3 for one layer: simple average of AlphaEdit client updates."""
    n = len(deltas_paper)
    assert n > 0
    acc = deltas_paper[0].clone()
    for D in deltas_paper[1:]:
        acc = acc + D
    return acc / n
