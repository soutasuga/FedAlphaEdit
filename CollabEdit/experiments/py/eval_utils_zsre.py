"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import os
import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets


def compute_rewrite_quality_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    # Flatten all the evaluated prefixes into one list.
    target_tok = tok(" " + target_new["str"])["input_ids"]
    inp_prompts_og = list(chain(*prob_prompts))
    inp_prompts = [
        el + tok.decode(target_tok[:i])
        for el in inp_prompts_og
        for i in range(len(target_tok))
    ]
    inp_targets = [
        tok.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]

    stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)

    # Predict for neighborhood prompts (dictionary format).
    neighborhood_correct = test_batch_prediction_acc(
        model,
        tok,
        [
            el["prompt"].format(record["requested_rewrite"])
            for el in neighborhood_prompts
        ],
        [el["target"] for el in neighborhood_prompts],
    )

    probs = stuff_probs + neighborhood_correct

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    ret["neighborhood_prompts_correct"] = neighborhood_correct

    return ret


def _tbpa_core(model, tok, prompts: typing.List[str], target):
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        if os.environ.get("Z_EVAL_FP32_HEAD") == "1":
            # fp32 output head: bf16 trunk forward, then
            # gather each sequence's last-real-token hidden state (post ln_f)
            # and project through lm_head in fp32. Widens the argmax tie band
            # beyond bf16 ULP so batch-split ULP noise cannot flip near-ties.
            # Numerically inert unless Z_EVAL_FP32_HEAD=1 (unset path below is
            # byte-identical to the original single-batch computation).
            # Qwen x zsRE: resolve the
            # backbone like the AlphaEdit-verified hasattr fallbacks -- GPT2
            # selects .transformer (identical to prior line), Qwen2 .model.
            # Fail-loud: anything else raises AttributeError.
            backbone = (
                model.transformer if hasattr(model, "transformer") else model.model
            )
            hidden = backbone(**prompt_tok).last_hidden_state
            h = hidden[
                torch.arange(hidden.size(0), device=hidden.device), last_non_masked
            ]
            gathered = h.float() @ model.lm_head.weight.float().T
            ans = torch.argmax(gathered, dim=1)
        else:
            logits = model(**prompt_tok).logits
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[
            "input_ids"
        ]
        # Temporary hack to deal with foreign characters.
        correct_id = correct_id[:, 0].squeeze()

        return (ans == correct_id).detach().cpu().numpy().tolist()


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    # Optional eval-batch splitting to
    # bound activation memory under the 7.4GB cap. Numerically inert unless
    # Z_EVAL_CHUNK is set (>0): the unset path is byte-identical to the prior
    # single-batch computation. Right-padding makes each prompt's last-real-token
    # logit invariant to batch composition.
    chunk = int(os.environ.get("Z_EVAL_CHUNK", "0"))
    if chunk <= 0 or chunk >= len(prompts):
        return _tbpa_core(model, tok, prompts, target)
    out = []
    for i in range(0, len(prompts), chunk):
        p = prompts[i : i + chunk]
        t = target[i : i + chunk] if not isinstance(target, str) else target
        r = _tbpa_core(model, tok, p, t)
        if not isinstance(r, list):  # single-element chunk -> squeeze() scalar
            r = [r]
        out.extend(r)
    return out
