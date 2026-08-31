"""Constrained fine-tuning editor, adapted from EasyEdit's ft_main.py
(https://github.com/zjunlp/EasyEdit), trimmed to the decoder-only path used here."""

from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

import nethook


def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams,
    copy=False,
    return_orig_weights=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """Fine-tune the weights selected by hparams on the requested rewrites.
    Returns (edited model, copy of the original weights that changed)."""
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_ft(model, tok, requests, hparams, **kwargs)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    return model, weights_copy


def execute_ft(model, tok, requests, hparams, **kwargs) -> Dict[str, torch.Tensor]:
    """Compute weight deltas via gradient descent on the rewrite targets.
    Invariant: model at beginning of function == model at end of function."""
    device = torch.device(f"cuda:{hparams.device}")
    loss_threshold = kwargs.get("loss_threshold", 1e-2)

    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        print(f"Executing FT algo for: [{request['prompt']}] -> [{request['target_new']}]")

    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n and "layernorm" not in n
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]

    opt = torch.optim.AdamW(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)):
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(device)

            if hparams.objective_optimization == "prompt_last":
                last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
                if tok.unk_token_id is not None:
                    loss_mask = torch.ne(target_ids, tok.unk_token_id)
                else:
                    loss_mask = torch.ones_like(target_ids, dtype=torch.bool)
            elif hparams.objective_optimization == "target_new":
                inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
                inputs_targets = tok(inputs_targets, return_tensors="pt", padding=True).to(device)
                num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in inputs["input_ids"].cpu()]
                num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in inputs_targets["input_ids"].cpu()]
                prompt_len = [x + y for x, y in zip(num_pad_toks, num_prompt_toks)]
                prompt_target_len = inputs_targets["input_ids"].size(1)
                label_mask = torch.tensor(
                    [[False] * length + [True] * (prompt_target_len - length) for length in prompt_len]
                ).to(device)
            else:
                raise NotImplementedError(f"{hparams.objective_optimization} is not supported")

            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            if hparams.objective_optimization == "prompt_last":
                probs = torch.nn.functional.log_softmax(
                    model(**inputs).logits[torch.arange(bs), last_token_inds], dim=-1
                )
                loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(1) / loss_mask.sum(1)
                loss = loss.mean()
            else:  # target_new
                logits = model(**inputs_targets).logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs_targets["input_ids"][..., 1:].contiguous()
                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(bs, -1)
                loss = (loss * label_mask[:, 1:]).sum(1) / label_mask[:, 1:].sum(1)
                loss = loss.mean()

            loss_meter.update(loss.item(), n=1)

            if loss.item() >= loss_threshold:
                loss.backward()
                opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(v, min=weights_copy[k] - eps, max=weights_copy[k] + eps)

        print(f"Total loss {loss_meter.avg}")
        if loss_meter.avg < loss_threshold:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")
    model.eval()
    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
