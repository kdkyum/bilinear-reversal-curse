import math
import random

from typing import Dict, List, Tuple, Iterable, Optional, Any
from functools import partial

import torch
import baukit
from misc import ModelAndTokenizer, parse_prompt

# ---------------------------
# Cross-layer operator fitting: h_low(subject) -> z_last(prompt)
# ---------------------------

class HiddenStateExtractor:
    """Extract {residual_pre_attn, attn_out, mlp_out} for a given token position, per layer."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @staticmethod
    def _token_at_pos(x: torch.Tensor, pos: List[int]) -> torch.Tensor:
        # x: (batch, seq, hidden)
        # pos: list of length `batch`
        return torch.stack([x[i, p, :] for i, p in enumerate(pos)]).detach().to("cpu")

    def extract_from_prompt(self, prompts: List[str], pos: List[int]):
        """Extract residual components for a batch of prompts (features at `pos` token per layer)."""
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        is_neox = hasattr(self.model, 'gpt_neox')
        # print(self.model.config)
        num_layers = self.model.config.num_hidden_layers
        transformer_layers = self.model.gpt_neox.layers if is_neox else self.model.transformer.h

        residual_pre_attn = [None] * num_layers
        attn_out = [None] * num_layers
        residual_pre_mlp = [None] * num_layers
        mlp_out = [None] * num_layers

        hooks = []

        # --- Hooks ---
        def pre_hook_layer(idx, module, inputs):
            hidden_states = inputs[0]
            residual_pre_attn[idx] = self._token_at_pos(hidden_states, pos)

        def hook_attn(idx, module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            attn_out[idx] = self._token_at_pos(out, pos)

        def pre_hook_mlp(idx, module, inputs, output):
            hidden_states = inputs[0]
            residual_pre_mlp[idx] = self._token_at_pos(hidden_states, pos)

        def hook_mlp(idx, module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            mlp_out[idx] = self._token_at_pos(out, pos)

        for i, layer in enumerate(transformer_layers):
            hooks.append(layer.register_forward_pre_hook(partial(pre_hook_layer, i)))
            
            attn_module = layer.attention if is_neox else layer.attn
            hooks.append(attn_module.register_forward_hook(partial(hook_attn, i)))
            
            # In NeoX, the pre-MLP layernorm is `post_attention_layernorm`
            # In GPT-2, it's `ln_2`
            ln_2_module = layer.post_attention_layernorm if is_neox else layer.ln_2
            hooks.append(ln_2_module.register_forward_hook(partial(pre_hook_mlp, i)))
            
            hooks.append(layer.mlp.register_forward_hook(partial(hook_mlp, i)))

        # --- Forward pass ---
        try:
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        # --- Stack results ---
        residual_pre_attn = torch.stack(residual_pre_attn, dim=0)
        attn_out = torch.stack(attn_out, dim=0)
        residual_pre_mlp = torch.stack(residual_pre_mlp, dim=0)
        mlp_out = torch.stack(mlp_out, dim=0)

        return {
            "residual_pre_attn": residual_pre_attn,
            "attn_out": attn_out,
            "residual_pre_mlp": residual_pre_mlp,
            "mlp_out": mlp_out,
            "prompts": prompts,
        }


@torch.no_grad()
def compute_hz_pair_batch(
    mt: ModelAndTokenizer,
    prompts: List[str],
    subjects: List[str],
    h_layer: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Compute (h_subj, z_last) for a batch of prompts.
    - h_subj: hidden at subject last token, from layer=h_layer
    - z_last: last hidden state at final position (includes final LN for GPT-2)
    """
    tok = mt.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    
    # subject positions on CPU
    subj_pos = [-2 for i in range(len(prompts))]

    # For GPT-NeoX style models, use the hook-based extractor
    extractor = HiddenStateExtractor(mt.model, mt.tokenizer)
    
    # Extract h_subj and z_last in one pass
    outputs = extractor.extract_from_prompt(prompts, pos=subj_pos)
    if h_layer < mt.model.config.num_hidden_layers:
        hs = outputs["residual_pre_attn"][h_layer]
    else:
        hs = (outputs["residual_pre_mlp"][-1] + outputs["mlp_out"][-1])

    # To get z_last, we need to run again with pos=-1 for all prompts
    # This is because subj_pos can be different for each prompt
    z_last_outputs = extractor.extract_from_prompt(prompts, pos=[-1] * len(prompts))
    
    # For GPT-NeoX, last_hidden_state is after the final layer norm.
    # To reconstruct it, we take the input to the final MLP and add the MLP output.
    # This is equivalent to `residual_pre_mlp + mlp_out` at the last layer.
    z = (z_last_outputs["residual_pre_mlp"][-1] + z_last_outputs["mlp_out"][-1])
    return hs, z, subj_pos


def untuple(x: Any) -> Any:
    """If `x` is a tuple, return the first element."""
    if isinstance(x, tuple):
        return x[0]
    return x


def determine_layer_paths(
    model,
    layers,
    *,
    return_dict = False,
):
    if layers is None:
        n_layer = model.config.num_hidden_layers
        layers = (*range(n_layer),)

    layer_paths = {}
    for layer in layers:
        layer_index = layer
        if layer_index < 0:
            layer_index = model.config.num_hidden_layers + layer

        layer_path = f"gpt_neox.layers.{layer_index}"
        layer_paths[layer] = layer_path

    return layer_paths if return_dict else tuple(layer_paths[la] for la in layers)


@torch.no_grad()
@torch.inference_mode(mode=False)
def fit_jacobian_hz_1order(mt: ModelAndTokenizer,
    ds_split,
    h_layer: int,
    train_size: int = 5,
    relations: Iterable[str] = ("husband", "wife", "father", "mother", "sister", "brother", "son", "daughter"),
    max_pairs_per_rel: int = 8000,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Compute a first-order approximation of the LM between `h` and `z`.

    Very simply, this computes the Jacobian of z with respect to h, as well as
    z - J * h to approximate the bias. Here, z layer is the last layer representation."""
    
    model = mt.model
    device = next(model.parameters()).device
    is_neox = hasattr(model, 'gpt_neox')
    
    by_rel_prompts: Dict[str, List[str]] = {r: [] for r in relations}
    by_rel_subjs: Dict[str, List[str]] = {r: [] for r in relations}
    
    for i in range(len(ds_split)):
        row = ds_split[i]
        prompt = row["prompt"]  # type: ignore
        subj, rel = parse_prompt(prompt)
        if rel not in by_rel_prompts:
            continue
        if len(by_rel_prompts[rel]) >= max_pairs_per_rel:
            continue
        by_rel_prompts[rel].append(prompt)
        by_rel_subjs[rel].append(subj)

    Ms = {}
    losses: List[float] = []
    
    # Process each relation
    for rel in relations:
        if not by_rel_prompts[rel]:
            continue
        
        prompts = by_rel_prompts[rel]
        weight = None
        bias = None

        samples = 0

        random.shuffle(prompts)

        for prompt in prompts[:train_size]:
            inputs = mt.tokenizer(prompt, return_tensors="pt").to(mt.model.device)
            # Precompute everything up to the subject, if there is anything before it.
            past_key_values = None
            input_ids = inputs.input_ids
            _h_index = -2
            _z_index = -1
            if _h_index > 0:
                outputs = mt.model(input_ids=input_ids[:, :_h_index], use_cache=True)
                past_key_values = outputs.past_key_values
                input_ids = input_ids[:, _h_index:]
                _h_index = 0
            use_cache = past_key_values is not None

            # Precompute initial h and z.
            [h_layer_name, z_layer_name] = determine_layer_paths(mt.model, [h_layer, -1])

            with baukit.TraceDict(
                mt.model, layers=(h_layer_name, z_layer_name), edit_output=None
            ) as ret:
                outputs = mt.model(
                    input_ids=input_ids,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                )
            h = untuple(ret[h_layer_name].output)[0, _h_index]
            z = untuple(ret[z_layer_name].output)[0, -1]

            # Now compute J and b.
            def compute_z_from_h(h: torch.Tensor) -> torch.Tensor:
                def insert_h(output: tuple, layer: str) -> tuple:
                    hs = untuple(output)
                    if layer != h_layer_name:
                        return output
                    hs[0, _h_index] = h
                    return output

                with baukit.TraceDict(
                    mt.model, (h_layer_name, z_layer_name), edit_output=insert_h
                ) as ret:
                    mt.model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )
                return untuple(ret[z_layer_name].output)[0, -1]

            if weight is None:
                weight = torch.autograd.functional.jacobian(compute_z_from_h, h, vectorize=True)
                bias = z[None] - h[None].mm(weight.t())
            else:
                weight += torch.autograd.functional.jacobian(compute_z_from_h, h, vectorize=True)
                bias += z[None] - h[None].mm(weight.t())
            torch.cuda.empty_cache()
            samples += 1
        weight /= samples
        bias /= samples
        Ms[rel] = (weight.cpu().detach(), bias.cpu().detach())
    return Ms


def ridge_regression_operator(X: torch.Tensor, Y: torch.Tensor, l2: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Y \in R^{N, D} and X \in R^{N, D}
    The equation I want o solve is Y = X M + b
    Solve M, b in  Y ≈ X M + b  (row-wise ridge with bias).
    X,Y: (N, D). Returns
        M: (D, D)  weight matrix
        b: (D,)    bias vector
    The bias term is not L2-regularised.
    """
    # X : (N, D) , Y : (N, D)
    N, D = X.shape

    # Add bias column (not regularised)
    X_aug = torch.cat([X, torch.ones(N, 1, dtype=X.dtype, device=X.device)], dim=1)  # (N, D+1)

    # Regularisation matrix: l2 on M part, 0 on bias part
    reg = torch.eye(D + 1, dtype=X.dtype, device=X.device)
    reg[-1, -1] = 0.0                                           # no L2 on bias
    reg *= l2

    # Closed-form ridge solution
    A = X_aug.T @ X_aug + reg                                   # (D+1, D+1)
    B = X_aug.T @ Y                                             # (D+1, D)
    W = torch.linalg.solve(A, B)                                # (D+1, D)

    M = W[:-1, :]                                               # (D, D)
    b = W[-1, :].view(-1)                                       # (D,)

    return M, b


def fit_lre_crosslayer(
    mt: ModelAndTokenizer,
    ds_split,
    h_layer: int,
    l2: float = 1e-2,
    relations: Iterable[str] = ("husband", "wife", "father", "mother", "sister", "brother", "son", "daughter"),
    max_pairs_per_rel: int = 8000,
    batch_size: int = 512,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Fit M_r mapping subject h at layer=h_layer to last-layer z for the same prompt.
    """
    by_rel_prompts: Dict[str, List[str]] = {r: [] for r in relations}
    by_rel_subjs: Dict[str, List[str]] = {r: [] for r in relations}
    
    for i in range(len(ds_split)):
        row = ds_split[i]
        prompt = row["prompt"]  # type: ignore
        subj, rel = parse_prompt(prompt)
        if rel not in by_rel_prompts:
            continue
        if len(by_rel_prompts[rel]) >= max_pairs_per_rel:
            continue
        by_rel_prompts[rel].append(prompt)
        by_rel_subjs[rel].append(subj)

    by_rel_X: Dict[str, List[torch.Tensor]] = {r: [] for r in relations}
    by_rel_Y: Dict[str, List[torch.Tensor]] = {r: [] for r in relations}

    for rel in relations:
        if not by_rel_prompts[rel]:
            continue
        
        prompts = by_rel_prompts[rel]
        subjs = by_rel_subjs[rel]
        
        for i in range(0, len(prompts), batch_size):
            prompt_batch = prompts[i:i+batch_size]
            subj_batch = subjs[i:i+batch_size]
            try:
                h, z, _ = compute_hz_pair_batch(mt, prompt_batch, subj_batch, h_layer)
                by_rel_X[rel].append(h)
                by_rel_Y[rel].append(z)
            except Exception as e:
                print(f"Skipping batch for relation {rel} due to error: {e}")
                continue

    Ms = {}
    losses: List[float] = []
    
    for rel in relations:
        if not by_rel_X[rel]:
            continue
        X = torch.cat(by_rel_X[rel], dim=0)
        Y = torch.cat(by_rel_Y[rel], dim=0)

        M, b = ridge_regression_operator(X, Y, l2=l2)
        Ms[rel] = (M.cpu().detach().T, b.cpu().detach())
    return Ms


def learn_lre_crosslayer(
    mt: ModelAndTokenizer,
    ds_split,
    h_layer: int,
    l2: float = 1e-2,
    relations: Iterable[str] = ("husband", "wife", "father", "mother", "sister", "brother", "son", "daughter"),
    max_pairs_per_rel: int = 8000,
    adam_lr: float = 1e-3,
    adam_epochs: int = 1000,
    batch_size: int = 512,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Fit M_r mapping subject h at layer=h_layer to last-layer z for the same prompt.
    """
    by_rel_prompts: Dict[str, List[str]] = {r: [] for r in relations}
    by_rel_subjs: Dict[str, List[str]] = {r: [] for r in relations}
    
    for i in range(len(ds_split)):
        row = ds_split[i]
        prompt = row["prompt"]  # type: ignore
        subj, rel = parse_prompt(prompt)
        if rel not in by_rel_prompts:
            continue
        if len(by_rel_prompts[rel]) >= max_pairs_per_rel:
            continue
        by_rel_prompts[rel].append(prompt)
        by_rel_subjs[rel].append(subj)

    by_rel_X: Dict[str, List[torch.Tensor]] = {r: [] for r in relations}
    by_rel_Y: Dict[str, List[torch.Tensor]] = {r: [] for r in relations}

    for rel in relations:
        if not by_rel_prompts[rel]:
            continue
        
        prompts = by_rel_prompts[rel]
        subjs = by_rel_subjs[rel]
        
        for i in range(0, len(prompts), batch_size):
            prompt_batch = prompts[i:i+batch_size]
            subj_batch = subjs[i:i+batch_size]
            try:
                h, z, _ = compute_hz_pair_batch(mt, prompt_batch, subj_batch, h_layer)
                by_rel_X[rel].append(h)
                by_rel_Y[rel].append(z)
            except Exception as e:
                print(f"Skipping batch for relation {rel} due to error: {e}")
                continue

    Ms = {}
    losses: List[float] = []
    
    for rel in relations:
        if not by_rel_X[rel]:
            continue
        X = torch.cat(by_rel_X[rel], dim=0)
        Y = torch.cat(by_rel_Y[rel], dim=0)
        # Normalize X and Y
        X_mean = X.mean(dim=0, keepdim=True)
        X_std = X.std(dim=0, keepdim=True) + 1e-8  # Add small epsilon to avoid division by zero
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std
        
        D = X.shape[1]
        M = torch.randn(D, D, device='cuda') * math.sqrt(2.0 / D)
        M.requires_grad_(True)
        
        # Initialize with bias term
        bias = torch.zeros(D, device='cuda')
        bias.requires_grad_(True)
        optimizer = torch.optim.AdamW([M, bias], lr=adam_lr, weight_decay=l2)

        final_loss = 0.0
        sgd_batch_size = 128

        # Select a random minibatch (single batch SGD) if batch_size < total samples
        N = X.shape[0]
        for epoch in range(adam_epochs):
            idx = torch.randperm(N)
            for start in range(0, N, sgd_batch_size):
                end = min(start + sgd_batch_size, N)
                X_batch = X[idx[start:end]].cuda()
                Y_batch = Y[idx[start:end]].cuda()
                optimizer.zero_grad()
                Y_pred = X_batch @ M.T + bias[None, ...]
                loss = torch.nn.functional.mse_loss(Y_pred, Y_batch)
                loss.backward()
                optimizer.step()
                final_loss = loss.cpu().item()
        
        Ms[rel] = (M.cpu().detach(), bias.cpu().detach(), X_mean.cpu().detach(), X_std.cpu().detach())
        losses.append(final_loss)
    
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return Ms, avg_loss


def fit_word2vec_crosslayer(
    mt: ModelAndTokenizer,
    ds_split,
    h_layer: int,
    relations: Iterable[str] = ("husband", "wife", "father", "mother", "sister", "brother", "son", "daughter"),
    max_pairs_per_rel: int = 8000,
    batch_size: int = 512,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Fit M_r mapping subject h at layer=h_layer to last-layer z for the same prompt.
    """
    by_rel_prompts: Dict[str, List[str]] = {r: [] for r in relations}
    by_rel_subjs: Dict[str, List[str]] = {r: [] for r in relations}
    
    for i in range(len(ds_split)):
        row = ds_split[i]
        prompt = row["prompt"]  # type: ignore
        subj, rel = parse_prompt(prompt)
        if rel not in by_rel_prompts:
            continue
        if len(by_rel_prompts[rel]) >= max_pairs_per_rel:
            continue
        by_rel_prompts[rel].append(prompt)
        by_rel_subjs[rel].append(subj)

    by_rel_X: Dict[str, List[torch.Tensor]] = {r: [] for r in relations}
    by_rel_Y: Dict[str, List[torch.Tensor]] = {r: [] for r in relations}

    for rel in relations:
        if not by_rel_prompts[rel]:
            continue
        
        prompts = by_rel_prompts[rel]
        subjs = by_rel_subjs[rel]
        
        for i in range(0, len(prompts), batch_size):
            prompt_batch = prompts[i:i+batch_size]
            subj_batch = subjs[i:i+batch_size]
            try:
                h, z, _ = compute_hz_pair_batch(mt, prompt_batch, subj_batch, h_layer)
                by_rel_X[rel].append(h)
                by_rel_Y[rel].append(z)
            except Exception as e:
                print(f"Skipping batch for relation {rel} due to error: {e}")
                continue

    Ms: Dict[str, torch.Tensor] = {}
    losses: List[float] = []
    
    for rel in relations:
        if not by_rel_X[rel]:
            continue
        X = torch.cat(by_rel_X[rel], dim=0)
        Y = torch.cat(by_rel_Y[rel], dim=0)
        bias = (Y - X).mean(dim=0, keepdim=True)
        Ms[rel] = bias.cpu().detach()
        # Calculate ridge regression loss
        Y_pred = X + bias
        ridge_loss = torch.nn.functional.mse_loss(Y_pred, Y).item()
        losses.append(ridge_loss)
    
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return Ms, avg_loss

@torch.no_grad()
def lre_ridge_top1_accuracy(
    mt: ModelAndTokenizer,
    Ms: Dict[str, torch.Tensor],
    ds_eval,
    h_layer: int,
    beta: float = 1.0,
    max_eval: int = 4500,
    batch_size: int = 32,
) -> float:
    """
    Evaluate top-1 accuracy using z_hat = M_r @ h_subj at last layer, then LM head logits.
    Compares decoded top-1 token string to provided 'answer'.
    """
    tok = mt.tokenizer
    model = mt.model
    device = model.device
    correct = 0
    total = 0

    prompts, subjs, rels, golds = [], [], [], []
    for i in range(min(len(ds_eval), max_eval)):
        row = ds_eval[i]
        prompt = row["prompt"]  # type: ignore
        answers = row.get("answer", [""])  # type: ignore
        if not answers:
            continue
        gold = answers[0]
        subj, rel = parse_prompt(prompt)
        if rel not in Ms:
            continue
        prompts.append(prompt)
        subjs.append(subj)
        rels.append(rel)
        golds.append(gold)

    for i in range(0, len(prompts), batch_size):
        prompt_batch = prompts[i:i+batch_size]
        subj_batch = subjs[i:i+batch_size]
        rel_batch = rels[i:i+batch_size]
        gold_batch = golds[i:i+batch_size]
        h, _, _ = compute_hz_pair_batch(mt, prompt_batch, subj_batch, h_layer)
        
        M, b = Ms[rel_batch[0]]  # all rels in batch are the same size-wise
        M, b = M.to(device), b.to(device)

        z = h.to(device).mm(M.t())
        z = beta * z + b                                           # use bias
        z = model.gpt_neox.final_layer_norm(z)
        logits = model.get_output_embeddings()(z)
        pred_ids = torch.argmax(logits, dim=-1)
        preds = tok.batch_decode(pred_ids, skip_special_tokens=True)
        
        for j in range(len(preds)):
            correct += int(preds[j].strip() in gold_batch[j])
        total += len(preds)

    return correct / max(1, total)


@torch.no_grad()
def lre_top1_accuracy(
    mt: ModelAndTokenizer,
    Ms: Dict[str, torch.Tensor],
    ds_eval,
    h_layer: int,
    betas: List[float] = (0.5, 1.0, 2.0, 4.0),
) -> Dict[float, Dict[str, float]]:
    """
    Return accuracies for each β in `betas`.
    accuracy[beta]["overall"]  : overall top-1 accuracy
    accuracy[beta][relation]   : accuracy for that relation
    """
    device = mt.model.device
    # collect evaluable examples first
    prompts, subjs, rels, golds = [], [], [], []
    for i in range(len(ds_eval)):
        row = ds_eval[i]
        prompt = row["prompt"]          # type: ignore
        answers = row.get("answer", []) # type: ignore
        if not answers:
            continue
        subj, rel = parse_prompt(prompt)
        if rel not in Ms:
            continue
        prompts.append(prompt)
        subjs.append(subj)
        rels.append(rel)
        golds.append(answers[0])

    # init result containers
    acc = {beta: {"overall_correct": 0, "overall_total": 0} for beta in betas}
    for beta in betas:
        for r in Ms.keys():
            acc[beta][r] = {"correct": 0, "total": 0}

    # iterate through each example (individual-step to avoid VRAM blow-up with TraceDict)
    for i in range(len(prompts)):
        prompt, subj, rel, gold = prompts[i], subjs[i], rels[i], golds[i]

        inputs = mt.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        _h_index = -2  # subject last token
        if _h_index > 0:  # never happens here because we always use -2
            outputs = mt.model(input_ids=input_ids[:, :_h_index], use_cache=True)
            past_key_values = outputs.past_key_values
            input_ids = input_ids[:, _h_index:]
            _h_index = 0
        else:
            past_key_values = None
        use_cache = past_key_values is not None
        h_layer_name, z_layer_name = determine_layer_paths(mt.model, [h_layer, -1])

        with baukit.TraceDict(mt.model, (h_layer_name, z_layer_name), edit_output=None) as ret:
            mt.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        h = untuple(ret[h_layer_name].output)[0, _h_index]           # (D,)
        M, b = Ms[rel]
        M, b = M.to(device), b.to(device)
        z_base = h @ M.t()                                           # (D,)

        # evaluate every beta
        for beta in betas:
            z = beta * z_base + b                                    # (D,)
            z = mt.model.gpt_neox.final_layer_norm(z)
            logits = mt.model.get_output_embeddings()(z)                           # (V,)
            pred_id = torch.argmax(logits)
            pred_str = mt.tokenizer.decode(pred_id, skip_special_tokens=True).strip()

            hit = int(pred_str in gold)
            acc[beta]["overall_correct"] += hit
            acc[beta]["overall_total"] += 1
            acc[beta][rel]["correct"] += hit
            acc[beta][rel]["total"] += 1
        torch.cuda.empty_cache()

    # convert counts to accuracies
    accuracy_by_beta: Dict[float, Dict[str, float]] = {}
    for beta in betas:
        beta_dict: Dict[str, float] = {}
        # per-relation
        for rel in Ms.keys():
            corr = acc[beta][rel]["correct"]
            tot = acc[beta][rel]["total"]
            if tot > 0:
                beta_dict[rel] = corr / tot
        # overall
        overall_corr = acc[beta]["overall_correct"]
        overall_tot = acc[beta]["overall_total"]
        beta_dict["overall"] = overall_corr / max(1, overall_tot)
        accuracy_by_beta[beta] = beta_dict

    return accuracy_by_beta

@torch.no_grad()
def word2vec_top1_accuracy(
    mt: ModelAndTokenizer,
    Ms: Dict[str, torch.Tensor],
    ds_eval,
    h_layer: int,
    max_eval: int = 2000,
    batch_size: int = 32,
) -> float:
    """
    Evaluate top-1 accuracy using z_hat = M_r @ h_subj at last layer, then LM head logits.
    Compares decoded top-1 token string to provided 'answer'.
    """
    tok = mt.tokenizer
    model = mt.model
    device = model.device
    correct = 0
    total = 0

    prompts, subjs, rels, golds = [], [], [], []
    for i in range(min(len(ds_eval), max_eval)):
        row = ds_eval[i]
        prompt = row["prompt"]  # type: ignore
        answers = row.get("answer", [""])  # type: ignore
        if not answers:
            continue
        gold = answers[0]
        subj, rel = parse_prompt(prompt)
        if rel not in Ms:
            continue
        prompts.append(prompt)
        subjs.append(subj)
        rels.append(rel)
        golds.append(gold)

    for i in range(0, len(prompts), batch_size):
        prompt_batch = prompts[i:i+batch_size]
        subj_batch = subjs[i:i+batch_size]
        rel_batch = rels[i:i+batch_size]
        gold_batch = golds[i:i+batch_size]
        h, _, _ = compute_hz_pair_batch(mt, prompt_batch, subj_batch, h_layer)
        
        z_hats = []
        for j in range(h.shape[0]):
            rel = rel_batch[j]
            b = Ms[rel]
            z_hat = (h[j] + b).to(device)
            z_hats.append(z_hat)
        
        z_hat_batch = torch.cat(z_hats, dim=0)
        z_hat_batch = model.gpt_neox.final_layer_norm(z_hat_batch)
        logits = model.get_output_embeddings()(z_hat_batch)  # type: ignore
        pred_ids = torch.argmax(logits, dim=-1)
        preds = mt.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        for j in range(len(preds)):
            correct += int(preds[j].strip() in gold_batch[j])
        total += len(preds)

    return correct / max(1, total)