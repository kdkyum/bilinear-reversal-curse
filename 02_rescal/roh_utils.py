import math
import re
import random
import json

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from misc import ModelAndTokenizer, parse_prompt


def ridge_regression_operator(X: torch.Tensor, Y: torch.Tensor, l2: float = 1e-2) -> torch.Tensor:
    """
    Solve M in Y ≈ X M^T (row-wise), i.e., M maps X -> Y.
    Here X,Y: (N, D). We return M: (D, D) such that Y_hat = X @ M^T.
    """
    # Closed-form ridge: M^T = (X^T X + l2 I)^(-1) X^T Y
    D = X.shape[1]
    XtX = X.T @ X
    reg = l2 * torch.eye(D, dtype=X.dtype)
    M_t = torch.linalg.solve(XtX + reg, X.T @ Y)
    return M_t.T  # (D, D)


def fit_relation_operators(
    ds_reverse_bi,
    v: Dict[str, torch.Tensor],
    l2: float = 1e-2,
    relations: Iterable[str] = ("husband", "wife", "father", "mother", "sister", "brother", "son", "daughter"),
    max_pairs_per_rel: int = -1,
    use_adam: bool = False,
    adam_lr: float = 1e-3,
    adam_epochs: int = 50,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Fit M_r via least squares on pairs (subject, object) from reverse_bi split.
    We use v(subject) -> v(object), aggregating across the dataset.
    """
    # Gather training matrices per relation
    by_rel: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {r: [] for r in relations}
    for i in range(len(ds_reverse_bi)):
        row = ds_reverse_bi[i]
        prompt = row["prompt"]  # type: ignore
        answer_list = row.get("answer", [])  # type: ignore
        subj, rel = parse_prompt(prompt)
        if rel not in by_rel:
            continue
        obj = answer_list[0]
        if i == 0:
            print(f"Example parsed: subj='{subj}', rel='{rel}', obj='{obj}'")
        if subj in v and f" {obj}" in v:
            by_rel[rel].append((v[subj], v[f" {obj}"]))

    Ms: Dict[str, torch.Tensor] = {}
    losses: List[float] = []
    
    for rel, pairs in by_rel.items():
        if not pairs:
            continue
        pairs = pairs[:max_pairs_per_rel]
        X = torch.stack([a for a, _ in pairs], dim=0)
        Y = torch.stack([b for _, b in pairs], dim=0)
        
        if use_adam:
            # Use Adam optimizer to learn the operator
            D = X.shape[1]
            M = torch.randn(D, D, requires_grad=True)
            optimizer = torch.optim.Adam([M], lr=adam_lr, weight_decay=l2)
            
            final_loss = 0.0
            for epoch in range(adam_epochs):
                optimizer.zero_grad()
                Y_pred = X @ M.T
                loss = torch.nn.functional.mse_loss(Y_pred, Y)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
            
            Ms[rel] = M.detach()
            losses.append(final_loss)
        else:
            M = ridge_regression_operator(X, Y, l2=l2)
            Ms[rel] = M
            # Calculate ridge regression loss
            Y_pred = X @ M.T
            ridge_loss = torch.nn.functional.mse_loss(Y_pred, Y).item()
            losses.append(ridge_loss)
    
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return Ms, avg_loss

# ---------------------------
# RESCAL: Three way tensor factorization
# ---------------------------

def fit_rescal_operators(
    ds_reverse_bi,
    v: Dict[str, torch.Tensor],
    l2: float = 0.1,
    relations: Iterable[str] = ("husband", "wife", "father", "mother", "sister", "brother", "son", "daughter"),
    max_pairs_per_rel: int = -1,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Fit RESCAL relation operators R_r using the closed-form solution with Kronecker product optimization.
    """
    # 1. Build fixed A matrix and entity-to-index mapping
    entity_names = list(v.keys())
    entity_to_idx = {name: i for i, name in enumerate(entity_names)}
    n_entities = len(entity_names)
    
    # Stack all entity vectors to form A matrix
    A = torch.stack([v[name] for name in entity_names], dim=0)  # (n_entities, embedding_dim)
    embedding_dim = A.shape[1]
    
    # 2. Build relation-specific training pairs
    by_rel: Dict[str, List[Tuple[int, int]]] = {r: [] for r in relations}
    
    for i in range(len(ds_reverse_bi)):
        row = ds_reverse_bi[i]
        prompt = row["prompt"]
        answer_list = row.get("answer", [])
        subj_name, rel = parse_prompt(prompt)
        
        if rel not in by_rel or not answer_list:
            continue
            
        obj_name_raw = answer_list[0]
        # Match v dictionary key format
        obj_name = f" {obj_name_raw}" if f" {obj_name_raw}" in v else obj_name_raw

        if subj_name in entity_to_idx and obj_name in entity_to_idx:
            subj_idx = entity_to_idx[subj_name]
            obj_idx = entity_to_idx[obj_name]
            by_rel[rel].append((subj_idx, obj_idx))
    
    # 4. Learn R_k for each relation using closed-form solution
    Rs: Dict[str, torch.Tensor] = {}
    losses: List[float] = []
    
    for rel, pairs in by_rel.items():
        if not pairs:
            continue
        if max_pairs_per_rel != -1:
            pairs = pairs[:max_pairs_per_rel]

        # Build sparse adjacency matrix X_r
        X_r = torch.zeros((n_entities, n_entities), dtype=A.dtype, device=A.device)
        subj_indices, obj_indices = zip(*pairs)
        X_r[torch.tensor(subj_indices), torch.tensor(obj_indices)] = 1.0

        R_k = update_Mr_exact_svd(A, X_r, l2)
        Rs[rel] = R_k.T
        
        # Calculate reconstruction loss
        with torch.no_grad():
            X_r_hat = A @ R_k @ A.T
            loss = torch.nn.functional.mse_loss(X_r_hat, X_r).item()
            losses.append(loss)
    
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return Rs, avg_loss


def update_Mr_exact_svd(A: torch.Tensor, X_r: torch.Tensor, lambda_R: float) -> torch.Tensor:
    """
    Computes the exact update for one relation matrix M_r in RESCAL using SVD,
    implemented with PyTorch tensors.

    Args:
        A (torch.Tensor): Entity embedding matrix of shape (n, d).
        X_r (torch.Tensor): (Sparse) adjacency matrix for relation r, shape (n, n).
        lambda_R (float): l2-regularization strength.

    Returns:
        torch.Tensor: Updated relation matrix M_r of shape (d, d).
    """
    device = A.device
    dtype = A.dtype
    # 1. SVD of A  A = U diag(s) Vᵀ
    #    U:(n,d), s:(d,), Vh:(d,d)  (Vh = Vᵀ)
    U, s_vec, Vh = torch.linalg.svd(A, full_matrices=False)
    V = Vh.transpose(-2, -1)       # (d, d)

    # 2. Singular values of A⊗A  outer product of s with itself
    s_outer = torch.outer(s_vec, s_vec)  # (d, d)

    # 3. Ridge filter
    s_filtered = s_outer / (s_outer.pow(2) + lambda_R)

    # 4. Transform X_r into SVD space: C = Uᵀ X_r U
    C = U.transpose(0, 1) @ X_r @ U     # (d, d)

    # 5. Element-wise filter
    C_filtered = s_filtered * C         # (d, d)

    # 6. Transform back: M_r = V C_filtered Vᵀ
    M_r = V @ C_filtered @ V.transpose(0, 1)  # (d, d)

    return M_r


# ---------------------------
# Fit Translation Operators
# ---------------------------

def fit_translational_operators(
    ds_reverse_bi,
    v: Dict[str, torch.Tensor],
    relations: Iterable[str] = ("husband", "wife", "father", "mother", "sister", "brother", "son", "daughter"),
    max_pairs_per_rel: int = -1,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Fit M_r via least squares on pairs (subject, object) from reverse_bi split.
    We use v(subject) -> v(object), aggregating across the dataset.
    """
    # Gather training matrices per relation
    by_rel: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {r: [] for r in relations}
    for i in range(len(ds_reverse_bi)):
        row = ds_reverse_bi[i]
        prompt = row["prompt"]  # type: ignore
        answer_list = row.get("answer", [])  # type: ignore
        subj, rel = parse_prompt(prompt)
        if rel not in by_rel:
            continue
        obj = answer_list[0]
        # print(subj, obj)
        if subj in v and f" {obj}" in v:
            by_rel[rel].append((v[subj], v[f" {obj}"]))

    bs: Dict[str, torch.Tensor] = {}
    losses: List[float] = []
    
    for rel, pairs in by_rel.items():
        if not pairs:
            continue
        pairs = pairs[:max_pairs_per_rel]
        X = torch.stack([a for a, _ in pairs], dim=0)
        Y = torch.stack([b for _, b in pairs], dim=0)

        b = (Y - X).mean(dim=0, keepdim=True)
        bs[rel] = b
        # Calculate ridge regression loss
        Y_pred = X + b
        ridge_loss = torch.nn.functional.mse_loss(Y_pred, Y).item()
        losses.append(ridge_loss)
    
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return bs, avg_loss


def build_entity_vectors(
    mt: ModelAndTokenizer,
    ds_reverse_bi,
    max_entities: int = 5000,
    min_obs: int = 1,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Aggregate entity vectors v(e) by averaging the hidden state at the subject position
    across different prompts where e appears as subject.
    Returns: Dict[layer_idx, Dict[subject, representation]]
    """
    tok = mt.tokenizer
    device = mt.model.device

    seen_subjects: set = set()
    for i in range(len(ds_reverse_bi)):
        row = ds_reverse_bi[i]
        prompt = row["prompt"]  # type: ignore
        subj, _ = parse_prompt(prompt)
        if subj in seen_subjects:
            continue
        seen_subjects.add(subj)

    # Select up to max_entities 
    pool = list(seen_subjects)[:max_entities]

    # Check if this is GPT-NeoX style model
    is_neox = True #hasattr(mt.model, 'gpt_neox')
    num_layers = mt.model.config.num_hidden_layers + 1

    # Initialize output dict for all layers
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer_idx in range(num_layers):
        out[layer_idx] = {}

    # Process in batches for efficiency
    batch_size = 128
    batch_subjects = []
    
    original_padding_side = tok.padding_side
    tok.padding_side = "left"
    
    for i in range(len(ds_reverse_bi)):
        row = ds_reverse_bi[i]
        prompt = row["prompt"]  # type: ignore
        subj, _ = parse_prompt(prompt)
        if subj not in pool:
            continue
        if subj in out[0]:  # Check if already processed (using layer 0 as reference)
            continue
            
        batch_subjects.append(subj)
        # Mark as being processed for all layers
        for layer_idx in range(num_layers):
            out[layer_idx][subj] = None

        if len(batch_subjects) >= batch_size:
            # Process batch
            inputs = tok(batch_subjects, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
            
            if is_neox:
                # Use HiddenStateExtractor for GPT-NeoX models
                extractor = HiddenStateExtractor(mt.model, mt.tokenizer)
                for bi, subj in enumerate(batch_subjects):
                    # Extract for each subject individually (position -1 for last token)
                    outputs = extractor.extract_from_prompt(subj, pos=-1)
                    
                    # Store representations for all layers
                    for layer_idx in range(num_layers):
                        # For GPT-NeoX, if we're at the last layer, use residual_pre_mlp + mlp_out
                        if layer_idx == num_layers - 1:
                            vec = outputs["residual_pre_mlp"][-1] + outputs["mlp_out"][-1]
                        else:
                            vec = outputs["residual_pre_attn"][layer_idx]
                        
                        out[layer_idx][subj] = vec
            else:
                # Use output_hidden_states for GPT-2 style models
                with torch.no_grad():
                    all_hs = mt.model(**inputs, output_hidden_states=True).hidden_states  # type: ignore
                
                for bi, subj in enumerate(batch_subjects):
                    # Store representations for all layers
                    for layer_idx in range(num_layers):
                        vec = all_hs[layer_idx][bi, -1, :].detach().cpu()
                        out[layer_idx][subj] = vec
            
            batch_subjects.clear()

    # Process remaining subjects
    if batch_subjects:
        inputs = tok(batch_subjects, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        
        if is_neox:
            # Use HiddenStateExtractor for GPT-NeoX models
            extractor = HiddenStateExtractor(mt.model, mt.tokenizer)
            for bi, subj in enumerate(batch_subjects):
                outputs = extractor.extract_from_prompt(subj, pos=-1)
                
                # Store representations for all layers
                for layer_idx in range(num_layers):
                    if layer_idx == num_layers - 1:
                        vec = outputs["residual_pre_mlp"][-1] + outputs["mlp_out"][-1]
                    else:
                        vec = outputs["residual_pre_attn"][layer_idx]
                    
                    out[layer_idx][subj] = vec
        else:
            # Use output_hidden_states for GPT-2 style models
            with torch.no_grad():
                all_hs = mt.model(**inputs, output_hidden_states=True).hidden_states  # type: ignore
            
            for bi, subj in enumerate(batch_subjects):
                # Store representations for all layers
                for layer_idx in range(num_layers):
                    vec = all_hs[layer_idx][bi, -1, :].detach().cpu()
                    out[layer_idx][subj] = vec

    tok.padding_side = original_padding_side
    return out


from functools import partial

class HiddenStateExtractor:
    """Extract {residual_pre_attn, attn_out, mlp_out} for a given token position, per layer."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @staticmethod
    def _token_at_pos(x: torch.Tensor, pos: int) -> torch.Tensor:
        # x: (batch, seq, hidden)
        return x[0, pos, :].detach().to("cpu")

    def extract_from_prompt(self, prompt: str, pos: int):
        """Extract residual components for an arbitrary prompt string (features at `pos` token per layer)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

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
            "prompt": prompt,
        }


@torch.no_grad()
def bilinear_top1_accuracy(
    Ms: Dict[str, torch.Tensor],
    v: Dict[str, torch.Tensor],
    eval_pairs: List[Tuple[str, str, str]],  # (subject, relation, true_object)
    k_candidates: int = 0,
) -> Dict[str, float]:
    """
    Predict object via argmax_e < M_r v(s), v(e) > and compute top-1 accuracy.
    If k_candidates > 0, restrict candidate set to k random entities + true.
    Returns accuracy for each relation.
    """
    names = list(v.keys())
    correct_by_rel = {}
    total_by_rel = {}
    
    for s, r, o in eval_pairs:
        if r not in Ms or s not in v or o not in v:
            continue
        
        if r not in correct_by_rel:
            correct_by_rel[r] = 0
            total_by_rel[r] = 0
            
        z = (Ms[r] @ v[s]).unsqueeze(0) # (1, D)
        if k_candidates and k_candidates < len(names):
            import random
            cand = set(random.sample(names, k_candidates)) | {o}
            cand = list(cand)
        else:
            cand = names
        Vc = torch.stack([v[n] for n in cand], dim=0)  # (C, D)
        scores = (z @ Vc.T).squeeze(0)
        pred = cand[int(torch.argmax(scores).item())]
        # print(pred, o)
        correct_by_rel[r] += int(pred == o)
        total_by_rel[r] += 1
    
    # Calculate accuracy per relation
    accuracy_by_rel = {}
    for rel in correct_by_rel:
        accuracy_by_rel[rel] = correct_by_rel[rel] / max(1, total_by_rel[rel])
    
    return accuracy_by_rel


@torch.no_grad()
def compositional_top1_accuracy(
    Ms: Dict[str, torch.Tensor],
    v: Dict[str, torch.Tensor],
    eval_compositions: List[Tuple[str, Tuple[str, ...], str]],  # (subject, (rel1, rel2, ...), object)
    k_candidates: int = 0,
) -> float:
    """
    Evaluate top-1 accuracy for composed relations.
    e.g. M_father ≈ M_husband @ M_mother.
    We test if (M_husband @ M_mother) @ v(subject) -> v(object).
    """
    names = list(v.keys())
    correct = 0
    total = 0

    for s, comp_rels, o in eval_compositions:
        # Check if all required entities and operators are available
        if s not in v or o not in v:# or not all(r in Ms for r in comp_rels):
            continue

        # Compute composed operator
        M_comp = torch.eye(list(Ms.values())[0].shape[0], dtype=list(Ms.values())[0].dtype)
        for r in comp_rels:  # Apply in order: M_r2 @ M_r1 for (r1, r2)
            M_comp = Ms[r] @ M_comp

        z = (M_comp @ v[s]).unsqueeze(0)  # (1, D)

        # Select candidates for prediction
        if k_candidates and k_candidates < len(names):
            import random
            cand = set(random.sample(names, k_candidates)) | {o}
            cand = list(cand)
        else:
            cand = names

        Vc = torch.stack([v[n] for n in cand], dim=0)  # (C, D)
        scores = (z @ Vc.T).squeeze(0)
        pred = cand[int(torch.argmax(scores).item())]

        correct += int(pred == o)
        total += 1
    return correct / max(1, total)


def transpose_top1_accuracy(Ms: Dict[str, torch.Tensor],
    v: Dict[str, torch.Tensor],
    eval_pairs: List[Tuple[str, str, str]],  # (subject, relation, true_object)
    k_candidates: int = 0,
) -> float:
    metrics = {}
    inverse_pairs = {
        "husband": "wife",
        "wife": "husband",
        "brother": "sister",
        "sister": "brother",
    }

    names = list(v.keys())
    correct_by_rel = {}
    total_by_rel = {}

    # s husband o;
    # o husband^T s ?
    
    for s, r, o in eval_pairs:
        if r not in Ms or s not in v or o not in v or r not in inverse_pairs.keys():
            continue
        
        if r not in correct_by_rel:
            correct_by_rel[r] = 0
            total_by_rel[r] = 0
        inv_r = inverse_pairs[r]
        z = (Ms[r].T @ v[o]).unsqueeze(0) # (1, D)
        if k_candidates and k_candidates < len(names):
            import random
            cand = set(random.sample(names, k_candidates)) | {o}
            cand = list(cand)
        else:
            cand = names
        Vc = torch.stack([v[n] for n in cand], dim=0)  # (C, D)
        scores = (z @ Vc.T).squeeze(0)
        pred = cand[int(torch.argmax(scores).item())]
        # print(pred, o)
        correct_by_rel[r] += int(pred == s)
        total_by_rel[r] += 1
    
    # Calculate accuracy per relation
    accuracy_by_rel = {}
    for rel in correct_by_rel:
        accuracy_by_rel[rel] = correct_by_rel[rel] / max(1, total_by_rel[rel])
    
    return accuracy_by_rel


@torch.no_grad()
def translational_top1_accuracy(
    bs: Dict[str, torch.Tensor],
    v: Dict[str, torch.Tensor],
    eval_pairs: List[Tuple[str, str, str]],  # (subject, relation, true_object)
    k_candidates: int = 0,
) -> Dict[str, float]:
    names = list(v.keys())
    correct_by_rel = {}
    total_by_rel = {}
    
    for s, r, o in eval_pairs:
        if r not in bs or s not in v or o not in v:
            continue
        
        if r not in correct_by_rel:
            correct_by_rel[r] = 0
            total_by_rel[r] = 0
            
        z = (v[s] + bs[r]).unsqueeze(0) # (1, D)
        if k_candidates and k_candidates < len(names):
            import random
            cand = set(random.sample(names, k_candidates)) | {o}
            cand = list(cand)
        else:
            cand = names
        Vc = torch.stack([v[n] for n in cand], dim=0)  # (C, D)
        scores = (z @ Vc.T).squeeze(0)
        pred = cand[int(torch.argmax(scores).item())]
        # print(pred, o)
        correct_by_rel[r] += int(pred == o)
        total_by_rel[r] += 1
    
    # Calculate accuracy per relation
    accuracy_by_rel = {}
    for rel in correct_by_rel:
        accuracy_by_rel[rel] = correct_by_rel[rel] / max(1, total_by_rel[rel])
    
    return accuracy_by_rel


def negation_top1_accuracy(bs: Dict[str, torch.Tensor],
    v: Dict[str, torch.Tensor],
    eval_pairs: List[Tuple[str, str, str]],  # (subject, relation, true_object)
    k_candidates: int = 0,
) -> float:
    metrics = {}
    inverse_pairs = {
        "husband": "wife",
        "wife": "husband",
        "brother": "sister",
        "sister": "brother",
    }

    names = list(v.keys())
    correct_by_rel = {}
    total_by_rel = {}
    
    for s, r, o in eval_pairs:
        if r not in bs or s not in v or o not in v or r not in inverse_pairs.keys():
            continue
        
        if r not in correct_by_rel:
            correct_by_rel[r] = 0
            total_by_rel[r] = 0
        inv_r = inverse_pairs[r]
        z = (v[o] - bs[r]).unsqueeze(0) # (1, D)
        if k_candidates and k_candidates < len(names):
            import random
            cand = set(random.sample(names, k_candidates)) | {o}
            cand = list(cand)
        else:
            cand = names
        Vc = torch.stack([v[n] for n in cand], dim=0)  # (C, D)
        scores = (z @ Vc.T).squeeze(0)
        pred = cand[int(torch.argmax(scores).item())]
        # print(pred, o)
        correct_by_rel[r] += int(pred == s)
        total_by_rel[r] += 1
    
    # Calculate accuracy per relation
    accuracy_by_rel = {}
    for rel in correct_by_rel:
        accuracy_by_rel[rel] = correct_by_rel[rel] / max(1, total_by_rel[rel])
    
    return accuracy_by_rel
