"""Train a GPT-NeoX language model from scratch on the family_relation dataset.

Replicates the training setup of "Bilinear representation mitigates reversal
curse and enables consistent model editing" (arXiv:2509.21993): a 12-layer
GPT-NeoX (~206M parameters) trained on a synthetic family knowledge graph,
sweeping weight decay and random seed.

Launch with torchrun (the paper uses 4 GPUs with a global batch size of 64):

    torchrun --standalone --nproc_per_node=4 train.py --weight_decay 3.0 --seed 0
"""

import argparse
import json
import math
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

DATA_REPO = "kdkyum/family_relation"
DATA_CONFIG = "lvl3_N1e+3"
TOKENIZER = "EleutherAI/gpt-neox-20b"
SEQ_LEN = 1024
WARMUP_RATIO = 0.01


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_dataset():
    """Download train/eval splits from the HuggingFace Hub."""
    def fetch(fname, key):
        path = hf_hub_download(DATA_REPO, f"{DATA_CONFIG}/{fname}", repo_type="dataset")
        with open(path) as f:
            return json.load(f)[key]

    return (
        fetch("train.json", "train"),
        fetch("eval_reverse_bi.json", "reverse_bi"),
        fetch("eval_reverse_uni.json", "reverse_uni"),
    )


def tokenize_and_chunk(tokenizer, texts):
    """Tokenize documents (BOS-prefixed), concatenate, and split into
    (SEQ_LEN + 1)-token chunks."""
    bos_id = tokenizer.bos_token_id
    parts = []
    batch = 10_000
    for i in range(0, len(texts), batch):
        buf = []
        for ids in tokenizer(texts[i:i + batch], add_special_tokens=False)["input_ids"]:
            buf.append(bos_id)
            buf.extend(ids)
        parts.append(torch.tensor(buf, dtype=torch.int32))
        print(f"  tokenized {min(i + batch, len(texts)):,}/{len(texts):,} docs", flush=True)

    flat = torch.cat(parts)
    block = SEQ_LEN + 1
    n_tokens = (flat.numel() // block) * block
    return flat[:n_tokens].view(-1, block)


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(vocab_size, bos_id):
    """GPT-NeoX architecture from Appendix A of the paper (~206M parameters)."""
    config = GPTNeoXConfig(
        vocab_size=vocab_size,
        hidden_size=896,
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=3584,
        hidden_act="gelu",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
        max_position_embeddings=SEQ_LEN,
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        use_parallel_residual=False,
        tie_word_embeddings=False,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        bos_token_id=bos_id,
        eos_token_id=bos_id,
        pad_token_id=bos_id,
    )
    return GPTNeoXForCausalLM(config)


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tokenizer, eval_data, device, max_answer_tokens=10, batch_size=64):
    """Greedy generation; a prediction counts as correct on exact match of the
    generated object name (up to the first period)."""
    model.eval()
    correct = 0
    bos_id = tokenizer.bos_token_id
    period_id = tokenizer.encode(".")[0]

    for start in range(0, len(eval_data), batch_size):
        items = eval_data[start:start + batch_size]
        prompts = [[bos_id] + tokenizer.encode(it["prompt"]) for it in items]
        max_len = max(len(p) for p in prompts)

        # Left-pad so generation starts from the last prompt token for all rows
        input_ids = torch.full((len(items), max_len), bos_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for i, p in enumerate(prompts):
            input_ids[i, max_len - len(p):] = torch.tensor(p, device=device)
            attention_mask[i, max_len - len(p):] = 1

        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_answer_tokens,
            do_sample=False,
        )

        for i, item in enumerate(items):
            answer_ids = []
            for tid in generated[i, max_len:].tolist():
                if tid in (period_id, bos_id):
                    break
                answer_ids.append(tid)
            predicted = tokenizer.decode(answer_ids).strip()
            answer = item["answer"]
            correct += predicted in answer if isinstance(answer, list) else predicted == answer

    return correct / len(eval_data)


def run_eval(model, tokenizer, eval_bi, eval_uni, device, step, log):
    bi_acc = evaluate(model, tokenizer, eval_bi, device)
    uni_acc = evaluate(model, tokenizer, eval_uni, device)
    print(f"  eval_reverse_bi (recall):        {bi_acc:.4f}")
    print(f"  eval_reverse_uni (reversal test): {uni_acc:.4f}", flush=True)
    log({"event": "eval", "step": step, "eval_reverse_bi_acc": bi_acc, "eval_reverse_uni_acc": uni_acc})
    return uni_acc


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64, help="global batch size across all GPUs")
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="runs/default")
    args = parser.parse_args()

    # Distributed setup (single-process if not launched with torchrun)
    ddp = "RANK" in os.environ
    if ddp:
        # Long timeout: non-main ranks wait while rank 0 tokenizes and evaluates
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=3))
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    assert args.batch_size % world_size == 0
    per_device_bs = args.batch_size // world_size
    torch.manual_seed(args.seed + rank)  # DDP broadcasts rank-0 weights at wrap time

    os.makedirs(args.out_dir, exist_ok=True)
    log_file = open(os.path.join(args.out_dir, "log.jsonl"), "w") if is_main else None

    def log(entry):
        if log_file:
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    bos_id = tokenizer.bos_token_id

    # Rank 0 downloads and tokenizes, then broadcasts the token chunks
    if is_main:
        print(f"Loading dataset {DATA_REPO}/{DATA_CONFIG} ...", flush=True)
        train_texts, eval_bi, eval_uni = load_dataset()
        chunks = tokenize_and_chunk(tokenizer, train_texts)
        del train_texts
    if ddp:
        shape = torch.tensor(chunks.shape if is_main else (0, 0), device=device)
        dist.broadcast(shape, src=0)
        buf = chunks.to(device) if is_main else torch.empty(*shape.tolist(), dtype=torch.int32, device=device)
        dist.broadcast(buf, src=0)
        chunks = buf.cpu()
        del buf
        torch.cuda.empty_cache()
    if is_main:
        print(f"Train: {chunks.shape[0]:,} chunks of {chunks.shape[1]} tokens "
              f"({chunks.numel() / 1e6:.0f}M tokens/epoch)")

    model = build_model(len(tokenizer), bos_id).to(device)
    raw_model = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    num_params = sum(p.numel() for p in raw_model.parameters())
    if is_main:
        print(f"Model parameters: {num_params:,}")
    log({"event": "config", **vars(args), "world_size": world_size, "num_params": num_params})

    # LayerNorm parameters are excluded from weight decay
    no_decay = {id(p) for m in raw_model.modules() if isinstance(m, torch.nn.LayerNorm) for p in m.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in raw_model.parameters() if id(p) not in no_decay],
             "weight_decay": args.weight_decay},
            {"params": [p for p in raw_model.parameters() if id(p) in no_decay],
             "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = chunks.shape[0] // world_size // per_device_bs
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if is_main:
        print(f"Training: {args.epochs} epochs x {steps_per_epoch} steps, warmup {warmup_steps} steps", flush=True)
    global_step = 0
    best_uni_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        # Same permutation on every rank (seeded), sharded round-robin by rank
        g = torch.Generator().manual_seed(args.seed * 100_003 + epoch)
        shard = torch.randperm(chunks.shape[0], generator=g)[rank::world_size]

        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for i in range(steps_per_epoch):
            batch = chunks[shard[i * per_device_bs:(i + 1) * per_device_bs]].to(device).long()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(batch[:, :-1]).logits
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if is_main and global_step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch}/{args.epochs} | Step {global_step}/{total_steps} | "
                      f"Loss {loss.item():.4f} | LR {lr:.2e} | {time.time() - t0:.0f}s", flush=True)
                log({"event": "train", "epoch": epoch, "step": global_step,
                     "loss": loss.item(), "lr": lr})

            if is_main and global_step % args.eval_every == 0:
                uni_acc = run_eval(raw_model, tokenizer, eval_bi, eval_uni, device, global_step, log)
                if uni_acc > best_uni_acc:
                    best_uni_acc = uni_acc
                    raw_model.save_pretrained(os.path.join(args.out_dir, "best"))
                model.train()

        log({"event": "epoch_end", "epoch": epoch, "avg_loss": epoch_loss / steps_per_epoch})
        if is_main:
            print(f"Epoch {epoch} done. Avg loss: {epoch_loss / steps_per_epoch:.4f}", flush=True)

    if is_main:
        print("Final evaluation...")
        uni_acc = run_eval(raw_model, tokenizer, eval_bi, eval_uni, device, global_step, log)
        if uni_acc > best_uni_acc:
            raw_model.save_pretrained(os.path.join(args.out_dir, "best"))
        raw_model.save_pretrained(os.path.join(args.out_dir, "final"))
        log_file.close()
        print(f"Done. Checkpoints and logs in {args.out_dir}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
