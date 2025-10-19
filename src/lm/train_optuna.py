import json
import math
import os
import time
from collections import deque
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Callable
from rich import print

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from lm.model import DecoderLM
from lm.utils import (
    determine_device,
    enable_tf32,
    dataset2tokens
)

# --- OPTUNA INTEGRATION START ---
import optuna

# Define a constant for your computational budget in FLOPs.
COMPUTATION_BUDGET_FLOPs = 1e15
# --- OPTUNA INTEGRATION END ---


def random_batch_sampler(
        tokens: torch.LongTensor, device: str, batch_size: int, seq_len: int
) -> Iterator[torch.LongTensor]:
    """An infinite generator that samples batches of sequences from the tokens."""
    while True:
        last_possible_start = len(tokens) - seq_len
        start_indices = torch.randint(0, last_possible_start + 1, (batch_size,))
        sequences = [tokens[i:i + seq_len] for i in start_indices]
        batch = torch.stack(sequences)
        yield batch.to(device)


def sequential_batch_sampler(
        tokens: torch.LongTensor, device: str, batch_size: int, seq_len: int
) -> Iterator[torch.LongTensor]:
    """A generator that yields batches of tokens sequentially."""
    num_tokens = len(tokens)
    tokens_per_batch = batch_size * seq_len
    num_batches = num_tokens // tokens_per_batch
    for i in range(num_batches):
        start_idx = i * tokens_per_batch
        end_idx = start_idx + tokens_per_batch
        batch_tokens = tokens[start_idx:end_idx]
        batch = batch_tokens.view(batch_size, seq_len)
        yield batch.to(device)


def cosine_lr_schedule(
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float,
        max_lr: float,
) -> Callable[[int], float]:
    def get_lr(t: int) -> float:
        """Outputs the learning rate at step t under the cosine schedule."""
        assert max_lr >= min_lr >= 0.0
        assert num_training_steps >= num_warmup_steps >= 0
        if t < num_warmup_steps:
            if num_warmup_steps > 0:
                return (t / num_warmup_steps) * max_lr
            return max_lr
        if t < num_training_steps:
            progress = (t - num_warmup_steps) / (num_training_steps - num_warmup_steps)
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
        return min_lr

    return get_lr


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def compute_language_modeling_loss(
        input_ids: torch.LongTensor, logits: torch.FloatTensor
) -> torch.FloatTensor:
    """Outputs the language modeling loss given input_ids and logits"""
    labels = input_ids[:, 1:].contiguous()
    logits = logits[:, :-1, :].contiguous()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss


def train(
        model: DecoderLM,
        batch_sampler: Iterator[torch.LongTensor],
        optimizer: torch.optim.Optimizer,
        lr_schedule: Callable[[int], float],
        autocast: torch.autocast | nullcontext,
        num_training_steps: int,
        grad_accumulation_steps: int,
) -> None:
    """A training loop for the language model"""
    losses = deque(maxlen=20 * grad_accumulation_steps)
    for step in (pbar := trange(num_training_steps)):
        t0 = time.time()
        lr = lr_schedule(step)
        set_lr(optimizer, lr)
        for _ in range(grad_accumulation_steps):
            input_ids = next(batch_sampler)
            with autocast:
                logits = model(input_ids)
            loss = compute_language_modeling_loss(input_ids, logits)
            (loss / grad_accumulation_steps).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        loss_mean = np.mean(losses).item()
        FLOPs_per_step = (
                model.flops_per_token
                * input_ids.shape[0]
                * input_ids.shape[1]
                * grad_accumulation_steps
        )
        t1 = time.time()
        dt = t1 - t0
        pbar.set_postfix(
            {
                "train loss": f"{loss_mean:.2f}",
                "TFLOPS": f"{FLOPs_per_step / dt / 1e12:.1f}",
            }
        )
        wandb.log({"train-loss": loss_mean, "learning-rate": lr}, step=step)


@torch.inference_mode()
def evaluate(
        model: DecoderLM,
        batch_sampler: Iterator[torch.LongTensor],
        autocast: torch.autocast | nullcontext,
) -> dict[str, float]:
    """Evaluates the model and returns a dictionary of metrics."""
    losses = []
    # Use a new tqdm iterator for evaluation
    for input_ids in tqdm(batch_sampler, desc="evaluating.."):
        with autocast:
            logits = model(input_ids)
        loss = compute_language_modeling_loss(input_ids, logits)
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    eval_results = {
        "val-loss": mean_loss,
        "val-perplexity": perplexity,
    }
    wandb.log(eval_results)
    return eval_results


# --- OPTUNA INTEGRATION START ---
def objective(trial: optuna.Trial) -> float:
    """The Optuna objective function to minimize."""
    # Start from a base config file
    config = OmegaConf.load("configs/GPT-tiny.yaml")

    # --- 1. Suggest Hyperparameters ---
    # Model parameters
    config.model_config.n_layer = trial.suggest_int("n_layer", 2, 6)
    config.model_config.n_head = trial.suggest_categorical("n_head", [2, 4, 8])
    # Suggest embedding size per head to ensure n_embd is divisible by n_head
    embd_per_head = trial.suggest_categorical("embd_per_head", [16, 32, 64])
    config.model_config.n_embd = embd_per_head * config.model_config.n_head

    # Training configuration parameters
    config.max_lr = trial.suggest_float("max_lr", 1e-4, 1e-2, log=True)
    config.min_lr = trial.suggest_float("min_lr", 1e-5, config.max_lr, log=True)
    config.weight_decay = trial.suggest_float("weight_decay", 1e-3, 0.1, log=True)
    config.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    config.seq_len = trial.suggest_categorical("seq_len", [64, 128, 256])
    config.model_config.n_positions = config.seq_len
    config.grad_accumulation_steps = trial.suggest_categorical("grad_accumulation_steps", [1, 2, 4, 8])

    # --- 2. Adaptively Compute Training Steps to Fill Budget ---
    # Temporarily create the model on CPU to calculate FLOPs
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    temp_model = DecoderLM(tokenizer.n_vocab, **config.model_config)
    flops_per_step = (
        temp_model.flops_per_token
        * config.batch_size
        * config.seq_len
        * config.grad_accumulation_steps
    )
    del temp_model # free memory

    if flops_per_step <= 0:
        raise optuna.exceptions.TrialPruned("FLOPs per step is zero.")

    # Calculate the total number of training steps to meet the budget
    num_training_steps = int(COMPUTATION_BUDGET_FLOPs // flops_per_step)
    config.num_training_steps = num_training_steps

    # Now suggest warmup steps based on the calculated total steps
    config.num_warmup_steps = trial.suggest_int("num_warmup_steps", 0, int(num_training_steps * 0.1))

    # Prune if there aren't enough steps to even warm up
    if num_training_steps < config.num_warmup_steps:
        raise optuna.exceptions.TrialPruned("Not enough training steps for warmup.")

    # --- 3. Setup and Run Training ---
    config.output_dir = os.path.join("outputs", f"trial_{trial.number}")
    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))

    print("#" * 40, f"STARTING TRIAL {trial.number}", "#" * 40, sep="\n")
    print(OmegaConf.to_yaml(config).strip())

    wandb.init(
        project="llms-hw2-optuna-updated",
        config=OmegaConf.to_container(config),
        name=f"trial-{trial.number}",
        reinit=True,
    )

    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)

    train_tokens, val_tokens, *_ = dataset2tokens(seq_len=config.seq_len)
    train_sampler = random_batch_sampler(
        train_tokens, device, config.batch_size, config.seq_len
    )
    val_sampler = sequential_batch_sampler(
        val_tokens, device, config.batch_size, config.seq_len
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        betas=(0.9, 0.99),
        fused=device == "cuda",
        weight_decay=config.weight_decay,
    )
    lr_schedule = cosine_lr_schedule(
        config.num_warmup_steps, config.num_training_steps, config.min_lr, config.max_lr
    )
    autocast = (
        torch.autocast(
            device,
            dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
        )
        if device == "cuda"
        else nullcontext()
    )

    model.train()
    train(
        model,
        train_sampler,
        optimizer,
        lr_schedule,
        autocast,
        config.num_training_steps,
        config.grad_accumulation_steps,
    )

    model.eval()
    eval_results = evaluate(model, val_sampler, autocast)
    print(f"Trial {trial.number} evaluation results:", json.dumps(eval_results, indent=2))

    wandb.finish()

    # --- 4. Return the metric for Optuna to optimize ---
    return eval_results["val-loss"]
# --- OPTUNA INTEGRATION END ---


def main():
    enable_tf32()
    # Create a study object and specify the direction to 'minimize' validation loss.
    study = optuna.create_study(direction="minimize")

    # Start the optimization. Optuna will call the objective function `n_trials` times.
    study.optimize(objective, n_trials=50)

    print("\n" + "="*40)
    print("HYPERPARAMETER OPTIMIZATION FINISHED")
    print("="*40)
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (val-loss): {trial.value:.4f}")
    print("  Best Parameters: ")
    for key, value in trial.params.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.2e}")
        else:
            print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
