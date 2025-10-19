import argparse
import json
import os
import math

import tiktoken
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import trange
from lm.model import DecoderLM
from lm.utils import determine_device, enable_tf32
from lm.train import compute_language_modeling_loss


def softmax_with_temperature(
    logits: torch.FloatTensor, temperature: float
) -> torch.FloatTensor:
    """Turns logits into probabilities under softmax (with temperature)

    Args:
        logits: a 2d torch tensor of token ids (B, V)
        temperature: temperature of the softmax function

    Returns:
        a 2d torch tensor of token probabilities (B, V)
    """

    # to avoid division by 0
    temperature = max(temperature, 1e-5)

    # Scale logits by temperature
    scaled_logits = logits / temperature

    # Apply softmax along the last dimension (the vocabulary dimension V)
    return F.softmax(scaled_logits, dim=-1)


@torch.inference_mode()
def generate(
        model: DecoderLM,
        device: str,
        tokenizer: tiktoken.Encoding,
        prefixes: list[str],
        batch_size: int,
        max_new_tokens: int = 32,
        temperature: float = 0.1,
        top_k: int = 0,
) -> list[str]:
    """Generates completions conditioned on prefixes.

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        prefixes: a list of strings as prefixes for generation
        batch_size: number of prefixes to batch together during generation
        max_new_tokens: the number of tokens to generate for each prefix
        temperature: temperature parameter of softmax
        top_k: top-k sampling (0 to disable)

    Returns:
        a list of strings (continuations to prefixes)

    Note: you should implement a batched version of this function by
        left-padding tokenized prefixes with `tokenizer.eot_token` so that all
        sequences have equal length. `attention_mask` should be set to 0.0 for
        padding tokens, and 1.0 everywhere else.

    hint: tokenizer.encode, tokenizer.decode
    """

    all_generations = []
    pad_token_id = tokenizer.eot_token

    for i in trange(0, len(prefixes), batch_size, desc="Generating"):
        # 1. Get batch and tokenize
        batch_prefixes = prefixes[i: i + batch_size]
        batch_tokenized = [tokenizer.encode(p) for p in batch_prefixes]

        # 2. Left-pad the batch
        B = len(batch_prefixes)
        max_len_in_batch = max(len(t) for t in batch_tokenized)
        T = max_len_in_batch

        input_ids = torch.full((B, T), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, T), dtype=torch.float, device=device)

        for j, tokens in enumerate(batch_tokenized):
            seq_len = len(tokens)
            # Paste the sequence on the right-hand side
            input_ids[j, T - seq_len:] = torch.tensor(tokens, dtype=torch.long, device=device)
            # Set mask to 1.0 for real tokens
            attention_mask[j, T - seq_len:] = 1.0

        # 3. Get initial logits to start generation
        prefix_logits = model(input_ids, attention_mask)  # (B, T, V)

        # 4. Generation loop
        generated_token_ids = []  # List to store (B, 1) tensors
        current_input_ids = input_ids
        current_attn_mask = attention_mask

        # Get the logits for the *last* token of the prefix to start
        last_token_logits = prefix_logits[:, -1, :]  # (B, V)

        for _ in range(max_new_tokens):

            # --- 4a. Apply sampling strategy (greedy, top-k, or temp) ---

            # Greedy sampling
            if temperature < 1e-5:
                next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(-1)  # (B, 1)

            # Sampling with temperature and (optional) top-k
            else:
                # Scale by temperature
                scaled_logits = last_token_logits / temperature

                # (Optional) Top-K filtering
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, top_k)
                    kth_value = v[:, -1].unsqueeze(-1)  # (B, 1)
                    # Set all logits *below* this k-th value to -inf
                    scaled_logits[scaled_logits < kth_value] = -float('Inf')

                # Get probabilities
                probs = F.softmax(scaled_logits, dim=-1)  # (B, V)

                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # --- 4b. (Perplexity calculation removed) ---

            # --- 4c. Store for decoding ---
            generated_token_ids.append(next_token)

            # --- 4d. Prepare for next iteration ---
            # Append the new token and update the attention mask
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            new_attn_bit = torch.ones((B, 1), dtype=torch.float, device=device)
            current_attn_mask = torch.cat([current_attn_mask, new_attn_bit], dim=1)

            # --- 4e. Get logits for the *next* step ---
            all_logits = model(current_input_ids, current_attn_mask)
            last_token_logits = all_logits[:, -1, :]  # (B, V)

        # 5. Decode the generations for this batch
        # Concatenate all generated (B, 1) tensors into (B, max_new_tokens)
        batch_new_tokens = torch.cat(generated_token_ids, dim=1)
        batch_new_tokens_list = batch_new_tokens.cpu().tolist()

        # Decode and add to the final list
        decoded_strings = [tokenizer.decode(t) for t in batch_new_tokens_list]
        all_generations.extend(decoded_strings)

    # 6. (Perplexity calculation removed)

    return all_generations

#
# def main():
#     prefixes = [
#     {"prefix": "In this paper"},
#     ]
#
#     with open("prefixes.jsonl", "w") as f:
#         for p in prefixes:
#             json.dump(p, f)
#             f.write("\n")
#
#     enable_tf32()
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--config",
#         type=OmegaConf.load,
#         help="the yaml config file used for model training",
#     )
#     parser.add_argument(
#         "--prefixes",
#         type=str,
#         default="prefixes.jsonl",
#         help="a json file with a list of strings as prefixes for generation",
#     )
#     parser.add_argument(
#         "--max_new_tokens",
#         type=int,
#         default=32,
#         help="number of new tokens to generate",
#     )
#     parser.add_argument(
#         "--temperature", type=float, default=0.1, help="temperature in sampling"
#     )
#
#     args = parser.parse_args()
#     config = args.config
#     with open(args.prefixes) as f:
#         prefixes = [json.loads(line)["prefix"] for line in f]
#     max_new_tokens = args.max_new_tokens
#     temperature = args.temperature
#
#     # initialize tokenizer and model
#     model_path = os.path.join(config.output_dir, "model.pt")
#     assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
#     tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
#     device = determine_device() if config.device == "auto" else config.device
#     model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#
#     # generate and save outputs
#     model.eval()
#     generations = generate(
#         model,
#         device,
#         tokenizer,
#         prefixes,
#         config.batch_size,
#         max_new_tokens,
#         temperature,
#     )
#
#     generation_path = os.path.join(config.output_dir, "generation.jsonl")
#     print(f"writing generations to {generation_path}")
#     with open(generation_path, "w") as f:
#         for prefix, generation in zip(prefixes, generations):
#             json.dump({"prefix": prefix, "generation": generation}, f)
#             f.write("\n")
#
#     print("done!")
def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        help="the yaml config file used for model training",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        default="prefixes.jsonl",
        help="a json file with a list of strings as prefixes for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="temperature in sampling"
    )
    # --- NOTE: I'm adding a top_k argument for flexibility ---
    # Although your request is for k=10, this makes the script reusable
    parser.add_argument(
        "--top_k", type=int, default=10, help="top-k sampling (0 to disable)"
    )

    args = parser.parse_args()
    config = args.config
    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f]
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_k = args.top_k  # Read top_k from args

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 1. GREEDY SAMPLING (1 Sample) ---
    print("Running greedy generation (1 sample)...")
    greedy_generations = generate(
        model,
        device,
        tokenizer,
        prefixes,
        config.batch_size,
        max_new_tokens,
        temperature=0.0,  # Setting temperature to 0 triggers greedy logic
        top_k=0
    )

    generation_path_greedy = os.path.join(config.output_dir, "generation_greedy.jsonl")
    print(f"writing greedy generations to {generation_path_greedy}")
    with open(generation_path_greedy, "w") as f:
        for prefix, generation in zip(prefixes, greedy_generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    # --- 2. TOP-K SAMPLING (5 Samples, k=10) ---
    num_top_k_samples = 5
    print(f"\nRunning top-k generation ({num_top_k_samples} samples, k={top_k})...")

    generation_path_topk = os.path.join(config.output_dir, f"generation_topk_k{top_k}.jsonl")
    print(f"writing top-k generations to {generation_path_topk}")

    with open(generation_path_topk, "w") as f:
        # Loop 5 times to get 5 different samples
        for i in range(num_top_k_samples):
            print(f"  Generating sample {i + 1}/{num_top_k_samples}...")
            top_k_generations = generate(
                model,
                device,
                tokenizer,
                prefixes,
                config.batch_size,
                max_new_tokens,
                temperature=temperature,  # Use temperature from args
                top_k=top_k  # Use top_k from args (or default 10)
            )

            # Write this sample's results to the file
            for prefix, generation in zip(prefixes, top_k_generations):
                json.dump({
                    "prefix": prefix,
                    "generation": generation,
                    "sample_index": i
                }, f)
                f.write("\n")

    print("\nDone!")

if __name__ == "__main__":
    main()
