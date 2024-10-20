import argparse
import time
import platform
from typing import List
import os
import sys

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer
from optimum.tpu.modeling import AutoModelForCausalLM
from loguru import logger

from sampling_logic_entropix_vanilla import SamplerConfig, sample

os.environ["PJRT_DEVICE"] = "TPU"

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
logger.add("generation_log.log", rotation="500 MB", level="INFO")

def sample_greedy(logits):
    next_logits = logits[:, -1, :]
    next_token_id = torch.argmax(next_logits, dim=-1)
    next_token_id = next_token_id.unsqueeze(-1).int()
    logger.info(f"Greedy sampling result: {next_token_id}")
    return next_token_id.to(logits.device)

def log_token_probabilities(tokenizer, logits, current_token, top_k=10):
    # Get the last token's logits
    last_token_logits = logits[:, -1, :]
    
    # Apply softmax to convert logits to probabilities
    probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)
    
    # Get top k token ids and probabilities
    top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k, dim=-1)
    
    # Convert to list for easier handling
    top_k_probs = top_k_probs[0].tolist()
    top_k_indices = top_k_indices[0].tolist()
    
    # Decode tokens
    current_token_decoded = tokenizer.decode(current_token)
    top_k_decoded = tokenizer.batch_decode(top_k_indices)
    
    # Prepare log message
    log_message = f"Current token: {current_token_decoded}\n"
    log_message += "Top 10 candidates:\n"
    for token, prob in zip(top_k_decoded, top_k_probs):
        log_message += f"  {token}: {prob:.4f}\n"
    
    # Log the message
    logger.info(log_message)


def setup_model_and_tokenizer(model_id: str, torch_dtype: torch.dtype, device: torch.device):
    logger.info(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model = model.to(device)
    model = model.eval()
    logger.info(f"Model loaded on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def prepare_inputs(tokenizer, prompts: List[str], device: torch.device, max_length: int):
    logger.info("Preparing inputs...")
    inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    batch_size, sequence_length = inputs["input_ids"].shape
    logger.info(f"Input prepared with batch_size={batch_size}, sequence_length={sequence_length}")
    
    assert batch_size > 0, f"Batch size must be greater than 0, got {batch_size}"
    assert sequence_length > 0, f"Sequence length must be greater than 0, got {sequence_length}"
    
    for i, (ids, mask) in enumerate(zip(inputs["input_ids"], inputs["attention_mask"])):
        actual_length = mask.sum().item()
        logger.info(f"Input {i}: {tokenizer.decode(ids[:actual_length])}")
        logger.info(f"Last token of input {i}: {tokenizer.decode([ids[actual_length-1]])}")
    
    return inputs, batch_size, sequence_length

def generate_text(model, tokenizer, inputs, max_new_tokens: int, cfg: SamplerConfig):
    logger.info("Starting text generation...")
    batch_size, max_sequence_length = inputs["input_ids"].shape
    device = model.device
    
    generated_ids = inputs["input_ids"].clone().to(device)
    attention_mask = inputs["attention_mask"].clone().to(device)
    
    # Find the actual end of each input sequence (last non-padding token)
    input_lengths = attention_mask.sum(dim=1)
    
    # Log initial input content
    for b in range(batch_size):
        logger.info(f"Initial input batch {b}:")
        actual_length = input_lengths[b].item()
        logger.info(f"  Tokens: {generated_ids[b, :actual_length].tolist()}")
        logger.info(f"  Decoded: {tokenizer.decode(generated_ids[b, :actual_length])}")
    
    for i in range(max_new_tokens):
        # Use only the non-padded part of the input plus generated tokens for each batch item
        max_current_length = input_lengths.max().item() + i
        active_generated_ids = [ids[:length + i] for ids, length in zip(generated_ids, input_lengths)]
        active_attention_mask = [mask[:length + i] for mask, length in zip(attention_mask, input_lengths)]
        
        # Pad sequences to the maximum current length
        active_generated_ids = torch.nn.utils.rnn.pad_sequence(active_generated_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        active_attention_mask = torch.nn.utils.rnn.pad_sequence(active_attention_mask, batch_first=True, padding_value=0)
        
        outputs = model(input_ids=active_generated_ids, 
                        attention_mask=active_attention_mask, 
                        return_dict=True)
        
        logits = outputs.logits
        next_token = sample_greedy(logits)
        
        # Append the new token to each sequence
        for b in range(batch_size):
            current_length = input_lengths[b].item() + i
            generated_ids[b, current_length] = next_token[b]
            attention_mask[b, current_length] = 1
            input_lengths[b] += 1
    
    logger.info("Text generation completed")
    
    # Prepare the final output
    final_output = []
    for b in range(batch_size):
        sequence = generated_ids[b, :input_lengths[b]].tolist()
        decoded = tokenizer.decode(sequence)
        # Replace <|eot_id|> with a newline character
        decoded = decoded.replace("<|eot_id|>", "\n")
        final_output.append(decoded)
    
    return final_output

def main():
    parser = argparse.ArgumentParser(description="Text generation example")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model ID (e.g.: google/gemma-2b, mistralai/Mistral-7B-v0.3)")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Max input sequence length")
    parser.add_argument("--device", type=str, default="xla:0", help="XLA device to use (e.g., xla:0, xla:1, xla:2, xla:3)")
    args = parser.parse_args()

    logger.info(f"Starting the program on device {args.device}")
    prg_start = time.time()
    
    device = torch.device(args.device)
    model, tokenizer = setup_model_and_tokenizer(args.model_id, torch.bfloat16, device)
    
    # Ensure we have at least one prompt, and repeat it to match the batch size
    base_prompts = ["Here's a funny thing:", "Once upon a time,"]
    prompts = (base_prompts * (args.batch_size // 2 + 1))[:args.batch_size]
    logger.info(f"Using prompts: {prompts}")
    
    inputs, batch_size, sequence_length = prepare_inputs(tokenizer, prompts, device, args.max_length)
    
    cfg = SamplerConfig()
    
    start = time.time()
    generated_ids = generate_text(model, tokenizer, inputs, args.max_new_tokens, cfg)
    generation_end = time.time()
    xm.mark_step()
    end = time.time()
    
    logger.info(f"Generation took {generation_end - start:.2f} seconds and the TPU needed another {end - generation_end:.2f} seconds \
                more for a total time of {end - start:.2f} seconds")
    
    decoded_texts = tokenizer.batch_decode(generated_ids)
    for i, text in enumerate(decoded_texts):
        logger.info(f"Generated text {i}: {text}")
    
    logger.info(f"Program run in {time.time() - prg_start:.2f} seconds. Device: {device} System: {platform.system()}")
    
    # Print TPU metrics
    logger.info(f"TPU Metrics: {torch_xla.debug.metrics.metrics_report()}")

if __name__ == "__main__":
    main()