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
    # Step 1: Extract the logits corresponding to the last token in the sequence
    next_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]

    # Step 2: Perform greedy sampling (argmax over vocab dimension)
    next_token_id = torch.argmax(next_logits, dim=-1)  # Shape: [batch_size]

    # Step 3: Optionally, expand dimensions to match expected output shape [batch_size, 1]
    next_token_id = next_token_id.unsqueeze(-1).int()  # Shape: [batch_size, 1]

    return next_token_id



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
    inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    batch_size, sequence_length = inputs["input_ids"].shape
    logger.info(f"Input prepared with batch_size={batch_size}, sequence_length={sequence_length}")
    assert batch_size > 0, f"Batch size must be greater than 0, got {batch_size}"
    assert sequence_length > 0, f"Sequence length must be greater than 0, got {sequence_length}"
    return inputs, batch_size, sequence_length

def generate_text(model, inputs, max_new_tokens: int, cfg: SamplerConfig):
    logger.info("Starting text generation...")
    batch_size, sequence_length = inputs["input_ids"].shape
    device = model.device
    
    # Assertions with inferred checks
    assert batch_size > 0, f"Batch size must be positive, but got {batch_size}"
    assert sequence_length > 0, f"Sequence length must be positive, but got {sequence_length}"

    generated_ids = inputs["input_ids"].clone()
    attention_mask = inputs["attention_mask"].clone()
    
    assert generated_ids.dim() == 2, f"Expected 2D tensor for generated_ids, but got {generated_ids.dim()}D"
    assert attention_mask.dim() == 2, f"Expected 2D tensor for attention_mask, but got {attention_mask.dim()}D"
    assert generated_ids.shape == attention_mask.shape, \
        f"generated_ids and attention_mask shapes must match, but got {generated_ids.shape} vs {attention_mask.shape}"
    
    for i in range(max_new_tokens):
        outputs = model(input_ids=generated_ids, attention_mask=attention_mask, return_dict=True, output_attentions=True)
        
        assert outputs.logits.dim() == 3, f"Expected 3D logits, but got {outputs.logits.dim()}D. logits.shape={outputs.logits.shape}"
        assert len(outputs.attentions) > 0, "Expected non-empty attention outputs"
        assert all(att.dim() == 4 for att in outputs.attentions), \
            f"Expected all attention tensors to be 4D, but got {[att.dim() for att in outputs.attentions]}"
        
        # logits = outputs.logits[:, -1, :]  # Last token logits
        logits = outputs.logits

        # attentions = outputs.attentions[-1]  # Use the last layer’s attention
        attentions = outputs.attentions
        
        # Validate logits and attention score dimensions
        assert logits.shape[0] == batch_size, f"Logits batch size mismatch: got {logits.shape[0]} instead of {batch_size}"
        assert logits.shape[2] == model.config.vocab_size, \
            f"Logits vocab size mismatch: got {logits.shape[1]}, expected {model.config.vocab_size}"

        # DEBUG ONLY: disable entropix sampling for now
        # next_token = sample_greedy(generated_ids, logits, attentions, cfg)
        next_token = sample_greedy(logits)
        
        # After sampling, check if the token dimensions match expected sizes
        assert next_token.dim() == 2, f"Expected 2D tensor for next_token, but got {next_token.dim()}D"
        assert next_token.shape[1] == 1, f"Expected next_token to have shape (batch_size, 1), but got {next_token.shape}"
        
        # Update tensors with new token
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device)], dim=-1)
        
        # Ensure consistency after update
        assert generated_ids.shape[1] == attention_mask.shape[1], \
            f"Mismatch after update: generated_ids {generated_ids.shape} vs attention_mask {attention_mask.shape}"
    
    logger.info("Text generation completed")
    return generated_ids

def main():
    parser = argparse.ArgumentParser(description="Text generation example")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model ID (e.g.: google/gemma-2b, mistralai/Mistral-7B-v0.3)")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
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
    generated_ids = generate_text(model, inputs, args.max_new_tokens, cfg)
    # xm.mark_step()
    end = time.time()
    
    logger.info(f"Generation took {end - start:.2f} seconds")
    
    decoded_texts = tokenizer.batch_decode(generated_ids)
    for i, text in enumerate(decoded_texts):
        logger.info(f"Generated text {i}: {text}")
    
    logger.info(f"Program run in {time.time() - prg_start:.2f} seconds. Device: {device} System: {platform.system()}")
    
    # Print TPU metrics
    logger.info(f"TPU Metrics: {torch_xla.debug.metrics.metric_data()}")

if __name__ == "__main__":
    main()