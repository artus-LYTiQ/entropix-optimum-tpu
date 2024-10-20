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
    return inputs, batch_size, sequence_length

def generate_text(model, inputs, max_new_tokens: int, cfg: SamplerConfig):
    logger.info("Starting text generation...")
    batch_size, sequence_length = inputs["input_ids"].shape
    device = model.device
    
    generated_ids = inputs["input_ids"].clone()
    attention_mask = inputs["attention_mask"].clone()
    
    for i in range(max_new_tokens):
        logger.debug(f"Generating token {i+1}/{max_new_tokens}")
        
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=True,
        )
        
        logits = outputs.logits[:, -1, :]
        attention_scores = outputs.attentions[-1]
        
        next_token = sample(generated_ids, logits, attention_scores, cfg)
        xm.mark_step()
        
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device)], dim=-1)
    
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
    
    prompts = ["Here's a funny thing:", "Once upon a time,"] * (args.batch_size // 2)
    inputs, batch_size, sequence_length = prepare_inputs(tokenizer, prompts, device, args.max_length)
    
    cfg = SamplerConfig()
    
    start = time.time()
    generated_ids = generate_text(model, inputs, args.max_new_tokens, cfg)
    xm.mark_step()
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