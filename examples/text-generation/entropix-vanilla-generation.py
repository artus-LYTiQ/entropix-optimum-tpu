import argparse
import time
import platform
from typing import List
import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import AutoTokenizer
from optimum.tpu.modeling import AutoModelForCausalLM
from loguru import logger

from sampling_logic import SamplerConfig, sample

os.environ["PJRT_DEVICE"] = "TPU"

logger.add("generation_log.log", rotation="500 MB")

def setup_model_and_tokenizer(model_id: str, torch_dtype: torch.dtype):
    logger.info(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, use_pjit=True)
    device = xm.xla_device()
    model = model.to(device)
    model = model.eval()
    logger.info(f"Model loaded on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device

def prepare_inputs(tokenizer, prompts: List[str], device: torch.device):
    logger.info("Preparing inputs...")
    inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
    batch_size, sequence_length = inputs["input_ids"].shape
    logger.info(f"Input prepared with batch_size={batch_size}, sequence_length={sequence_length}")
    return inputs, batch_size, sequence_length

def initialize_generation(batch_size: int, sequence_length: int, max_new_tokens: int, device: torch.device):
    logger.info("Initializing generation...")
    cache_position = torch.arange(sequence_length, device=device)
    generated_ids = torch.zeros(
        (batch_size, sequence_length + max_new_tokens),
        dtype=torch.int32,
        device=device,
    )
    return cache_position, generated_ids

def generate_text(model, inputs, generated_ids, attention_mask, pos_ids, cache_position, max_new_tokens: int, cfg: SamplerConfig):
    logger.info("Starting text generation...")
    sequence_length = inputs["input_ids"].shape[1]
    batch_size = inputs["input_ids"].shape[0]
    
    for i in range(max_new_tokens):
        logger.debug(f"Generating token {i+1}/{max_new_tokens}")
        outputs = model(
            input_ids=generated_ids[:, :sequence_length + i],
            attention_mask=attention_mask,
            position_ids=pos_ids,
            cache_position=cache_position,
            return_dict=True,
            use_cache=True,
            output_attentions=True,
        )
        
        logits = outputs.logits
        attention_scores = outputs.attentions[-1]
        
        next_token = sample(generated_ids, logits, attention_scores, cfg)
        xm.mark_step()
        generated_ids[:, sequence_length + i] = next_token.squeeze(-1)
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=model.device)], dim=1)
        pos_ids = torch.cat([pos_ids, (pos_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
        cache_position = torch.tensor([sequence_length + i], device=model.device)
    
    logger.info("Text generation completed")
    return generated_ids

def main(index):
    parser = argparse.ArgumentParser(description="Text generation example")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model ID (e.g.: google/gemma-2b, mistralai/Mistral-7B-v0.3)")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per TPU device")
    args = parser.parse_args()

    logger.info(f"Starting the program on TPU device {index}")
    prg_start = time.time()
    
    model, tokenizer, device = setup_model_and_tokenizer(args.model_id, torch.bfloat16)
    
    # Adjust prompts for the number of devices
    prompts = ["Here's a funny thing:", "Once upon a time,"] * (args.batch_size // 2)
    inputs, batch_size, sequence_length = prepare_inputs(tokenizer, prompts, device)
    
    cache_position, generated_ids = initialize_generation(batch_size, sequence_length, args.max_new_tokens, device)
    generated_ids[:, :sequence_length] = inputs["input_ids"].to(torch.int32)
    
    attention_mask = inputs["attention_mask"]
    pos_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
    
    cfg = SamplerConfig()
    
    start = time.time()
    generated_ids = generate_text(model, inputs, generated_ids, attention_mask, pos_ids, cache_position, args.max_new_tokens, cfg)
    xm.mark_step()
    end = time.time()
    
    logger.info(f"Generation took {end - start:.2f} seconds")
    
    # Gather results from all devices
    all_generated_ids = xm.mesh_reduce('generated_ids', generated_ids, torch.cat)
    if xm.get_ordinal() == 0:  # Only the first process decodes and logs
        decoded_texts = tokenizer.batch_decode(all_generated_ids)
        for i, text in enumerate(decoded_texts):
            logger.info(f"Generated text {i}: {text}")
    
    logger.info(f"Program run in {time.time() - prg_start:.2f} seconds. Device: {device} System: {platform.system()}")
    
    # Print TPU metrics
    logger.info(f"TPU Metrics: {torch_xla.debug.metrics.metric_data()}")

if __name__ == "__main__":
    def _mp_fn(index):
        with torch.no_grad():
            main(index)
    xmp.spawn(_mp_fn, nprocs=4)