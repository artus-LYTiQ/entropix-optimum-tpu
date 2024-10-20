import argparse
import datetime
import os
import platform
import time
from typing import List, Dict, Tuple

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, StaticCache
from optimum.tpu.modeling import AutoModelForCausalLM

os.environ["PJRT_DEVICE"] = "TPU"

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
    return entropy, varentropy

def calculate_metrics(logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
    attn_varentropy = torch.var(attn_entropy, dim=1)

    mean_attention = torch.mean(attention_probs, dim=1)
    agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))

    interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))

    return {
        "logits_entropy": torch.mean(entropy),
        "logits_varentropy": torch.mean(varentropy),
        "attn_entropy": torch.mean(attn_entropy),
        "attn_varentropy": torch.mean(attn_varentropy),
        "agreement": torch.mean(agreement),
        "interaction_strength": interaction_strength
    }

class SamplerConfig:
    def __init__(self):
        self.temp = 0.666
        self.top_p = 0.90
        self.top_k = 27
        self.min_p = 0.03
        self.low_ent_thresh = 0.1
        self.low_vent_thresh = 0.1
        self.med_ent_thresh = 3.0
        self.high_ent_thresh = 5.0
        self.high_vent_thresh = 5.0
        self.helv_attn_ent_offset = 1.3
        self.helv_attn_ent_coef = 0.2
        self.lehv_interaction_strength_offset = 1.2
        self.lehv_interaction_strength_coef = 0.3
        self.hehv_attn_ent_coef = 0.2
        self.hehv_attn_vent_offset = 2.0
        self.hehv_attn_vent_coef = 0.5
        self.n_adaptive_samples = 5
        self.ada_temp_logits = 0.3
        self.ada_temp_attn = 0.2
        self.ada_temp_agree = 0.2
        self.ada_top_p = 0.1
        self.ada_top_k_int = 0.3
        self.ada_top_k_agree = 0.2
        self.ada_min_p = 0.5
        self.ada_score_logits_ent = 0.1
        self.ada_score_attn_ent = 0.2
        self.ada_score_logits_vent = 0.3
        self.ada_score_attn_vent = 0.4
        self.ada_score_agree = 0.5
        self.ada_score_int = 0.6

def sample_top_p_top_k(logits, temperature, top_p, top_k, min_p):
    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
    
    # Apply min_p sampling
    if min_p > 0.0:
        p_max = torch.max(probs, dim=-1, keepdim=True).values
        probs = torch.where(probs < (min_p * p_max), torch.zeros_like(probs), probs)
    
    # Apply top-k sampling
    top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
    
    # Apply top-p sampling
    cumulative_probs = torch.cumsum(top_k_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    top_k_probs[sorted_indices_to_remove] = 0.0
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    
    # Sample from the filtered distribution
    sample_probs = torch.distributions.Categorical(top_k_probs).sample()
    sample = torch.gather(top_k_indices, -1, sample_probs.unsqueeze(-1))
    
    return sample

def sample(gen_tokens: torch.Tensor, logits: torch.Tensor, attention_scores: torch.Tensor, cfg: SamplerConfig,
           clarifying_question_token: int = 2564) -> torch.Tensor:
    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if ent < cfg.low_ent_thresh and vent < cfg.low_vent_thresh:
        return torch.argmax(logits[:, -1], dim=-1, keepdim=True)

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif ent > cfg.high_ent_thresh and vent < cfg.low_vent_thresh:
        # Insert a clarifying question token if not already present
        if not torch.isin(gen_tokens[:, -1], clarifying_question_token).any():
            return torch.tensor([[clarifying_question_token]], device=logits.device)
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = cfg.helv_attn_ent_offset + cfg.helv_attn_ent_coef * attn_ent
            return sample_top_p_top_k(logits[:, -1], min(1.5, cfg.temp * temp_adj), cfg.top_p, cfg.top_k, cfg.min_p)

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif ent < cfg.high_ent_thresh and vent > cfg.high_vent_thresh:
        temp_adj = cfg.lehv_interaction_strength_offset + cfg.lehv_interaction_strength_coef * interaction_strength
        top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))
        return sample_top_p_top_k(logits[:, -1], min(1.5, cfg.temp * temp_adj), cfg.top_p, top_k_adj, cfg.min_p)

    # High Entropy, High Varentropy: "resampling in the mist"
    elif ent > cfg.med_ent_thresh and vent > cfg.high_vent_thresh:
        temp_adj = cfg.hehv_attn_vent_offset + cfg.hehv_attn_vent_coef * attn_vent
        top_p_adj = max(0.5, cfg.top_p - cfg.hehv_attn_ent_coef * attn_ent)
        return sample_top_p_top_k(logits[:, -1], max(2.0, cfg.temp * temp_adj), top_p_adj, cfg.top_k, cfg.min_p)

    # Middle ground: use adaptive sampling
    else:
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

        temperature = cfg.temp * (1 + cfg.ada_temp_logits * logits_uncertainty + cfg.ada_temp_attn * attn_uncertainty - cfg.ada_temp_agree * metrics["agreement"])
        top_p = torch.clamp(cfg.top_p * (1 + cfg.ada_top_p * metrics["attn_varentropy"]), 0.1, 1.0)
        top_k = int(torch.clamp(
            torch.round(cfg.top_k * (1 + cfg.ada_top_k_int * metrics["interaction_strength"].item() - cfg.ada_top_k_agree * metrics["agreement"].item())),
            min=1,
            max=100
        ))
        min_p = torch.clamp(cfg.min_p * (1 - cfg.ada_min_p * logits_uncertainty), 0.01, 0.5)

        samples = []
        for _ in range(cfg.n_adaptive_samples):
            sample = sample_top_p_top_k(logits[:, -1], temperature, top_p, top_k, min_p)
            samples.append(sample)

        def score_sample(sample):
            log_prob = torch.sum(torch.nn.functional.log_softmax(logits[:, -1], dim=-1) * torch.nn.functional.one_hot(sample, logits.shape[-1]))
            confidence_score = (
                (1 - metrics["logits_entropy"]) * cfg.ada_score_logits_ent +
                (1 - metrics["attn_entropy"]) * cfg.ada_score_attn_ent +
                (1 - metrics["logits_varentropy"]) * cfg.ada_score_logits_vent +
                (1 - metrics["attn_varentropy"]) * cfg.ada_score_attn_vent +
                metrics["agreement"] * cfg.ada_score_agree +
                metrics["interaction_strength"] * cfg.ada_score_int
            )
            return log_prob + confidence_score

        sample_scores = torch.stack([score_sample(sample) for sample in samples])
        best_sample_idx = torch.argmax(sample_scores)
        return samples[best_sample_idx]

def main():
    parser = argparse.ArgumentParser(description="Text generation example")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model ID (e.g.: google/gemma-2b, mistralai/Mistral-7B-v0.3)")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--max_cache_length", type=int, default=256, help="Maximum cache length for the model")
    args = parser.parse_args()

    prg_start = time.time()
    print(f"⏳ Loading model {args.model_id}...")
    model_id = args.model_id
    torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    device = model.device
    model = model.eval()
    print(f"✅ Model loaded in {time.time() - prg_start} seconds on {device=}.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = ["Here's a funny thing:", "Once upon a time,"]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    batch_size, sequence_length = inputs["input_ids"].shape
    max_cache_length = 1024
    max_new_tokens = args.max_new_tokens

    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_length,
        device=model.device,
        dtype=model.dtype,
    )
    start = time.time()
    cache_position = torch.arange(sequence_length, device=device)
    generated_ids = torch.zeros(
        (batch_size, sequence_length + max_new_tokens + 1),
        dtype=torch.int,
        device=device,
    )
    generated_ids[:, cache_position] = inputs["input_ids"].to(torch.int)

    attention_mask = inputs["attention_mask"]
    pos_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
    
    cfg = SamplerConfig()

    for i in range(max_new_tokens):
        outputs = model(
            input_ids=generated_ids[:, :sequence_length + i],
            attention_mask=attention_mask,
            position_ids=pos_ids,
            cache_position=cache_position,
            return_dict=True,
            use_cache=True,
            past_key_values=past_key_values,
            output_attentions=True,
        )
        
        logits = outputs.logits
        attention_scores = outputs.attentions[-1]  # Use the last layer's attention scores
        
        next_token = sample(generated_ids, logits, attention_scores, cfg)
        xm.mark_step()
        generated_ids[:, sequence_length + i] = next_token.squeeze(-1)
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device)], dim=1)
        pos_ids = torch.cat([pos_ids, (pos_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
        cache_position = torch.tensor([sequence_length + i], device=device)

    end = time.time()
    print(f"Generation took {end - start} seconds.")

    decoded_texts = tokenizer.batch_decode(generated_ids)
    for i, text in enumerate(decoded_texts):
        print(i,text)

cfg = SamplerConfig()

for i in range(max_new_tokens):
    outputs = model(
        input_ids=generated_ids[:, :sequence_length + i],
        attention_mask=attention_mask,
        position_ids=pos_ids,
        cache_position=cache_position,
        return_dict=True,
        use_cache=True,
        past_key_values=past_key_values,
        output_attentions=True,
    )
    
    logits = outputs.logits
    attention_scores = outputs.attentions[-1]  # Use the last layer's attention scores
    
    next_token = sample(generated_ids, logits, attention_scores, cfg)
    xm.mark_step()
    generated_ids[:, sequence_length + i] = next_token.squeeze(-1)
    attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device)], dim=1)
    pos_ids = torch.cat([pos_ids, (pos_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
    cache_position = torch.tensor([sequence_length + i], device=device)

end = time.time()
print(f"Generation took {end - start} seconds.")

decoded_texts = tokenizer.batch_decode(generated_ids)
for i, text in enumerate(decoded_texts):
    print(f"Generated text {i}:", text)

print(f"Program run in {time.time() - prg_start} seconds. Device: {device} System: {platform.system()}")
if __name__ == "main":
    with torch.no_grad():
        main()