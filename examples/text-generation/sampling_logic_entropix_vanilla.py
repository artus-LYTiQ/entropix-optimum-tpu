import torch
from typing import Dict, Tuple
from loguru import logger

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

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

def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug(f"calculate_varentropy_logsoftmax input shape: {logits.shape}")
    assert logits.dim() >= 2, f"Expected logits to have at least 2 dimensions, but got {logits.dim()}"
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
    
    logger.debug(f"Entropy shape: {entropy.shape}, Varentropy shape: {varentropy.shape}")
    return entropy, varentropy

def calculate_metrics(logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
    logger.debug(f"calculate_metrics input shapes - logits: {logits.shape}, attention_scores: {attention_scores.shape}")
    assert logits.dim() >= 2, f"Expected logits to have at least 2 dimensions, but got {logits.dim()}"
    assert attention_scores.dim() >= 4, f"Expected attention_scores to have at least 4 dimensions, but got {attention_scores.dim()}"
    
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)
    
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
    attn_varentropy = torch.var(attn_entropy, dim=1)
    
    mean_attention = torch.mean(attention_probs, dim=1)
    agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))
    
    interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))
    
    metrics = {
        "logits_entropy": torch.mean(entropy),
        "logits_varentropy": torch.mean(varentropy),
        "attn_entropy": torch.mean(attn_entropy),
        "attn_varentropy": torch.mean(attn_varentropy),
        "agreement": torch.mean(agreement),
        "interaction_strength": interaction_strength
    }
    
    logger.debug(f"Calculated metrics: {metrics}")
    return metrics

def sample_top_p_top_k(logits, temperature, top_p, top_k, min_p):
    logger.debug(f"sample_top_p_top_k input shape: {logits.shape}")
    logger.debug(f"Sampling parameters: temperature={temperature}, top_p={top_p}, top_k={top_k}, min_p={min_p}")
    
    assert logits.dim() == 2, f"Expected logits to have 2 dimensions, but got {logits.dim()}"
    vocab_size = logits.size(-1)
    logger.debug(f"Vocabulary size: {vocab_size}")
    
    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
    
    if min_p > 0.0:
        logger.debug("Applying min_p sampling")
        p_max = torch.max(probs, dim=-1, keepdim=True).values
        probs = torch.where(probs < (min_p * p_max), torch.zeros_like(probs), probs)
    
    k = min(top_k, vocab_size)
    logger.debug(f"Effective top-k: {k}")
    
    top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)
    logger.debug(f"Top-k probs shape: {top_k_probs.shape}, indices shape: {top_k_indices.shape}")
    
    cumulative_probs = torch.cumsum(top_k_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    top_k_probs[sorted_indices_to_remove] = 0.0
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    
    sample_probs = torch.distributions.Categorical(top_k_probs).sample()
    sample = torch.gather(top_k_indices, -1, sample_probs.unsqueeze(-1))
    
    logger.debug(f"Sampled token: {sample.item()}")
    return sample

def sample(gen_tokens: torch.Tensor, logits: torch.Tensor, attention_scores: torch.Tensor, cfg: SamplerConfig) -> torch.Tensor:
    logger.debug("Starting sampling process")
    logger.debug(f"gen_tokens shape: {gen_tokens.shape}")
    logger.debug(f"logits shape: {logits.shape}")
    logger.debug(f"attention_scores shape: {attention_scores.shape}")
    
    assert gen_tokens.dim() == 2, f"Expected gen_tokens to have 2 dimensions, but got {gen_tokens.dim()}"
    assert logits.dim() == 2, f"Expected logits to have 2 dimensions, but got {logits.dim()}"
    assert attention_scores.dim() >= 4, f"Expected attention_scores to have at least 4 dimensions, but got {attention_scores.dim()}"
    
    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]
    
    logger.debug(f"Entropy: {ent.item()}, Varentropy: {vent.item()}")
    logger.debug(f"Attention Entropy: {attn_ent.item()}, Attention Varentropy: {attn_vent.item()}")
    logger.debug(f"Agreement: {agreement.item()}, Interaction Strength: {interaction_strength.item()}")

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if ent < cfg.low_ent_thresh and vent < cfg.low_vent_thresh:
        logger.debug("Low Entropy, Low Varentropy: Choosing argmax")
        return torch.argmax(logits, dim=-1, keepdim=True)

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif ent > cfg.high_ent_thresh and vent < cfg.low_vent_thresh:
        logger.debug("High Entropy, Low Varentropy: Considering clarifying question")
        if not torch.isin(gen_tokens[:, -1], cfg.clarifying_question_token).any():
            logger.debug("Inserting clarifying question token")
            return torch.tensor([[cfg.clarifying_question_token]], device=logits.device)
        else:
            logger.debug("Sampling with higher temperature after clarifying question")
            temp_adj = cfg.helv_attn_ent_offset + cfg.helv_attn_ent_coef * attn_ent
            return sample_top_p_top_k(logits, min(1.5, cfg.temp * temp_adj), cfg.top_p, cfg.top_k, cfg.min_p)

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif ent < cfg.high_ent_thresh and vent > cfg.high_vent_thresh:
        logger.debug("Low Entropy, High Varentropy: Exploring forks")
        temp_adj = cfg.lehv_interaction_strength_offset + cfg.lehv_interaction_strength_coef * interaction_strength
        top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))
        return sample_top_p_top_k(logits, min(1.5, cfg.temp * temp_adj), cfg.top_p, top_k_adj, cfg.min_p)

    # High Entropy, High Varentropy: "resampling in the mist"
    elif ent > cfg.med_ent_thresh and vent > cfg.high_vent_thresh:
        logger.debug("High Entropy, High Varentropy: Resampling in the mist")
        temp_adj = cfg.hehv_attn_vent_offset + cfg.hehv_attn_vent_coef * attn_vent
        top_p_adj = max(0.5, cfg.top_p - cfg.hehv_attn_ent_coef * attn_ent)
        return sample_top_p_top_k(logits, max(2.0, cfg.temp * temp_adj), top_p_adj, cfg.top_k, cfg.min_p)

    # Middle ground: use adaptive sampling
    else:
        logger.debug("Middle ground: Using adaptive sampling")
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

        logger.debug(f"Adaptive sampling parameters: temperature={temperature.item()}, top_p={top_p.item()}, top_k={top_k}, min_p={min_p.item()}")

        samples = []
        for _ in range(cfg.n_adaptive_samples):
            sample = sample_top_p_top_k(logits, temperature, top_p, top_k, min_p)
            samples.append(sample)

        def score_sample(sample):
            log_prob = torch.sum(torch.nn.functional.log_softmax(logits, dim=-1) * torch.nn.functional.one_hot(sample, logits.shape[-1]))
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
        logger.debug(f"Chose best sample with score {sample_scores[best_sample_idx].item()}")
        return samples[best_sample_idx]