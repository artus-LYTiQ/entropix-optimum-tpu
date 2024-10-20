import os
import torch
from transformers import AutoTokenizer
from optimum.tpu.distributed_model import DistributedModel
from loguru import logger
import sys

# Remove default handler
logger.remove()

# Add a handler to write to file
logger.add("distributed_model.log", rotation="100 MB", level="DEBUG")

# Add a handler to write to stderr
logger.add(sys.stderr, level="INFO")

def sample_greedy(logits):
    next_logits = logits[:, -1]
    next_token_id = torch.argmax(next_logits, dim=-1)[:, None].int()
    return next_token_id

def _test_distributed_model_generation(model_id, max_new_tokens=20):
    print(f"Beginning test with model: {model_id}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text = ["Running something in parallel means"]
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tokens = input_ids.clone()

    print("Initializing DistributedModel...")
    model = DistributedModel(model_id, sample_greedy)
    
    print("Generating tokens...")
    pos_ids = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    next_token, sequence_length = model.prefill(input_ids=tokens, attention_mask=attention_mask)
    tokens = torch.cat([tokens, next_token], dim=-1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
    
    for _ in range(max_new_tokens - 1):  # -1 because we've already generated one token
        pos_ids = torch.tensor([[tokens.shape[1] - 1]], dtype=torch.long)
        next_token = model.decode(input_ids=next_token, attention_mask=torch.ones_like(next_token), position_ids=pos_ids)
        tokens = torch.cat([tokens, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        
        # Optional: Break if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break

    decoded_text = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    print("\n------------------------------------------")
    print("Generated text:")
    print(decoded_text[0])
    print("------------------------------------------")

if __name__ == "__main__":
    print("Script started")
    try:
        _test_distributed_model_generation("meta-llama/Llama-3.2-1B", max_new_tokens=5)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    print("Script completed")