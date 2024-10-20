import os
from enum import Enum
import time
from loguru import logger
import torch

os.environ["PJRT_DEVICE"] = "TPU"

import torch.multiprocessing as mp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from optimum.tpu.modeling import AutoModelForCausalLM
from transformers import StaticCache

from .xla_mp_comm import AgentMailbox, RootMailbox


class ModelCommand(Enum):
    LEAVE = 0
    PREFILL = 1
    DECODE = 2


def _mp_fn(rank, model_id, root_mailbox: RootMailbox, sample_fn: callable):
    logger.info(f"[Rank {rank}] Starting _mp_fn")
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    mailbox = AgentMailbox(root_mailbox)

    logger.info(f"[Rank {rank}] Loading model")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.eval()
    model.to(device)
    logger.info(f"[Rank {rank}] Model loaded and moved to device")

    # Initialize static cache
    max_cache_length = 1024  # Adjust this value based on your requirements
    batch_size = 1  # Adjust based on your batch size
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_length,
        device=device,
        dtype=model.dtype,
    )

    def prefill(inputs):
        logger.info(f"[Rank {rank}] Starting prefill")
        model_inputs = {k: v.to(device) for k, v in inputs.items()}
        attention_mask = model_inputs['attention_mask']
        sequence_length = attention_mask.sum(dim=1)
        cache_position = torch.arange(sequence_length.max().item(), device=device)
        
        # Remove position_ids from model_inputs if it exists
        model_inputs.pop('position_ids', None)
        
        # Calculate pos_ids
        pos_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)

        logger.info(f"[Rank {rank}] Running model inference for prefill")
        outputs = model(
            **model_inputs,
            cache_position=cache_position,
            return_dict=False,
            use_cache=True,
            position_ids=pos_ids,
            past_key_values=past_key_values,
        )[0]
        xm.mark_step()

        if rank == 0:
            logger.info(f"[Rank {rank}] Sampling next token")
            next_token = sample_fn(outputs)
            xm.mark_step()
            logger.info(f"[Rank {rank}] Sending next token")
            mailbox.send((next_token.cpu(), sequence_length))

        logger.info(f"[Rank {rank}] Finished prefill")

    def decode(inputs, cache_position):
        logger.info(f"[Rank {rank}] Starting decode")
        cur_token = inputs['input_ids'].to(device)
        pos_ids = inputs['position_ids'].to(device)

        logger.info(f"[Rank {rank}] Running model inference for decode")
        outputs = model(
            cur_token,
            position_ids=pos_ids,
            cache_position=cache_position,
            return_dict=False,
            use_cache=True,
            past_key_values=past_key_values,
        )[0]
        xm.mark_step()

        if rank == 0:
            logger.info(f"[Rank {rank}] Sampling next token")
            next_token = sample_fn(outputs)
            xm.mark_step()
            logger.info(f"[Rank {rank}] Sending next token")
            mailbox.send(next_token.cpu())

        logger.info(f"[Rank {rank}] Finished decode")

    while True:
        if rank == 0:
            mailbox.agent_ready.set()
            logger.info(f"[Rank {rank}] Waiting for commands")
            mailbox.receive()
        xm.rendezvous("start")

        logger.info(f"[Rank {rank}] Received command")
        command, data = mailbox.command_data
        inputs = data[0] if data else None
        if command == ModelCommand.PREFILL:
            logger.info(f"[Rank {rank}] Executing PREFILL")
            prefill(inputs)
        elif command == ModelCommand.DECODE:
            logger.info(f"[Rank {rank}] Executing DECODE")
            cache_position = data[1]
            decode(inputs, cache_position)
        elif command == ModelCommand.LEAVE:
            logger.info(f"[Rank {rank}] Executing LEAVE")
            mailbox.agent_ready.set()
            break
    logger.info(f"[Rank {rank}] Exiting _mp_fn")


def model_loop_fn(*args):
    """Spawn processes in the TPUs forwarding arguments"""
    xmp.spawn(_mp_fn, args=(args), join=True, daemon=False)


class DistributedModel:
    def __init__(self, model_id: str, sample_fn: callable):
        logger.info(f"Initializing DistributedModel with model_id: {model_id}")
        start_time = time.time()
        manager = mp.Manager()
        self.mailbox = RootMailbox(manager)

        self.model_loop = mp.Process(target=model_loop_fn, args=(model_id, self.mailbox, sample_fn))
        self.model_loop.start()
        logger.info(f"DistributedModel initialization completed in {time.time() - start_time:.2f} seconds")

        self.cache_position = None
        self.sequence_length = None

    def prefill(self, **model_args):
        logger.info("Starting prefill operation")
        start_time = time.time()
        assert self.mailbox is not None, "DistributedModel is not initialized"
        result, self.sequence_length = self.mailbox.send(ModelCommand.PREFILL, model_args)[0]
        self.cache_position = self.sequence_length
        logger.info(f"Prefill operation completed in {time.time() - start_time:.2f} seconds")
        return result

    def decode(self, **model_args):
        logger.info("Starting decode operation")
        start_time = time.time()
        assert self.mailbox is not None, "DistributedModel is not initialized"
        model_args['position_ids'] = torch.tensor([[self.cache_position]], dtype=torch.long)
        result = self.mailbox.send(ModelCommand.DECODE, model_args, self.cache_position)[0]
        self.cache_position += 1
        logger.info(f"Decode operation completed in {time.time() - start_time:.2f} seconds")
        return result

    def leave(self):
        if self.mailbox is None:
            logger.info("DistributedModel already left")
            return
        logger.info("Initiating leave operation")
        start_time = time.time()
        self.mailbox.send(ModelCommand.LEAVE)
        logger.info("Joining model loop...")
        self.model_loop.join()
        logger.info(f"Model loop finished in {time.time() - start_time:.2f} seconds")
        self.mailbox = None

    def __del__(self):
        logger.info("DistributedModel destructor called")
        self.leave()