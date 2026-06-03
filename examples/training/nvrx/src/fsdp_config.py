"""
FSDP Configuration.

Returns FSDP-specific kwargs for ``FullyShardedDataParallel``.  Model-aware
``auto_wrap_policy`` is set for models that benefit from per-layer sharding
(e.g. LLaMA).  Called by ``distributed_utils.wrap_model`` when
``strategy="fsdp"``.
"""

import functools
import logging

from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

logger = logging.getLogger(__name__)


def get_fsdp_config(model_name):
    """Return a dict of FSDP kwargs suitable for the given *model_name*.

    For small models (GPT-2 124M) no auto-wrap policy is needed.  For larger
    transformer models (LLaMA, Mistral, etc.) we wrap at the decoder-layer
    level so that FSDP can shard each layer independently -- critical for
    fitting 7B+ models in GPU memory.
    """
    config = {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "cpu_offload": None,
        "limit_all_gathers": True,
        "use_orig_params": False,
    }

    if "gpt2" in model_name.lower():
        # GPT-2 (124M) is small enough to shard without per-layer wrapping.
        config["auto_wrap_policy"] = None

    elif "llama" in model_name.lower():
        try:
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer

            config["auto_wrap_policy"] = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer},
            )
            logger.info("FSDP: using LlamaDecoderLayer auto-wrap policy")
        except ImportError:
            logger.warning(
                "Could not import LlamaDecoderLayer -- falling back to no "
                "auto-wrap policy.  Install a recent transformers version."
            )
            config["auto_wrap_policy"] = None

    elif "mistral" in model_name.lower():
        try:
            from transformers.models.mistral.modeling_mistral import (
                MistralDecoderLayer,
            )

            config["auto_wrap_policy"] = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={MistralDecoderLayer},
            )
            logger.info("FSDP: using MistralDecoderLayer auto-wrap policy")
        except ImportError:
            config["auto_wrap_policy"] = None

    else:
        # Unknown model -- no auto-wrap.  Works for any model but may not be
        # memory-optimal for very large models.
        logger.info(
            f"FSDP: no model-specific auto-wrap policy for '{model_name}'. "
            "Using flat sharding."
        )
        config["auto_wrap_policy"] = None

    return config
