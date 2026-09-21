"""Bound native PyTorch delta-rule backward memory without truncating gradients."""
import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule


def checkpointed_delta_rule(query, key, value, g, beta, chunk_size=64,
                            initial_state=None, output_final_state=False,
                            use_qk_l2norm_in_kernel=False):
    # The native fallback clones intermediate matrices during its in-place loops.
    # Recompute bounded blocks instead of retaining them for the full sequence.
    block_size = 512
    if (query.shape[1] <= block_size or not torch.is_grad_enabled()
            or not any(x.requires_grad for x in (query, key, value, g, beta))):
        return torch_chunk_gated_delta_rule(
            query, key, value, g, beta, chunk_size, initial_state,
            output_final_state, use_qk_l2norm_in_kernel)
    if block_size % chunk_size:
        raise ValueError("delta-rule block must align with the native chunk size")

    def run_block(q, k, v, decay, weight, state):
        return torch_chunk_gated_delta_rule(
            q, k, v, decay, weight, chunk_size=chunk_size,
            initial_state=state, output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

    outputs = []
    state = initial_state
    for start in range(0, query.shape[1], block_size):
        pieces = [x[:, start:start + block_size] for x in (query, key, value, g, beta)]
        output, state = checkpoint(run_block, *pieces, state, use_reentrant=False)
        outputs.append(output)
        # Keep state attached: later blocks backpropagate through all earlier blocks.
    return torch.cat(outputs, dim=1), state if output_final_state else None


def configure_native_delta_rule(backbone):
    for module in backbone.modules():
        if getattr(module, "chunk_gated_delta_rule", None) is torch_chunk_gated_delta_rule:
            module.chunk_gated_delta_rule = checkpointed_delta_rule
