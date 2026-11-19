"""Sink-token interventions at the point of pruning.

Each wraps an original PrunableClassifier. The original's unpruned prediction
still defines y_hat; only the pruned draws run through the intervention.
"""

from __future__ import annotations

import copy
from collections import defaultdict

import torch

from models.token_pruning.prunable_classifier import PreparedState, PrunableClassifier


class PrefixRemovedClassifier(PrunableClassifier):
    """Blocks before l see the [CLS] and register tokens; the pruned stream from block l has none.

    Needs a GAP-only probe, since no [CLS] token reaches the readout. Shares the original's weights.
    """

    def __init__(self, original: PrunableClassifier):
        if original.probe_readout != "gap" or not original.num_prefix_tokens:
            raise ValueError("Prefix removal needs a backbone with prefix tokens and a GAP-only probe.")
        super().__init__(original.backbone, original.head, original.probe_readout)

    def forward_kept(self, state: PreparedState, keep_mask: torch.Tensor) -> torch.Tensor:
        prefix_count = self.num_prefix_tokens
        # [B, P + N, C] -> [B, N, C]; the RoPE table already covers patch tokens only.
        patch_state = PreparedState(state.tokens[:, prefix_count:], state.positions, state.depth)
        # timm rotates all but the first num_prefix_tokens tokens: with no prefix left, rotate every token.
        attention = [block.attn for block in self.backbone.blocks[state.depth :]]
        self.num_prefix_tokens = 0
        for module in attention:
            module.num_prefix_tokens = 0
        try:
            return super().forward_kept(patch_state, keep_mask)
        finally:
            self.num_prefix_tokens = prefix_count
            for module in attention:
                module.num_prefix_tokens = prefix_count


def _move_to_register(units: torch.Tensor):
    """Forward pre-hook on an MLP output projection: move the listed hidden units onto the register token."""

    def hook(module, args):
        hidden = args[0].clone()  # [B, 1 + K, H] activations entering fc2; the register is token 0
        indices = units.to(hidden.device)
        largest = hidden[..., indices].amax(dim=1)  # [B, U]: maximum over tokens, per image and unit
        hidden[..., indices] = 0
        hidden[:, 0, indices] = largest
        return (hidden, *args[1:])

    return hook


class TestTimeRegisterClassifier(PrunableClassifier):
    """A prefix-free backbone (SigLIP 2) with one added register token that every draw keeps.

    The register enters as zeros with no position embedding. For each listed
    MLP neuron (block, unit), every token's activation is zeroed and the
    register takes their maximum (test-time registers, Jiang et al.). With no
    neurons this is the empty-register control. The register is excluded from
    the readout.
    """

    def __init__(self, original: PrunableClassifier, neurons: list[tuple[int, int]]):
        if original.num_prefix_tokens:
            raise ValueError("Test-time registers are added to backbones without prefix tokens.")
        # Copies, so the hooks never reach the original model, whose unpruned prediction defines y_hat.
        super().__init__(copy.deepcopy(original.backbone), copy.deepcopy(original.head), original.probe_readout)
        self.num_prefix_tokens = 1
        units_by_block = defaultdict(list)
        for block, unit in neurons:
            units_by_block[int(block)].append(int(unit))
        for block, units in units_by_block.items():
            self.backbone.blocks[block].mlp.fc2.register_forward_pre_hook(_move_to_register(torch.tensor(units)))

    def embed(self, images: torch.Tensor) -> PreparedState:
        state = super().embed(images)
        register = state.tokens.new_zeros(state.tokens.shape[0], 1, state.tokens.shape[-1])
        return PreparedState(torch.cat((register, state.tokens), dim=1), state.positions, 0)  # [B, N, C] -> [B, 1 + N, C]

    def global_token(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.backbone.global_token(tokens[:, 1:])  # attention-pool the patch tokens only
