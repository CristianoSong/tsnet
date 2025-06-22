"""
This module provides fine-tuning utilities for TSNet models.
"""

import torch
from torch import nn


def apply_minimal_finetuning(model: nn.Module):
    """
    Min-FT strategy: freeze attention and FFN layers while enabling gradients
    for embedding, normalization, and projection head layers.
    The filtering is based on actual module name fragments as printed by model.named_parameters().
    """
    for name, param in model.named_parameters():
        lname = name.lower()
        # Unfreeze embedding, norm, and head layers
        if (
            'embed' in lname
            or 'norm' in lname
            or 'head' in lname
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False


def apply_minimal_finetuning_alt(model: nn.Module):
    """
    Alternate Min-FT: explicitly freeze parameters in attention and FFN layers
    based on name fragments, matching the actual printed structure.
    This includes both encoder and decoder attention/ffn layers.
    """
    for name, param in model.named_parameters():
        lname = name.lower()
        if any(keyword in lname for keyword in ['attention', 'attn', 'ffn']):
            param.requires_grad = False
        elif any(keyword in lname for keyword in ['embed', 'norm', 'head']):
            param.requires_grad = True
        else:
            param.requires_grad = False