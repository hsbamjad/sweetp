"""
custom_modules.py — Custom Neural Network Modules for Sweet Potato Grading

Ported from chestnut_grading/models/custom_modules.py.
Adaptations for sweet potato (3-class, elongated morphology):
  - SPDConv: unchanged — spatial preservation is equally important
  - NIRDiffFusion: unchanged — NIR1/NIR2 spectral difference still encodes defects
  - ChestnutPriorHook: REMOVED — chestnuts are round; sweet potatoes are elongated
    and variable-shaped. An aspect-ratio prior would hurt performance here.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


# ─────────────────────────────────────────────────────────────────────────────
# 1. SPD-Conv: Space-to-Depth Convolutional Stem
# ─────────────────────────────────────────────────────────────────────────────

class SPDConv(nn.Module):
    """SPD-Conv: Space-to-Depth Convolutional layer.

    Replaces standard stride-2 convolutions to preserve fine-grained spatial
    details — critical for detecting surface defects on sweet potato skin.

    Ref: https://arxiv.org/abs/2208.03640
    """
    def __init__(self, c_in, c_out, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        # SPD layer: space-to-depth increases channels 4×, halves spatial dims
        self.conv = Conv(c1=c_in * 4, c2=c_out, k=k, s=s, p=p, g=g, act=act)

    def forward(self, x):
        # [B, C, H, W] → [B, 4C, H/2, W/2]
        return self.conv(torch.cat([
            x[..., ::2,  ::2],
            x[..., 1::2, ::2],
            x[..., ::2,  1::2],
            x[..., 1::2, 1::2],
        ], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 2. NIR-Diff Fusion: Learnable Spectral Embedding
# ─────────────────────────────────────────────────────────────────────────────

class NIRDiffFusion(nn.Module):
    """Learned spectral fusion for 3-channel NIR input (R, NIR1, NIR2).

    Internally computes NIR1-NIR2 difference, creating a 4-channel tensor
    [R, NIR1, NIR2, NIR1-NIR2], then learns a 4→3 channel fusion.

    This way the same R_NIR1_NIR2 dataset (3-ch) works for all models —
    no special 4-channel dataset is needed.

    Input:  [B, 3, H, W]  — R, NIR1, NIR2
    Output: [B, 3, H, W]  — spectral embedding (backbone sees 3-channel)
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 3):
        super().__init__()
        # internally uses 4 channels (3 raw + 1 computed diff)
        self.fusion = nn.Conv2d(4, out_ch, kernel_size=1, bias=False)

        # Initialize as near-passthrough:
        # R→R', NIR1→G', NIR2→B', diff→tiny influence on all
        with torch.no_grad():
            w = torch.zeros(out_ch, 4, 1, 1)
            w[0, 0] = 1.0    # R    → R'
            w[1, 1] = 1.0    # NIR1 → G'
            w[2, 2] = 1.0    # NIR2 → B'
            w[:, 3] = 0.05   # diff → learned from data
            self.fusion.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] — R, NIR1, NIR2
        diff = x[:, 1:2] - x[:, 2:3]          # NIR1 - NIR2
        x4   = torch.cat([x, diff], dim=1)     # [B, 4, H, W]
        return self.fusion(x4)

    def get_channel_weights(self) -> dict:
        """Return learned weights for inspection / paper reporting."""
        w = self.fusion.weight.data.squeeze()
        return {
            "R_to_R_prime":    w[0, 0].item(),
            "NIR1_to_G_prime": w[1, 1].item(),
            "NIR2_to_B_prime": w[2, 2].item(),
            "diff_to_R_prime": w[0, 3].item(),
            "diff_to_G_prime": w[1, 3].item(),
            "diff_to_B_prime": w[2, 3].item(),
        }


class NIRFusionHook:
    """Forward pre-hook: applies NIRDiffFusion before backbone layer[0].

    Only registered when enable_nir_fusion=True. When attached, always
    applies — no channel check needed because it's only added to
    Models 3 and 5 which always feed 3-ch R/NIR1/NIR2 input.
    """
    def __init__(self, fusion_module: NIRDiffFusion):
        self.fusion = fusion_module

    def __call__(self, module, inp):
        x      = inp[0]
        device = next(self.fusion.parameters()).device
        return (self.fusion(x.to(device)),)


# ─────────────────────────────────────────────────────────────────────────────
# NOTE: ChestnutPriorHook intentionally NOT ported here.
#
# The chestnut pipeline used a circular aspect-ratio prior because chestnuts
# are approximately round. Sweet potatoes are elongated and highly variable
# in shape — adding a shape prior would penalise valid detections.
# The YOLO detection head is used as-is (no shape bias).
# ─────────────────────────────────────────────────────────────────────────────
