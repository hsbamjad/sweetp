"""
custom_yolo.py — SweetPotatoYOLO: Custom YOLO26 for 3-class Sweet Potato Grading

Ported from chestnut_grading/models/custom_yolo.py.

Key differences from ChestnutYOLO:
  - nc=3  (Normal / Moderate defect / Severe defect)
  - ChestnutPriorHook REMOVED (shape prior inappropriate for sweet potato)
  - Class name updated throughout (SweetPotatoYOLO / SweetPotatoTrainer)
  - 2026 BMP + 2024 JPG input both handled by OpenCV in data loader

Architecture:
  Modification 1: SPD-Conv stem  (layers [0, 1]) — preserves defect-scale detail
  Modification 2: NIRDiffFusion  (pre-hook)      — encodes NIR1-NIR2 difference

Usage:
    python models/train.py --config configs/model5/sweetpotato_yolo.yaml \\
                           --name model5_full --custom
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from train import MultispectralYOLO
from custom_modules import SPDConv, NIRDiffFusion, NIRFusionHook
from ultralytics.nn.modules import Conv
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import RANK, LOGGER


# ─────────────────────────────────────────────────────────────────────────────
# SweetPotatoTrainer — re-injects modifications after ultralytics model rebuild
# ─────────────────────────────────────────────────────────────────────────────

class SweetPotatoTrainer(SegmentationTrainer):
    """Custom trainer that survives ultralytics' model rebuild in get_model().

    ultralytics' SegmentationTrainer.get_model() rebuilds the model from the
    checkpoint YAML and reloads weights — discarding our SPD/NIR modifications.

    This subclass overrides get_model() to:
      1. Build a clean SegmentationModel with ch=3 (backbone always sees 3ch)
      2. Load compatible pretrained weights (skipping SPD-modified layers)
      3. Re-inject SPD-Conv and NIRDiffFusion in the correct order
    """

    def __init__(self, sp_cfg: dict, *args, **kwargs):
        self._sp_cfg     = sp_cfg
        self._nir_fusion = None
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        from ultralytics.utils.torch_utils import intersect_dicts

        nc = self.data['nc']
        print(f"\n[SweetPotatoTrainer] Building model (nc={nc}, ch=3)...", flush=True)

        model = SegmentationModel(
            cfg,
            nc=nc,
            ch=3,   # backbone always sees 3-channel (NIRDiffFusion maps 4→3)
            verbose=verbose and RANK == -1,
        )

        if weights:
            if isinstance(weights, nn.Module):
                csd = weights.float().state_dict()
            elif isinstance(weights, (str, Path)):
                ckpt = torch.load(weights, map_location='cpu', weights_only=False)
                csd  = ckpt['model'].float().state_dict() if 'model' in ckpt else ckpt.state_dict()
            else:
                csd = weights

            updated = intersect_dicts(csd, model.state_dict())
            model.load_state_dict(updated, strict=False)
            if verbose:
                LOGGER.info(f"Transferred {len(updated)}/{len(model.state_dict())} "
                            f"items from pretrained weights")

        cfg_ = self._sp_cfg

        # ── Inject SPD-Conv ────────────────────────────────────────────────
        if cfg_.get('enable_spd', True):
            print("[SweetPotatoTrainer] Injecting SPD-Conv stem...", flush=True)
            layers = model.model
            for idx in cfg_.get('spd_layers', [0, 1]):
                layer = layers[idx]
                if not isinstance(layer, Conv):
                    continue
                spd = SPDConv(c_in=layer.conv.in_channels,
                              c_out=layer.conv.out_channels, k=3)
                for attr in ('f', 'i', 'type'):
                    if hasattr(layer, attr):
                        setattr(spd, attr, getattr(layer, attr))
                layers[idx] = spd

        # ── Attach NIR-Diff Fusion ─────────────────────────────────────────
        if cfg_.get('enable_nir_fusion', True):
            print("[SweetPotatoTrainer] Attaching NIR-Diff Fusion hook...", flush=True)
            self._nir_fusion = NIRDiffFusion(in_ch=3, out_ch=3)
            model.nir_fusion = self._nir_fusion
            layer0 = model.model[0]
            layer0.register_forward_pre_hook(NIRFusionHook(self._nir_fusion))

        model.float()
        model.requires_grad_(True)

        print("[SweetPotatoTrainer] Model ready.\n", flush=True)
        return model


# ─────────────────────────────────────────────────────────────────────────────
# SweetPotatoYOLO
# ─────────────────────────────────────────────────────────────────────────────

class SweetPotatoYOLO(MultispectralYOLO):
    """Custom YOLO26 for sweet potato grading (3-class instance segmentation).

    Architectural novelties vs. standard YOLO26:
      1. SPD-Conv stem — preserves fine skin-defect detail lost by stride-2
      2. NIRDiffFusion — learns optimal blend of R / NIR1 / NIR2 / (NIR1-NIR2)

    Key difference from ChestnutYOLO:
      - No aspect-ratio head hook (sweet potatoes are elongated, not round)
      - nc=3 throughout

    Args:
        model_base     (str):   Base checkpoint. Default 'yolo26m-seg.pt'.
        model_size     (str):   YOLO size suffix. Default 'm'.
        nc             (int):   Number of classes. Default 3.
        enable_spd     (bool):  Inject SPD-Conv stem.
        enable_nir_fusion (bool): Prepend NIRDiffFusion (4-ch → 3-ch).
        alpha_init     (float): Unused (kept for API compatibility with train.py).
        spd_layers     (list):  Backbone indices to replace. Default [0, 1].
    """

    def __init__(
        self,
        model_base: str    = 'yolo26m-seg.pt',
        model_size: str    = 'm',
        nc: int            = 3,
        enable_spd: bool   = True,
        enable_nir_fusion: bool = True,
        alpha_init: float  = 0.1,   # kept for API compat, unused here
        spd_layers: list   = None,
    ):
        n_channels = 3  # always 3 — NIRDiffFusion computes diff internally

        print("\n" + "=" * 60)
        print("  SweetPotatoYOLO")
        print(f"  Input channels : {n_channels} (R, NIR1, NIR2)")
        print(f"  Classes (nc)   : {nc}")
        print(f"  SPD-Conv       : {enable_spd}")
        print(f"  NIR-Diff Fusion: {enable_nir_fusion} (diff computed internally)")
        print("=" * 60 + "\n")

        # Backbone always sees 3-channel (NIRDiffFusion maps 4→3 before layer[0])
        super().__init__(
            model_size   = model_size,
            n_channels   = 3,
            nc           = nc,
            model_base   = model_base,
        )

        self.enable_spd        = enable_spd
        self.enable_nir_fusion = enable_nir_fusion
        self.n_input_channels  = n_channels
        self._spd_layers       = spd_layers if spd_layers is not None else [0, 1]

        # Order matters: SPD first, then NIR hook (attaches to whatever is at layer[0])
        if enable_spd:
            self._inject_spd_conv()
        if enable_nir_fusion:
            self._attach_nir_diff_fusion()

        self.model.model.requires_grad_(True)
        self._print_summary()

    def _inject_spd_conv(self):
        layers = self.model.model.model
        for idx in self._spd_layers:
            layer = layers[idx]
            if not isinstance(layer, Conv):
                print(f"  Layer [{idx}] is {type(layer).__name__}, not Conv — skipped")
                continue
            spd = SPDConv(c_in=layer.conv.in_channels,
                          c_out=layer.conv.out_channels, k=3)
            for attr in ('f', 'i', 'type'):
                if hasattr(layer, attr):
                    setattr(spd, attr, getattr(layer, attr))
            layers[idx] = spd
            print(f"  SPDConv injected at layer [{idx}] "
                  f"({layer.conv.in_channels}->{layer.conv.out_channels}ch)")

    def _attach_nir_diff_fusion(self):
        self.nir_fusion = NIRDiffFusion(in_ch=3, out_ch=3)
        layer0 = self.model.model.model[0]

        def _pre_hook(module, args):
            x      = args[0]
            device = next(self.nir_fusion.parameters()).device
            return (self.nir_fusion(x.to(device)),)

        self._nir_hook_handle = layer0.register_forward_pre_hook(_pre_hook)
        print(f"  NIRDiffFusion attached on {type(layer0).__name__} (layer[0])")
        print(f"    Computes NIR1-NIR2 diff internally, learns 4->3 fusion")

    def _print_summary(self):
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        novel     = sum(p.numel() for p in self.nir_fusion.parameters()) \
                    if self.enable_nir_fusion else 0
        print(f"\n  Parameters: {total:,} total | {trainable:,} trainable")
        print(f"  Novel (NIRDiffFusion): {novel} params\n")

    def train(self, data_yaml: str, epochs: int = 150,
              imgsz: int = 640, batch: int = 16, **kwargs):
        """Train via SweetPotatoTrainer (modifications survive model rebuild)."""

        sp_cfg = {
            'enable_spd':       self.enable_spd,
            'enable_nir_fusion': self.enable_nir_fusion,
            'alpha_init':       kwargs.pop('alpha_init', 0.1),
            'spd_layers':       self._spd_layers,
        }

        train_args = {
            'data':        data_yaml,
            'epochs':      epochs,
            'imgsz':       imgsz,
            'batch':       batch,
            'device':      0 if torch.cuda.is_available() else 'cpu',
            'workers':     8,
            'patience':    50,
            'save':        True,
            'save_period': 10,
            'verbose':     True,
            'plots':       True,
            **kwargs,
        }

        results = self.model.train(
            trainer=lambda overrides, _callbacks: SweetPotatoTrainer(
                sp_cfg=sp_cfg,
                overrides=overrides,
                _callbacks=_callbacks,
            ),
            **train_args,
        )
        return results

    def get_nir_fusion_weights(self) -> dict:
        if not self.enable_nir_fusion:
            return {}
        return self.nir_fusion.get_channel_weights()


# ─────────────────────────────────────────────────────────────────────────────
# Quick forward-pass test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-base', default='yolo26m-seg.pt')
    parser.add_argument('--no-spd',     action='store_true')
    parser.add_argument('--no-nir',     action='store_true')
    args = parser.parse_args()

    model = SweetPotatoYOLO(
        model_base=args.model_base,
        nc=3,
        enable_spd=not args.no_spd,
        enable_nir_fusion=not args.no_nir,
    )

    n_ch = 4 if not args.no_nir else 3
    x    = torch.randn(1, n_ch, 640, 640)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.model.to(device)
    if hasattr(model, 'nir_fusion'):
        model.nir_fusion.to(device)
    x = x.to(device)

    model.model.eval()
    with torch.no_grad():
        out = model.model(x)

    print(f"\nForward pass OK — input: {tuple(x.shape)}")
    if isinstance(out, (list, tuple)):
        for i, o in enumerate(out):
            if isinstance(o, torch.Tensor):
                print(f"  Output [{i}]: {tuple(o.shape)}")
    if not args.no_nir:
        print(f"\nNIR-diff fusion weights:\n  {model.get_nir_fusion_weights()}")
