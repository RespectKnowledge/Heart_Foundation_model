"""
UxLSTM (DINOv3+UxLSTM SSL) encoder + classification/regression heads for CVD tasks.

Adapted from the ConsisMAE version. Swaps the ConsisMAE backbone for the
BaseDINOv3UxLSTMTrainer_Stable Stage-1 *teacher* checkpoint, trained on
Dataset701_UKBB_MRI.

The pretrained network is a UXlstmBot:
    ResidualEncoder  ->  xLSTM bottleneck (applied to deepest skip)  ->  decoder

Only the encoder (+ the xLSTM bottleneck) is used here; the SSL/seg decoder is
dropped. A fresh LayerNorm+Linear head is attached on top of the pooled,
concatenated multi-scale encoder features -- exactly the same head design as the
ConsisMAE script.
"""

import torch
import torch.nn as nn

# Exact same backbone class + init the SSL trainer (BaseMAETrainerUxLSTM) uses.
from nnunetv2.nets.UxLSTMBot_3d import UXlstmBot
from nnunetv2.utilities.network_initialization import InitWeights_He

# ── paths ──────────────────────────────────────────────────────────────────────
CKPT_PATH = (
    "/mnt/all_data/Abdul/ssl_foundation/data/nnssl_results/"
    "Dataset701_UKBB_MRI/BaseDINOv3UxLSTMTrainer_Stable__nnsslPlans__onemmiso/"
    "fold_all/checkpoint_final_teacher.pth"
)

# Try to find the xLSTM bottleneck submodule on the bot by these attribute names.
# (Set USE_BOTTLENECK=False to use raw encoder skips only.)
USE_BOTTLENECK = True
_BOTTLENECK_ATTRS = ("mamba_layer", "xlstm", "lstm_layer", "bottleneck", "vil", "xlstm_layer")

SEP = "=" * 60


# ── build the pretrained UXlstmBot ──────────────────────────────────────────────
def build_uxlstm_bot(num_input_channels: int = 1, num_output_channels: int = 1):
    """
    Rebuild the exact UXlstmBot from BaseMAETrainerUxLSTM.get_uxlstm_bot so the
    teacher checkpoint loads cleanly.

    num_output_channels only sizes the (unused) seg decoder. The SSL MAE recon
    head used 1 channel, so keep it at 1 -> decoder.seg_layers load without
    shape mismatch (we drop the decoder anyway).
    """
    n_stages = 6
    bot = UXlstmBot(
        input_channels=num_input_channels,
        n_stages=n_stages,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=num_output_channels,
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )
    bot.apply(InitWeights_He(1e-2))
    return bot


def load_pretrained_bot(ckpt_path: str = CKPT_PATH, num_input_channels: int = 1):
    """Build a UXlstmBot and load the Stage-1 teacher encoder weights into it."""
    print(f"\n{SEP}")
    print("Loading pretrained UxLSTM (DINOv3+UxLSTM) teacher...")
    print(SEP)

    bot = build_uxlstm_bot(num_input_channels=num_input_channels)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        print(f"Checkpoint keys: {list(ckpt.keys())[:8]}")
        # nnssl DINOv3 teacher checkpoint stores the backbone under 'teacher';
        # the DINO/iBOT projection heads (t_cls_head/t_patch_head) are separate
        # and intentionally dropped here.
        state_dict = ckpt.get("teacher",
                     ckpt.get("network_weights",
                     ckpt.get("state_dict",
                     ckpt.get("model", ckpt))))
    else:
        state_dict = ckpt

    # strip common prefixes: DDP (module.), torch.compile (_orig_mod.),
    # and the DINOv3 teacher wrapper (backbone.)
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ("module.", "_orig_mod.", "backbone.", "teacher.", "student."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v

    missing, unexpected = bot.load_state_dict(cleaned, strict=False)
    n_enc_missing = sum(1 for k in missing if k.startswith("encoder.") or k.startswith("stem."))
    print(f"Missing keys    : {len(missing)}  (encoder-related: {n_enc_missing} -- want this ~0)")
    print(f"Unexpected keys : {len(unexpected)}  (decoder/projector placeholders -- expected)")
    print("Bot loaded.")
    return bot


# ── feature extractor over the bot encoder (+ xLSTM bottleneck) ─────────────────
class _UxLSTMBackbone(nn.Module):
    """Wraps bot.encoder (ResidualEncoder) + optional xLSTM bottleneck -> pooled embed."""

    def __init__(self, bot, use_bottleneck: bool = USE_BOTTLENECK):
        super().__init__()
        self.encoder = bot.encoder
        self.pool    = nn.AdaptiveAvgPool3d(1)

        self.bottleneck = None
        if use_bottleneck:
            for attr in _BOTTLENECK_ATTRS:
                if hasattr(bot, attr) and isinstance(getattr(bot, attr), nn.Module):
                    self.bottleneck = getattr(bot, attr)
                    print(f"Using xLSTM bottleneck: bot.{attr}")
                    break
            if self.bottleneck is None:
                print("No xLSTM bottleneck attr found -- using raw encoder skips only.")

        self.embed_dim = int(sum(self.encoder.output_channels))
        print(f"Encoder embedding dim: {self.embed_dim}")

    def forward(self, x):
        b = x.shape[0]
        skips = self.encoder(x)                 # list of [B, C_i, d, h, w]
        if self.bottleneck is not None:
            skips[-1] = self.bottleneck(skips[-1])
        embedding = torch.cat([self.pool(s) for s in skips], dim=1).reshape(b, -1)
        return embedding                        # [B, sum(output_channels)]


# ── heads ──────────────────────────────────────────────────────────────────────
class UxLSTMClassifier(nn.Module):
    """UxLSTM encoder + classification head for CVD (5 classes)."""

    def __init__(self, bot, num_classes: int = 5, freeze_encoder: bool = True):
        super().__init__()
        self.backbone = _UxLSTMBackbone(bot)

        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("Encoder frozen -- only classification head will be trained.")
        else:
            print("Full fine-tuning -- encoder + head will be trained.")

        self.clf_head = nn.Sequential(
            nn.LayerNorm(self.backbone.embed_dim),
            nn.Linear(self.backbone.embed_dim, num_classes),
        )

    def forward(self, x):
        return self.clf_head(self.backbone(x))


class UxLSTMRegressor(nn.Module):
    """UxLSTM encoder + regression head for continuous values (EF, BMI, etc.)."""

    def __init__(self, bot, freeze_encoder: bool = True):
        super().__init__()
        self.backbone = _UxLSTMBackbone(bot)

        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("Encoder frozen -- only regression head will be trained.")

        self.reg_head = nn.Sequential(
            nn.LayerNorm(self.backbone.embed_dim),
            nn.Linear(self.backbone.embed_dim, 1),
        )

    def forward(self, x):
        return self.reg_head(self.backbone(x))


# ── smoke test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # classification ------------------------------------------------------------
    print(f"\n{SEP}")
    print("Building Classification Model (5 CVD classes)")
    print(SEP)

    bot = load_pretrained_bot()
    clf_model = UxLSTMClassifier(bot, num_classes=5, freeze_encoder=True).to(device)

    n_total     = sum(p.numel() for p in clf_model.parameters())
    n_trainable = sum(p.numel() for p in clf_model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {n_total:,}")
    print(f"Trainable parameters : {n_trainable:,}  (head only)")
    print(f"Frozen parameters    : {n_total - n_trainable:,}  (encoder)")

    clf_model.eval()
    dummy = torch.randn(2, 1, 128, 128, 128, device=device)
    with torch.no_grad():
        logits = clf_model(dummy)
    probs = torch.softmax(logits, dim=1)
    classes = ["DCM", "HCM", "MINF", "NOR", "RV"]
    print(f"\nLogits shape : {list(logits.shape)}")
    for cls, prob in zip(classes, probs[0].tolist()):
        print(f"  {cls}: {prob:.4f}")

    # regression ----------------------------------------------------------------
    print(f"\n{SEP}")
    print("Building Regression Model (EF prediction)")
    print(SEP)

    bot2 = load_pretrained_bot()
    reg_model = UxLSTMRegressor(bot2, freeze_encoder=True).to(device)
    reg_model.eval()
    with torch.no_grad():
        pred_ef = reg_model(dummy)
    print(f"EF pred shape : {list(pred_ef.shape)}")
    print(f"EF pred values: {pred_ef.squeeze().tolist()}")

    print(f"\n{SEP}")
    print("Both models built and tested successfully!")
    print("  Classification: UxLSTM encoder -> Linear(embed->5) -> DCM/HCM/MINF/NOR/RV")
    print("  Regression    : UxLSTM encoder -> Linear(embed->1) -> EF value")
    print(SEP)
