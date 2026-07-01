"""
UxLSTMBrainTrainer
===================
nnU-Net trainer for cardiac (ACDC) segmentation, using UXlstmBot
(CNN encoder + xLSTM bottleneck + CNN decoder) initialized from
brain SSL Stage 1 pretraining (BaseDINOv3UxLSTMTrainer_Stable on
Dataset701_UKBB_MRI).

Only the encoder weights are loaded from the pretrained checkpoint
(strict=False). The decoder/segmentation head is freshly initialized
and trained from scratch on ACDC, since the SSL pretraining checkpoint's
decoder was a single-channel placeholder, not a meaningful segmentation head.

build_network_architecture is self-sufficient (no `self` access needed),
so it works correctly both at training time (called via self.build_network_architecture)
and at prediction time (called as trainer_class.build_network_architecture by
predict_from_raw_data.py, which has no trainer instance).

Author: Abdul Qayyum
"""

import inspect
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch import distributed as dist
from torch import GradScaler

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.logging.nnunet_logger import MetaLogger
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

sys.path.insert(0, '/home/aqayyum/Scar_Segmentation_models')
from nnunetv2.utilities.network_initialization import InitWeights_He

# UXlstmBot lives in a *different* "nnunetv2" package (under
# Scar_Segmentation_models) than the real installed nnunetv2 (from
# cardiac_downstream). The real nnunetv2 is already cached in sys.modules
# by the time this file loads, so plain imports always resolve to it.
# UxLSTMBot_3d.py also uses a relative import ("from .vision_lstm import ..."),
# which requires proper package context - a standalone file-path import
# (spec_from_file_location) breaks that. Instead, we temporarily swap
# sys.modules['nnunetv2'] to point at the Scar_Segmentation_models package,
# import what we need, then restore the original nnunetv2 immediately after.
import sys as _sys
import importlib as _importlib

_real_nnunetv2_modules = {
    name: mod for name, mod in _sys.modules.items()
    if name == "nnunetv2" or name.startswith("nnunetv2.")
}
for name in list(_real_nnunetv2_modules.keys()):
    del _sys.modules[name]

_old_sys_path = list(_sys.path)
_sys.path.insert(0, '/home/aqayyum/Scar_Segmentation_models')
try:
    _scar_uxlstm_module = _importlib.import_module("nnunetv2.nets.UxLSTMBot_3d")
    UXlstmBot = _scar_uxlstm_module.UXlstmBot
finally:
    # Restore: drop whatever got cached from Scar_Segmentation_models,
    # restore the real nnunetv2.* modules, restore sys.path.
    for name in list(_sys.modules.keys()):
        if name == "nnunetv2" or name.startswith("nnunetv2."):
            del _sys.modules[name]
    _sys.modules.update(_real_nnunetv2_modules)
    _sys.path = _old_sys_path


# ---------------------------------------------------------------------------
# Pretrained checkpoint location (brain SSL Stage 1, _Stable trainer)
# ---------------------------------------------------------------------------
BRAIN_UXLSTM_SSL_CKPT = (
    "/mnt/all_data/Abdul/ssl_foundation/data/nnssl_results/Dataset701_UKBB_MRI/"
    "BaseDINOv3UxLSTMTrainer_Stable__nnsslPlans__onemmiso/fold_all/"
    "checkpoint_final_teacher.pth"
)

# UXlstmBot is not under dynamic_network_architectures, so it can't be
# pydoc.locate'd from the plans file's network_class_name. We bypass that
# and call get_network_from_plans with this hardcoded import path instead.
_UXLSTM_BOT_IMPORT_PATH = "nnunetv2.nets.UxLSTMBot_3d.UXlstmBot"


def load_pretrained_encoder(model: nn.Module, ckpt_path: str):
    """
    Loads ONLY encoder/stem weights from a brain SSL teacher checkpoint
    (saved by BaseDINOv3UxLSTMTrainer_Stable.save_checkpoint) into a
    freshly built UXlstmBot. Decoder is left at random init.
    """
    if not ckpt_path:
        print("[UxLSTMBrain] No ckpt_path provided - training from random init.")
        return model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    teacher_sd = ckpt.get("teacher", ckpt)

    # Keys are saved as "backbone.encoder.*" / "backbone.decoder.*"
    # (UxLSTMFeatureExtractor.backbone -> UXlstmBot). Strip "backbone."
    # and keep only encoder/stem weights; skip decoder (placeholder, 1-channel).
    new_sd = {}
    for k, v in teacher_sd.items():
        if not k.startswith("backbone."):
            continue
        stripped = k[len("backbone."):]
        if stripped.startswith("encoder."):
            new_sd[stripped] = v
        # decoder.* intentionally skipped - placeholder weights from SSL,
        # not meaningful for a 4-class segmentation head.

    # Shape-safe filtering: drop any key whose checkpoint shape doesn't match
    # the freshly-built model's shape (e.g. anisotropic ACDC kernels vs
    # isotropic brain SSL kernels). strict=False alone only tolerates
    # missing/extra keys, not shape mismatches on keys present in both.
    model_sd = model.state_dict()
    compatible_sd = {}
    shape_mismatches = []
    for k, v in new_sd.items():
        if k not in model_sd:
            continue  # will show up as "unexpected" - shouldn't happen here
        if model_sd[k].shape == v.shape:
            compatible_sd[k] = v
        else:
            shape_mismatches.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    missing, unexpected = model.load_state_dict(compatible_sd, strict=False)
    print(f"[UxLSTMBrain] Checkpoint encoder tensors found: {len(new_sd)}")
    print(f"[UxLSTMBrain] Loaded (shape-compatible): {len(compatible_sd)} from {ckpt_path}")
    print(f"[UxLSTMBrain] Skipped (shape mismatch): {len(shape_mismatches)}")
    if shape_mismatches:
        print(f"[UxLSTMBrain] First few shape mismatches (ckpt_shape -> model_shape):")
        for k, ckpt_shape, model_shape in shape_mismatches[:10]:
            print(f"    {k}: {ckpt_shape} -> {model_shape}")
    print(f"[UxLSTMBrain] Missing keys (expected: decoder + seg head + skipped mismatches): {len(missing)}")
    print(f"[UxLSTMBrain] Unexpected keys (should be 0): {len(unexpected)}")
    if unexpected:
        print(f"[UxLSTMBrain] WARNING - unexpected keys found: {unexpected[:10]}")
    return model


def _build_uxlstm_bot(arch_init_kwargs, arch_init_kwargs_req_import,
                       num_input_channels, num_output_channels,
                       enable_deep_supervision=True):
    """
    Self-sufficient network builder. UXlstmBot is imported directly above
    (bypassing the nnunetv2 namespace collision), so we resolve only the
    conv_op/norm_op/nonlin string kwargs here (those are unambiguous
    standard-library paths, e.g. "torch.nn.modules.conv.Conv3d", and resolve
    fine via pydoc.locate) and construct UXlstmBot directly ourselves.
    """
    import pydoc
    kwargs = dict(arch_init_kwargs)
    for key in arch_init_kwargs_req_import:
        if kwargs.get(key) is not None:
            kwargs[key] = pydoc.locate(kwargs[key])

    network = UXlstmBot(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        **kwargs,
    )
    network.apply(InitWeights_He(1e-2))
    return network


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class UxLSTMHeartTrainer(nnUNetTrainer):
    """
    ACDC cardiac segmentation, UXlstmBot encoder pretrained via brain SSL
    (BaseDINOv3UxLSTMTrainer_Stable on Dataset701_UKBB_MRI).
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.is_ddp     = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()
        self.device     = device
        if self.device.type == 'cuda':
            self.device = torch.device(type='cuda', index=0)

        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        self.plans_manager         = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name    = configuration
        self.dataset_json          = dataset_json
        self.fold                  = fold

        self.preprocessed_dataset_folder_base = join(
            nnUNet_preprocessed, self.plans_manager.dataset_name)
        self.output_folder_base = join(
            nnUNet_results, self.plans_manager.dataset_name,
            self.__class__.__name__ + '__' +
            self.plans_manager.plans_name + '__' + configuration)
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')
        self.preprocessed_dataset_folder = join(
            self.preprocessed_dataset_folder_base,
            self.configuration_manager.data_identifier)

        self.batch_size     = self.configuration_manager.batch_size
        self.dataset_class  = None
        self.is_cascaded    = False
        self.folder_with_segs_from_previous_stage = None

        self.initial_lr    = 1e-3
        self.encoder_lr     = 1e-4
        self.weight_decay   = 5e-2
        self.oversample_foreground_percent  = 0.33
        self.num_iterations_per_epoch       = 250
        self.num_val_iterations_per_epoch   = 50
        self.warmup_epochs = 0
        self.num_epochs    = 1000
        self.current_epoch = 0
        self.enable_deep_supervision = True
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        self.num_input_channels = None
        self.network = None
        self.optimizer = self.lr_scheduler = None
        self.grad_scaler = GradScaler('cuda') if self.device.type == 'cuda' else None
        self.loss = None

        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder,
            'training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt' % (
                timestamp.year, timestamp.month, timestamp.day,
                timestamp.hour, timestamp.minute, timestamp.second))
        self.logger = MetaLogger(self.output_folder, False)
        self.dataloader_train = self.dataloader_val = None
        self._best_ema  = None
        self.inference_allowed_mirroring_axes = None
        self.save_every = 50
        self.disable_checkpointing = False
        self.was_initialized = False

    @staticmethod
    def build_network_architecture(architecture_class_name,
            arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels,
            enable_deep_supervision=True):
        # architecture_class_name from the plans file is IGNORED on purpose -
        # we always build UXlstmBot regardless of what the plans say, since
        # that's the architecture this trainer is designed around.
        return _build_uxlstm_bot(
            arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels,
            enable_deep_supervision,
        )

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = 1  # ACDC: single MRI channel

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
            ).to(self.device)

            load_pretrained_encoder(self.network, BRAIN_UXLSTM_SSL_CKPT)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("Trainer was already initialized.")


class UxLSTMHeartTrainer_NoPretrain(UxLSTMHeartTrainer):
    """Ablation: same architecture, random init (no SSL checkpoint loaded)."""

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = 1

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
            ).to(self.device)

            print("[UxLSTMBrain_NoPretrain] random init baseline - no checkpoint loaded.")

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("Trainer was already initialized.")
