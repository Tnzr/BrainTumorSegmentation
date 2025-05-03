import datetime
import json
import os
import time
from pathlib import Path
import socket
from typing import Dict, List, Tuple, Optional
import monai
import numpy as np
import torch
import tqdm
from monai.data import DataLoader, Dataset, decollate_batch, NibabelReader
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd, AsDiscreted, Compose, EnsureChannelFirstd,
    EnsureTyped, LoadImaged, RandFlipd, RandRotate90d,
    ScaleIntensityd, MapTransform, RandCropByPosNegLabeld, SqueezeDimd, Lambdad
)
from sklearn.model_selection import KFold
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import argparse
import nibabel as nib
from monai.inferers import SlidingWindowInferer
from torch.utils.checkpoint import checkpoint
import torchvision.utils as vutils
import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai.metrics.utils")
from BrainTumorSegmentation import RemapLabelsd
from matplotlib import pyplot as plt


class Config:
    """Centralized configuration management"""
    def __init__(self):
        self.num_classes = 4
        self.in_channels = 4
        self.spatial_dims = 3
        self.base_channels = (16, 32, 64, 128, 256)
        self.strides = (2, 2, 2, 2)
        self.num_res_units = 2
        self.norm = "batch"
        self.lr = 1e-4
        self.grad_clip = 1.0
        self.amp = True


config = Config()


class FlexibleUNet(UNet):
    """Extendable UNet with configurable parameters"""

    def __init__(self, cfg: Config = config):
        super().__init__(
            spatial_dims=cfg.spatial_dims,
            in_channels=cfg.in_channels,
            out_channels=cfg.num_classes,
            channels=cfg.base_channels,
            strides=cfg.strides,
            num_res_units=cfg.num_res_units,
            norm=cfg.norm
        )

    def set_grad_checkpointing(self, mode=True):
        self.grad_checkpointing = mode

    def forward(self, x):
        if self.grad_checkpointing and self.training:
            return checkpoint(
                self.custom_forward,  # Use method reference instead of lambda
                x,
                use_reentrant=False
            )
        return super().forward(x)

    def custom_forward(self, x):
        """Explicit forward method for checkpointing"""
        return super(FlexibleUNet, self).forward(x)

class SmartPadTransform(MapTransform):
    """Dynamic padding/cropping transform to ensure exact target size"""
    def __init__(self, keys, target_size, mode="minimum", method="symmetric"):
        super().__init__(keys)
        self.target_size = target_size
        self.mode = mode
        self.method = method

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            current_spatial = img.shape[1:]  # CHWD format
            pad_width = []
            crop_slices = [slice(None)]  # Keep channel dimension

            for i, (current, target) in enumerate(zip(current_spatial, self.target_size)):
                if current < target:
                    # Calculate padding
                    diff = target - current
                    if self.method == "symmetric":
                        before = diff // 2
                        after = diff - before
                    else:
                        before = 0
                        after = diff
                    pad_width.append((before, after))
                    # No cropping needed
                    crop_slices.append(slice(0, target))
                elif current > target:
                    # Calculate center crop
                    start = (current - target) // 2
                    end = start + target
                    pad_width.append((0, 0))  # No padding
                    crop_slices.append(slice(start, end))
                else:
                    pad_width.append((0, 0))
                    crop_slices.append(slice(0, target))

            # Apply padding
            full_pad = [(0, 0)] + pad_width  # Add channel padding
            if self.mode == "minimum":
                constant = np.min(img)
                padded = np.pad(img, full_pad, mode="constant", constant_values=constant)
            else:
                padded = np.pad(img, full_pad, mode=self.mode)

            # Apply cropping if needed
            cropped = padded[tuple(crop_slices)]
            d[key] = cropped
        return d


class CustomSpatialPadd(monai.transforms.MapTransform):
    """Enhanced padding/cropping to ensure exact spatial dimensions"""

    def __init__(self, keys, spatial_size, mode="minimum", method="symmetric"):
        super().__init__(keys)
        self.keys = keys
        self.spatial_size = spatial_size
        self.mode = mode
        self.method = method

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            current_spatial = img.shape[1:]  # CHWD format
            pad_width = []
            crop_slices = [slice(None)]  # Keep channel dimension

            # Calculate padding/crop for each spatial dimension
            for i in range(3):
                target = self.spatial_size[i]
                current = current_spatial[i]

                if current < target:
                    # Padding needed
                    diff = target - current
                    before = diff // 2
                    after = diff - before
                    pad_width.append((before, after))
                    crop_slices.append(slice(None))  # No cropping
                elif current > target:
                    # Center cropping needed
                    start = (current - target) // 2
                    end = start + target
                    pad_width.append((0, 0))
                    crop_slices.append(slice(start, end))
                else:
                    pad_width.append((0, 0))
                    crop_slices.append(slice(None))

            # Apply padding
            full_pad = [(0, 0)] + pad_width  # Add channel padding
            if self.mode == "minimum":
                padded = np.pad(img, full_pad, mode="constant",
                                constant_values=np.min(img))
            else:
                padded = np.pad(img, full_pad, mode=self.mode)

            # Apply cropping
            final_img = padded[tuple(crop_slices)]
            d[key] = final_img

            # Debug shape verification
            assert d[key].shape[1:] == tuple(self.spatial_size), \
                f"{key} shape mismatch: {d[key].shape} vs target {self.spatial_size}"

        return d

class RemapLabelsd(MapTransform):
    """Label remapping transform"""

    def __init__(self, keys):
        super().__init__(keys)
        self.mapping = {4: 3}  # Map label 4→3

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            remapped = np.vectorize(lambda x: self.mapping.get(x, x))(label)
            if remapped.ndim == 3:
                remapped = remapped[None]  # Add channel dim
            d[key] = remapped
        return d

class SmartDataHandler:
    """Enhanced data handling with dynamic transform generation"""

    def __init__(self, dataset_path: str, max_samples: int = None):
        self.dataset_path = Path(dataset_path)
        self.max_samples = max_samples
        self.data = self._load_dataset()
        self.class_counts = self._compute_class_counts()

    def _load_dataset(self) -> Dict[str, List]:
        """Load and format dataset from JSON with max_samples support"""
        with open(self.dataset_path / "dataset.json", 'r') as f:
            dataset_json = json.load(f)

        training_data = [
            {
                "image": str(self.dataset_path / item["image"]),
                "label": str(self.dataset_path / item["label"])
            } for item in dataset_json["training"]
        ]

        if self.max_samples:
            training_data = training_data[:self.max_samples]

        return {"training": training_data}

    def create_transforms(self, mode: str = "train") -> Compose:
        """Generate transforms based on mode"""
        transforms = [
            LoadImaged(keys=["image", "label"], reader=NibabelReader),
            EnsureChannelFirstd(keys=["image"], channel_dim=-1),
            EnsureChannelFirstd(keys=["label"], strict_check=False),
            RemapLabelsd(keys=["label"]),
            ScaleIntensityd(keys=["image"]),
            SmartPadTransform(keys=["image", "label"], target_size=(240, 240, 160)),
            EnsureTyped(keys=["image"], dtype=torch.float32),  # Added for both modes
            EnsureTyped(keys=["label"], dtype=torch.long)  # Added for both modes
        ]
        if mode == "train":
            transforms.extend([
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key='label',
                    spatial_size=(128, 128, 128)),
                RandFlipd(keys=["image", "label"], prob=0.5),
                RandRotate90d(keys=["image", "label"], prob=0.5),
            ])
        return Compose(transforms)

    def _compute_class_counts(self):
        """Compute class distribution across entire dataset"""
        class_counts = np.zeros(4, dtype=np.int64)  # Now includes class 0
        for item in tqdm.tqdm(self.data["training"], desc="Calculating class weights"):
            label = nib.load(item["label"]).get_fdata()
            label[label == 4] = 3  # Remap ET (4→3)
            unique, counts = np.unique(label, return_counts=True)
            for u, c in zip(unique, counts):
                if u in [0, 1, 2, 3]:  # Include class 0
                    class_counts[int(u)] += c
        return class_counts


class SmartSegmenter:
    """Modular segmentation system with improved validation"""
    def __init__(self, rank: int, world_size: int, class_counts: np.ndarray):
        self.device = torch.device(f"cuda:{rank}")
        self.rank = rank  # Add this line to store rank
        self.model = self._init_model()
        # Dice loss handles one-hot targets (auto-converted)
        self.dice_loss = monai.losses.DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            weight=self._calculate_weights(class_counts[1:])  # Exclude background
        )
        # CrossEntropyLoss works with class indices directly
        self.ce_loss = torch.nn.CrossEntropyLoss(
            weight=self._calculate_weights(class_counts)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scaler = torch.amp.GradScaler(enabled=config.amp)

    def _init_model(self) -> DDP:
        """Proper unified model initialization"""
        model = FlexibleUNet(config)
        model.set_grad_checkpointing(True)
        model = model.to(self.device)
        return DDP(
            model,
            device_ids=[self.rank],
            output_device=self.device
        )

    def _calculate_weights(self, class_counts: np.ndarray) -> torch.Tensor:
        """Calculate class weights with proper dtype handling"""
        weights = 1.0 / (class_counts.astype(np.float32) + 1e-5)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def validate_step(self, val_loader: DataLoader, epoch: int, writer: SummaryWriter, current_fold_num: int):
        """Memory-optimized validation with sliding window inference"""
        self.model.eval()
        dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        hd_metric = HausdorffDistanceMetric(percentile=95, include_background=False, reduction="mean_batch")

        # Lightweight post-processing
        post_pred = Compose([
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True, to_onehot=4),
            EnsureTyped(keys="pred", dtype=torch.float16)  # FP16 to save memory
        ])

        post_label = Compose([
            SqueezeDimd(keys="label", dim=0),
            Lambdad(keys="label", func=lambda x: self.debug_transform(
                F.one_hot(x.long(), num_classes=4),
                "After one_hot"
            ).permute(3, 0, 1, 2)),  # Add batch dim
            EnsureTyped(keys="label", dtype=torch.float16)
        ])

        # Configure sliding window inference
        inferer = SlidingWindowInferer(
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            overlap=0.25,
            mode="gaussian",
            padding_mode="constant",
            device=self.device
        )

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config.amp):
            pbar = tqdm.tqdm(val_loader, desc=f"Fold {current_fold_num} Epoch {epoch + 1} Validation")
            for val_data in pbar:
                inputs = val_data["image"].to(self.device, non_blocking=True)
                labels = val_data["label"].to(self.device, non_blocking=True)

                # Raw output
                outputs = inferer(inputs, self.model).half()  # [B, C, H, W, D]

                # Save probability maps BEFORE argmax
                prob_maps = torch.softmax(outputs, dim=1)

                pred_dicts = decollate_batch({"pred": outputs})
                prob_dicts = decollate_batch({"pred": prob_maps})
                label_dicts = decollate_batch({"label": labels})

                # Process each sample separately to minimize memory
                for idx, (pred_dict, prob_dict, label_dict) in enumerate(zip(pred_dicts, prob_dicts, label_dicts)):
                    try:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float32, enabled=False):
                            label = post_label(label_dict)["label"].float()
                        hard_pred = AsDiscreted(keys="pred", argmax=True, to_onehot=4)(pred_dict)["pred"]
                        dice_metric(y_pred=hard_pred, y=label)
                        hd_metric(y_pred=hard_pred, y=label)
                        # Process prediction
                        pred = post_pred(pred_dict)["pred"]
                        # print(f"Prediction shape: {pred.shape}")

                        # Verify shapes match
                        assert pred.shape == label.shape, \
                            f"Shape mismatch: pred {pred.shape} vs label {label.shape}"

                        # Update metrics
                        # dice_metric(y_pred=pred, y=label)
                        # hd_metric(y_pred=pred, y=label)
                        if idx == 0 and self.rank == 0:
                            self.log_visuals(
                                writer,
                                inputs[:1],  # input image
                                label.unsqueeze(0),  # ground truth
                                prob_dict["pred"].unsqueeze(0),  # probability maps
                                epoch,
                                fold=current_fold_num
                            )
                        # Manual memory cleanup
                        del pred, label
                    finally:
                        del pred_dict, label_dict
                        torch.cuda.empty_cache()

                # Clean up batch tensors
                del inputs, labels, outputs
                gc.collect()
                torch.cuda.empty_cache()

        # Aggregate and log metrics
        dice_values = dice_metric.aggregate()
        hd_values = hd_metric.aggregate()

        # Handle multi-GPU logging properly
        if self.rank == 0:
            class_names = ["ET", "TC", "WT"]  # Should match 3 classes now
            for i, name in enumerate(class_names):
                writer.add_scalar(f"Fold_{current_fold_num}/Validation/Dice/{name}",
                                  dice_values[i].item(), epoch)
                writer.add_scalar(f"Fold_{current_fold_num}/Validation/HD95/{name}",
                                  hd_values[i].item(), epoch)
            # Log mean metrics
            writer.add_scalar(f"Fold_{current_fold_num}/Validation/Dice/Mean",
                              dice_values.mean().item(), epoch)
            writer.add_scalar(f"Fold_{current_fold_num}/Validation/HD95/Mean",
                              hd_values.mean().item(), epoch)
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        dice_metric.reset()
        hd_metric.reset()
        return dice_values.mean().item(), hd_values.mean().item()

    def _init_model(self) -> DDP:
        """Add gradient checkpointing to save memory"""
        model = FlexibleUNet(config)
        model.set_grad_checkpointing(True)  # Enable MONAI's checkpointing
        return DDP(model.to(self.device), device_ids=[self.rank])

    def debug_transform(self, tensor, msg):
        """Debugging helper for tensor transformations"""
        if self.rank == 0:
            print(f"{msg} - Shape: {tensor.shape}, Dtype: {tensor.dtype}")
        return tensor

    def log_visuals(self, writer: SummaryWriter, inputs, labels, predictions, epoch, fold, tag_prefix="Validation"):
        """
        Log input image, ground truth, prediction (argmax), and combined probability maps to TensorBoard.
        """
        def extract_center_slice(tensor):
            z = tensor.shape[-1] // 2
            return tensor[..., z]  # [B, ..., H, W]
        inputs = inputs.detach().cpu()
        labels = labels.detach().cpu()
        predictions = predictions.detach().cpu()
        for idx in range(min(2, inputs.shape[0])):
            image = extract_center_slice(inputs[idx, 0:1])  # [1, H, W]
            gt_onehot = extract_center_slice(labels[idx])  # [C, H, W]
            pred_probs = extract_center_slice(predictions[idx])  # [C, H, W]
            pred_mask = torch.argmax(pred_probs, dim=0)  # [H, W]
            gt_mask = torch.argmax(gt_onehot, dim=0)  # [H, W]
            # Normalize input image for visualization
            image_norm = (image - image.min()) / (image.max() - image.min() + 1e-5)
            # Normalize each class prob for visualization
            min_vals = pred_probs.view(pred_probs.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
            max_vals = pred_probs.view(pred_probs.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
            norm_probs = (pred_probs - min_vals) / (max_vals - min_vals + 1e-5)
            # Combine: [input, gt_mask, pred_mask, prob_0, prob_1, ..., prob_n]
            input_channel = image_norm  # [1, H, W]
            gt_channel = gt_mask.unsqueeze(0).float() / (gt_onehot.shape[0] - 1)  # normalize to [0,1]
            pred_channel = pred_mask.unsqueeze(0).float() / (pred_probs.shape[0] - 1)
            all_visuals = torch.cat([input_channel, gt_channel, pred_channel, norm_probs], dim=0)  # [C+3, H, W]
            # Make grid: 1 row (horizontal) showing input, gt, pred, and each class prob
            combined = vutils.make_grid(
                all_visuals.unsqueeze(1),  # [C+3, 1, H, W]
                nrow=all_visuals.shape[0], normalize=False, scale_each=False
            )  # [3, H, W * (num_visuals)]
            writer.add_image(f"{tag_prefix}/Sample_{idx}/VisualGrid", combined, epoch)

def find_free_port():
    """Find an available port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))  # OS assigns an available port
    port = s.getsockname()[1]
    s.close()
    return port


class DistributedTrainer:
    """Robust distributed training manager with cross-validation support"""

    def __init__(self, data_handler: SmartDataHandler, epochs: int = 20, num_folds: int = 5):
        self.data_handler = data_handler
        self.epochs = epochs
        self.num_folds = num_folds
        self.world_size = torch.cuda.device_count()
        self.kfold = KFold(n_splits=num_folds, shuffle=True)
        self.current_fold = 0  # Add tracking for current fold

    def create_loaders(self, train_files: List, val_files: List) -> Tuple[DataLoader]:
        """Create distributed data loaders"""
        train_ds = Dataset(train_files, transform=self.data_handler.create_transforms("train"))
        val_ds = Dataset(val_files, transform=self.data_handler.create_transforms("val"))

        train_sampler = DistributedSampler(train_ds, num_replicas=self.world_size, rank=self.rank)
        return (
            DataLoader(train_ds, batch_size=1, sampler=train_sampler, pin_memory=True),
            DataLoader(val_ds, batch_size=1, shuffle=False)
        )

    def setup_distributed(self, rank: int, port: int):
        """Initialize distributed training environment"""
        self.rank = rank
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank
        )
        torch.cuda.set_device(self.rank)

    def train_process(self, rank: int, port: int, validate_freq: int = 4):
        """Main training process handler for serial cross-validation with distributed batches"""
        self.setup_distributed(rank, port)
        all_indices = list(range(len(self.data_handler.data["training"])))
        fold_splits = list(self.kfold.split(all_indices))

        segmenter = SmartSegmenter(
            rank=rank,
            world_size=self.world_size,
            class_counts=self.data_handler.class_counts
        )

        # Run folds serially, distribute batches across GPUs
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            current_fold_num = fold_idx + 1
            if rank == 0:
                print(f"\n[INFO] Starting Fold {current_fold_num}/{self.num_folds}")

            train_files = [self.data_handler.data["training"][i] for i in train_idx]
            val_files = [self.data_handler.data["training"][i] for i in val_idx]

            train_loader, val_loader = self.create_loaders(train_files, val_files)

            log_dir = f"runs/fold_{current_fold_num}_rank_{rank}"
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)

            for epoch in range(self.epochs):
                train_sampler = train_loader.sampler
                train_sampler.set_epoch(epoch)
                self.train_epoch(segmenter, train_loader, epoch, writer, current_fold_num)

                if (epoch + 1) % validate_freq == 0:
                    self.validate_epoch(segmenter, val_loader, epoch, writer, current_fold_num)

            if dist.is_initialized():
                dist.barrier()  # sync before next fold

            writer.close()
            del writer

        if dist.is_initialized():
            dist.destroy_process_group()

    def train_single_fold(self, segmenter, fold_idx: int, train_idx: List[int], val_idx: List[int], current_fold_num: int):
        """Train and validate a single fold"""
        self.current_fold = current_fold_num # Update the instance variable if needed elsewhere

        train_files = [self.data_handler.data["training"][i] for i in train_idx]
        val_files = [self.data_handler.data["training"][i] for i in val_idx]

        train_loader, val_loader = self.create_loaders(train_files, val_files)

        # Ensure log directory uses the correct fold number
        log_dir = f"runs/fold_{current_fold_num}_rank_{self.rank}"
        os.makedirs(log_dir, exist_ok=True) # Ensure directory exists
        writer = SummaryWriter(log_dir=log_dir)
        print(f"Rank {self.rank} starting training for Fold {current_fold_num}. Train size: {len(train_files)}, Val size: {len(val_files)}")

        for epoch in range(self.epochs):
            # Pass current_fold_num if needed by train/validate_epoch for logging
            self.train_epoch(segmenter, train_loader, epoch, writer, current_fold_num)
            if (epoch + 1) % 2 == 0: # Validate every 2 epochs (adjust as needed)
                self.validate_epoch(segmenter, val_loader, epoch, writer, current_fold_num)

        if dist.is_initialized(): # Add this after all epochs complete for a fold:
            dist.barrier()
        writer.close() # Close writer for the fold

    def train_epoch(self, segmenter: SmartSegmenter, loader: DataLoader,
                    epoch: int, writer: SummaryWriter, current_fold_num: int):
        """Train one epoch"""
        segmenter.model.train()
        total_loss = 0.0
        loader.sampler.set_epoch(epoch) # Important for shuffling in DDP!

        # Add rank info to tqdm description for clarity
        desc = f"Fold {current_fold_num} Epoch {epoch + 1} Rank {self.rank}"
        pbar = tqdm.tqdm(loader, desc=desc) # Only show progress on rank 0

        for batch in pbar:
            inputs = batch["image"].to(segmenter.device, non_blocking=True)
            # *** Important Label Shape Correction ***
            # DiceCELoss with to_onehot_y=True expects labels of shape [B, H, W, D].
            # Your transforms result in [B, 1, H, W, D]. Need to squeeze the channel dim.
            labels = batch["label"].long().to(segmenter.device, non_blocking=True)  # Keep channel
            labels_ce = labels.squeeze(1)  # Remove channel for CE loss

            segmenter.optimizer.zero_grad(set_to_none=True) # More efficient zeroing

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config.amp):
                outputs = segmenter.model(inputs)
                dice_loss = segmenter.dice_loss(outputs, labels)  # Uses labels with channel
                ce_loss = segmenter.ce_loss(outputs, labels_ce)  # Uses labels without channel

                loss = dice_loss + ce_loss  # You can add weights if needed

            segmenter.scaler.scale(loss).backward()
            # Optional: Gradient clipping (before optimizer step)
            # segmenter.scaler.unscale_(segmenter.optimizer) # Needed before clip_grad_norm_
            # torch.nn.utils.clip_grad_norm_(segmenter.model.parameters(), config.grad_clip)
            segmenter.scaler.step(segmenter.optimizer)
            segmenter.scaler.update()

            # Aggregate loss across GPUs for accurate logging
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            total_loss += loss.item()

            if self.rank == 0:
                 pbar.set_postfix({"loss": loss.item()}) # Update tqdm postfix

        avg_loss = total_loss / len(loader)
        if self.rank == 0:
            print(f"Fold {current_fold_num} Epoch {epoch + 1} Avg Train Loss: {avg_loss:.4f}")
            writer.add_scalar(f"Fold_{current_fold_num}/Train/Loss", avg_loss, epoch)

    def validate_epoch(self, segmenter: SmartSegmenter, loader: DataLoader,
                       epoch: int, writer: SummaryWriter, current_fold_num: int):
        """Run validation"""
        # Pass current_fold_num to validate_step
        dice, hd95 = segmenter.validate_step(loader, epoch, writer, current_fold_num)
        if self.rank == 0:
            # Use current_fold_num in the print statement
            print(f"\nFold {current_fold_num} Epoch {epoch + 1} Validation:")
            print(f"Dice: {dice:.4f}, HD95: {hd95:.4f}")


def get_balanced_fold_assignment(fold_splits, world_size):
    # Create a list of list: one list per rank
    assigned_folds = [[] for _ in range(world_size)]
    for i, split in enumerate(fold_splits):
        assigned_folds[i % world_size].append((i, *split))
    return assigned_folds

def main():
    """Main execution flow with proper cleanup"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = argparse.ArgumentParser()
    dataset_path = "/media/tnzr/HDD1/Task01_BrainTumour"

    parser.add_argument("--data_path", type=str, default=dataset_path)
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()

    data_handler = SmartDataHandler(args.data_path, max_samples=32)
    trainer = DistributedTrainer(data_handler, args.epochs)

    try:
        port = find_free_port()
        mp.spawn(
            trainer.train_process,  # Correct method name
            args=(port,),
            nprocs=trainer.world_size,
            join=True
        )
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()