import io
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from PIL import Image
from pathlib import Path
import imageio
from tqdm import tqdm

# Class labels from dataset.json
CLASS_LABELS = {
    'modality': {
        0: "FLAIR",
        1: "T1w",
        2: "t1gd",
        3: "T2w"
    },
    'labels': {
        0: "background",
        1: "edema",
        2: "non-enhancing tumor",
        3: "enhancing tumour"
    }
}


def visualize_training_performance(log_dir="runs", output_dir="visualizations"):
    # Create output directories
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    timelines_dir = output_dir / "timelines"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    timelines_dir.mkdir(parents=True, exist_ok=True)

    # Collect all fold directories
    fold_dirs = [d for d in Path(log_dir).iterdir() if d.is_dir() and "fold_" in d.name]

    # Metric containers
    all_metrics = {
        'train_loss': [],
        'val_dice': [],
        'val_hd95': []
    }

    # Add progress bar for fold processing
    for fold_dir in tqdm(fold_dirs, desc="Processing folds", unit="fold"):
        fold_num = int(re.search(r"fold_(\d+)", fold_dir.name).group(1))
        print(f"\nProcessing fold {fold_num}")

        # Load TensorBoard data
        ea = event_accumulator.EventAccumulator(str(fold_dir))
        ea.Reload()

        # Get actual tag names with fold number prefix
        tags = ea.Tags()['scalars']
        train_loss_tag = f"Fold_{fold_num}/Train/Loss"
        val_dice_tag = f"Fold_{fold_num}/Validation/Dice/Mean"
        val_hd95_tag = f"Fold_{fold_num}/Validation/HD95/Mean"

        # Extract metrics
        metrics = {
            'train_loss': pd.DataFrame(ea.Scalars(train_loss_tag)) if train_loss_tag in tags else None,
            'val_dice': pd.DataFrame(ea.Scalars(val_dice_tag)) if val_dice_tag in tags else None,
            'val_hd95': pd.DataFrame(ea.Scalars(val_hd95_tag)) if val_hd95_tag in tags else None
        }

        # Skip if no metrics found
        if any(v is None or v.empty for v in metrics.values()):
            print(f"Skipping fold {fold_num} - missing metrics")
            continue

        # Store for aggregated plots
        for metric in all_metrics:
            if metrics[metric] is not None:
                # Convert 0-based steps to 1-based epochs
                adjusted_steps = metrics[metric]['step'].values + 1
                all_metrics[metric].append({
                    'values': metrics[metric]['value'].values,
                    'steps': adjusted_steps,  # Now contains actual epochs (4,8,12,16)
                    'fold': fold_num
                })

        # Plot individual fold metrics
        plot_fold_metrics(metrics, fold_num, metrics_dir)

        # Process timeline images
        process_timeline_images(fold_dir, timelines_dir, fold_num)

    # Create aggregated plots
    if all_metrics['train_loss'] and all_metrics['val_dice'] and all_metrics['val_hd95']:
        plot_aggregated_metrics(all_metrics, metrics_dir)


def plot_fold_metrics(metrics, fold_num, output_dir):
    fig, ax = plt.subplots(3, 1, figsize=(12, 15))

    # Training Loss
    ax[0].plot(metrics['train_loss']['step'], metrics['train_loss']['value'], label='Training Loss')
    ax[0].set_title(f'Fold {fold_num} - Training Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # Validation Dice
    ax[1].plot(metrics['val_dice']['step'], metrics['val_dice']['value'], label='Validation Dice')
    ax[1].set_title(f'Fold {fold_num} - Validation Dice Score')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Dice Score')
    ax[1].legend()

    # Validation HD95
    ax[2].plot(metrics['val_hd95']['step'], metrics['val_hd95']['value'], label='Validation HD95')
    ax[2].set_title(f'Fold {fold_num} - Validation HD95')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('HD95 (mm)')
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"fold_{fold_num}_metrics.png")
    plt.close()


def process_timeline_images(fold_dir, output_dir, fold_num):
    # Collect all visualization events
    ea = event_accumulator.EventAccumulator(str(fold_dir))
    ea.Reload()

    # Extract image events
    image_tags = [t for t in ea.Tags()['images'] if "Validation/Sample" in t]

    # Add progress bar for sample processing
    for tag in tqdm(image_tags, desc=f"Processing samples (fold {fold_num})", unit="sample"):
        sample_num = tag.split("/")[-1].split("_")[-1]
        images = ea.Images(tag)

        # Create output directory for this sample
        sample_dir = output_dir / f"fold_{fold_num}" / f"sample_{sample_num}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save individual frames and create GIF
        frames = []
        # Add progress bar for image processing within each sample
        for i, img in enumerate(tqdm(images, desc=f"Processing images (sample {sample_num})", leave=False)):
            try:
                # Convert tensorboard image data to PIL Image
                pil_img = Image.open(io.BytesIO(img.encoded_image_string))
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')

                # Save individual frame
                actual_epoch = (i + 1) * 4  # Validation every 4 epochs
                frame_path = sample_dir / f"epoch_{actual_epoch:03d}.png"
                pil_img.save(frame_path)

                # Create separated visualization
                create_separated_visualization(
                    np.array(pil_img), sample_dir, i,
                    pil_img.width, pil_img.height, fold_num
                )
            except Exception as e:
                print(f"\nError processing image {i} in fold {fold_num}, sample {sample_num}: {str(e)}")
                continue

        if frames:  # Only create GIF if we have frames
            # Create timeline GIF
            gif_path = sample_dir / f"timeline.gif"
            imageio.mimsave(gif_path, frames, duration=0.5)


def create_separated_visualization(arr, output_dir, epoch_idx, total_width, height, fold_num):
    # Calculate component widths (original image + 5 components)
    num_components = 7
    validate_freq = 4
    actual_epoch = (epoch_idx + 1) * validate_freq  # 4, 8, 12, 16 etc

    comp_width = total_width // num_components

    # Extract components
    components = {
        'input': arr[:, :comp_width],
        'ground_truth': arr[:, comp_width:2 * comp_width],
        'prediction': arr[:, 2 * comp_width:3 * comp_width],
        'prob_0': arr[:, 3 * comp_width:4 * comp_width],
        'prob_1': arr[:, 4 * comp_width:5 * comp_width],
        'prob_2': arr[:, 5 * comp_width:6 * comp_width],
        'prob_3': arr[:, 6 * comp_width:]
    }

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Plot each component with proper labels
    titles = [
        'Input Image (FLAIR)',
        'Ground Truth',
        'Prediction',
        f'Probability - {CLASS_LABELS["labels"][0]}',
        f'Probability - {CLASS_LABELS["labels"][1]}',
        f'Probability - {CLASS_LABELS["labels"][2]}',
        f'Probability - {CLASS_LABELS["labels"][3]}'
    ]

    for idx, (key, title) in enumerate(zip(components.keys(), titles)):
        axes[idx].imshow(components[key], cmap='gray' if idx > 2 else None)
        axes[idx].set_title(title)
        axes[idx].axis('off')

    # Hide unused axes
    for ax in axes[len(components):]:
        ax.axis('off')

    plt.suptitle(f"Fold {fold_num} - Epoch {actual_epoch} - Detailed Visualization")
    plt.tight_layout()
    plt.savefig(output_dir / f"separated_epoch_{actual_epoch:03d}.png")
    plt.close()


def plot_aggregated_metrics(all_metrics, output_dir):
    for metric_type, fold_data in all_metrics.items():
        if not fold_data:
            continue

        plt.figure(figsize=(10, 6))

        # Determine epoch steps (validation metrics are less frequent)
        is_validation = metric_type in ['val_dice', 'val_hd95']

        for fold in fold_data:
            steps = fold['steps']
            values = fold['values']
            plt.plot(steps, values, label=f'Fold {fold["fold"]}', alpha=0.7)

        # Configure plot
        title_map = {
            'train_loss': 'Training Loss',
            'val_dice': 'Validation Dice Score',
            'val_hd95': 'Validation HD95'
        }
        plt.title(f"{title_map[metric_type]} Across All Folds")
        plt.xlabel('Epoch')
        plt.ylabel(title_map[metric_type].split()[-1])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"aggregated_{metric_type}.png")
        plt.close()

    # Combined plot
    plt.figure(figsize=(12, 8))
    max_epochs = max(
        max(len(f['values']) for f in metric_data)
        for metric_data in all_metrics.values()
    )

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    for metric_type, fold_data in all_metrics.items():
        if not fold_data:
            continue

        # Use actual logged steps
        is_validation = metric_type in ['val_dice', 'val_hd95']
        all_steps = [f['steps'] for f in fold_data]
        max_len = max(len(s) for s in all_steps)

        padded_values = []
        for f in fold_data:
            pad_len = max_len - len(f['values'])
            padded = np.pad(f['values'], (0, pad_len), constant_values=np.nan)
            padded_values.append(padded)

        mean_values = np.nanmean(padded_values, axis=0)
        steps = all_steps[0]  # Assume all steps are same length/aligned for plotting
        label = metric_type.replace('_', ' ').title()
        linestyle = '--' if 'train' in metric_type else '-'

        if metric_type == 'train_loss':
            ax1.plot(steps, mean_values, label=label, linestyle=linestyle, color='tab:blue')
        elif metric_type == 'val_dice':
            ax1.plot(steps, mean_values, label=label, linestyle=linestyle, color='tab:orange')
        elif metric_type == 'val_hd95':
            ax2.plot(steps, mean_values, label=label, linestyle=linestyle, color='tab:green')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss / Dice Score', color='tab:blue')
    ax2.set_ylabel('HD95 (mm)', color='tab:green')

    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Handle combined legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title('Training and Validation Metrics')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "combined_metrics.png")
    plt.close()

if __name__ == "__main__":
    visualize_training_performance()