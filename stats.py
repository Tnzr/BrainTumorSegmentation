from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import pandas as pd
import numpy as np
import re

log_dir = Path("runs")

# Metrics for WT (Whole Tumor)
METRICS = {
    "Dice": "WT",
    "HD95": "WT"
}

results = {metric: [] for metric in METRICS}

# Only use rank_0 folders (standard logging rank)
for fold_dir in sorted(log_dir.glob("fold_*_rank_0")):
    fold_match = re.search(r"fold_(\d+)", fold_dir.name)
    if not fold_match:
        continue
    fold_num = int(fold_match.group(1))
    print(f"[INFO] Processing Fold {fold_num} from {fold_dir.name}")

    ea = event_accumulator.EventAccumulator(str(fold_dir))
    try:
        ea.Reload()
    except Exception as e:
        print(f"[ERROR] Skipping {fold_dir.name}: {e}")
        continue

    tags = ea.Tags().get('scalars', [])

    for metric, subtag in METRICS.items():
        full_tag = f"Fold_{fold_num}/Validation/{metric}/{subtag}"
        if full_tag in tags:
            df = pd.DataFrame(ea.Scalars(full_tag))
            if not df.empty:
                last_val = df['value'].values[-1]
                results[metric].append(last_val)
            else:
                print(f"[WARNING] Tag present but empty: {full_tag}")
        else:
            print(f"[WARNING] Missing tag: {full_tag}")

# ----------------------------
# Print LaTeX-formatted Table
# ----------------------------
print("\n[RESULT] LaTeX Table:\n")
print("\\begin{tabular}{|l|c|c|}")
print("\\hline")
print("\\textbf{Metric} & \\textbf{Dice} & \\textbf{HD95 (mm)} \\\\ \\hline")

if results["Dice"] and results["HD95"]:
    dice_mean = np.mean(results["Dice"])
    dice_std = np.std(results["Dice"])
    hd95_mean = np.mean(results["HD95"])
    hd95_std = np.std(results["HD95"])

    print(f"Whole Tumor & {dice_mean:.2f} ± {dice_std:.2f} & {hd95_mean:.1f} ± {hd95_std:.1f} \\\\ \\hline")
else:
    print("Whole Tumor & N/A & N/A \\\\ \\hline")

print("\\end{tabular}")
