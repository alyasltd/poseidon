import re
import csv

log_file = "/Users/alyazouzou/Desktop/pnp_yolo_nas/output_naspt_w_200 epoch.log"
csv_file = "metrics_without_coco.csv"

epoch_re = re.compile(r"SUMMARY OF EPOCH (\d+)")
metric_re = re.compile(r"([A-Za-z0-9_/\.]+)\s*=\s*([0-9\.eE+-]+)")

results = []
current_epoch = None
current_metrics = {}
section = None  # "train" ou "val"

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        # Nouveau bloc d’epoch
        m = epoch_re.search(line)
        if m:
            if current_epoch is not None:
                current_metrics["epoch"] = current_epoch
                results.append(current_metrics)
            current_epoch = int(m.group(1))
            current_metrics = {}
            section = None
            continue

        # Détection de section
        if "Train" in line:
            section = "train"
            continue
        if "Validation" in line:
            section = "val"
            continue

        # Extraction des métriques
        m = metric_re.search(line)
        if m:
            name, value = m.groups()
            val = value.strip().rstrip(".")
            try:
                val = float(val)
            except:
                continue

            name = name.lower()

            # mapping pour les losses détaillées
            if "loss_cls" in name:
                key = f"{section}_loss_cls"
            elif "loss_iou" in name:
                key = f"{section}_loss_iou"
            elif "loss_dfl" in name:
                key = f"{section}_loss_dfl"
            elif "loss_pose_cls" in name:
                key = f"{section}_loss_pose_cls"
            elif "loss_pose_reg" in name:
                key = f"{section}_loss_pose_reg"
            elif name.endswith("yolonasposeloss/loss"):
                key = f"{section}_loss"
            elif "ap_0.50" in name:
                key = "ap_0.50"
            elif "ap_0.75" in name:
                key = "ap_0.75"
            elif "ar_0.50" in name:
                key = "ar_0.50"
            elif "ar_0.75" in name:
                key = "ar_0.75"
            elif name.endswith("ap"):
                key = "ap"
            elif name.endswith("ar"):
                key = "ar"
            else:
                continue

            current_metrics[key] = val

# Sauvegarde du dernier epoch
if current_epoch is not None:
    current_metrics["epoch"] = current_epoch
    results.append(current_metrics)

# Colonnes fixes (train + val détaillé + metrics)
columns = [
    "epoch",
    "train_loss", "train_loss_cls", "train_loss_iou", "train_loss_dfl", "train_loss_pose_cls", "train_loss_pose_reg",
    "val_loss", "val_loss_cls", "val_loss_iou", "val_loss_dfl", "val_loss_pose_cls", "val_loss_pose_reg",
    "ap", "ar", "ap_0.50", "ap_0.75", "ar_0.50", "ar_0.75"
]

with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for row in results:
        writer.writerow({col: row.get(col, "") for col in columns})

print(f"✅ CSV détaillé enregistré dans {csv_file}")