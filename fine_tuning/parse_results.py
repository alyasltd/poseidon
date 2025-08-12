import csv
from pathlib import Path

def parse_log_file(log_path, csv_path):
    rows = []
    all_metrics = set()

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_epoch = None
    current_data = {}
    section = None  # 'train' or 'val'

    for line in lines:
        line = line.strip()

        # Detect new epoch
        if line.startswith("SUMMARY OF EPOCH"):
            if current_epoch is not None:
                rows.append(current_data)
            current_epoch = int(line.split()[-1])
            current_data = {"epoch": current_epoch}
            section = None

        elif line.startswith("Train"):
            section = "train"
        elif line.startswith("Validation"):
            section = "val"

        # Match metrics lines like: Yolonasposeloss/loss_cls = 0.1761
        elif "=" in line and section:
            try:
                key, value = line.split("=")
                key = section + "_" + key.strip("├└│ ").strip()
                value = float(value.strip("├└│ ").strip())
                current_data[key] = value
                all_metrics.add(key)
            except ValueError:
                pass

    # Add last epoch
    if current_epoch is not None:
        rows.append(current_data)

    # Write CSV
    headers = ["epoch"] + sorted(all_metrics)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Parsed {len(rows)} epochs into {csv_path}")

if __name__ == "__main__":
    parse_log_file(
        "/home/aws_install/poseidon_prog/fine_tuning/output_naspt_w_200 epoch.log",
        "metrics_from_scratch.csv"
    )
