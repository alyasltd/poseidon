import csv
import json
from pathlib import Path

# we get the csv, images and output paths
csv_root = Path("/home/aws_install/data/csv")
image_base = Path("/home/aws_install/data/yolonas_pose_base/images")
output_json = "/home/aws_install/data/yolonas_pose_base/annotations/yolonas_pose_annotations.json"

# our annotation in yolo nas pose COCO format
json_data = {
    "info": {}, # Add any relevant info here
    "categories": [{
        "id": 0, # Category ID for runway
        "name": "runway", # Category name
        "keypoints": ["A", "B", "C", "D"], # Keypoint names 4 corners
        "skeleton": [[0, 1], [1, 2], [2, 3], [3, 0]] # Skeleton connections between keypoints (a square)
    }],
    "images": [], # List of images
    "annotations": [] # List of annotations
}

annotation_id = 1
image_id = 1

# Collect all CSV files recursively
csv_files = list(csv_root.rglob("*.csv"))
print(f"🔍 {len(csv_files)} CSV found in {csv_root}")

# Parse all CSV rows into memory, index by image name (basename only)
csv_data_index = {}
for csv_file in csv_files:
    try:
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                img_name = Path(row["image"]).name
                csv_data_index[img_name] = row
    except Exception as e:
        print(f"⚠️ Trouble Reading {csv_file.name}: {e}")

# Process train and val folders
for split in ["train", "val", "test"]:
    for ext in ["*.jpeg", "*.png"]:
        for image_path in (image_base / split).glob(ext):
            img_name = image_path.name

            if img_name not in csv_data_index:
                print(f"❌ No annotation found : {img_name}")
                continue

            row = csv_data_index[img_name]

            try:
                width = int(row["width"]) # in pixels
                height = int(row["height"])
                # Extract keypoints A, B, C, D coordinates in absolute pixel so each (x,y) is within the real image dimensions
                xA, yA = float(row["x_A"]), float(row["y_A"])
                xB, yB = float(row["x_B"]), float(row["y_B"])
                xC, yC = float(row["x_C"]), float(row["y_C"])
                xD, yD = float(row["x_D"]), float(row["y_D"])
            except Exception as e:
                print(f"⚠️ Missing or invalid format for {img_name}: {e}")
                continue
            
            # we switch the coordinates to match COCO format
            keypoints = [xA, yA, 2, xB, yB, 2, xC, yC, 2, xD, yD, 2] # Keypoints format: [x1, y1, v1, x2, y2, v2, ...] where v is visibility (2 = visible)
            x_coords = [xA, xB, xC, xD] # Collect x-coordinates of keypoints
            y_coords = [yA, yB, yC, yD] # Collect y-coordinates of keypoints
            # Calculate bounding box from keypoints
            # bbox format: [x,y, width, height] in absolute pixel coordinates
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            # add image to JSON
            json_data["images"].append({ 
                "id": image_id, # Unique image ID
                "file_name": str(image_path.relative_to(image_base)), # Relative path to the image
                "width": width, 
                "height": height
            })

            # add annotation
            json_data["annotations"].append({
                "id": annotation_id, # Unique annotation ID
                "image_id": image_id, # ID of the image this annotation belongs to
                "category_id": 0, # Category ID for runway
                "bbox": bbox, # Bounding box [x, y, width, height] 
                "keypoints": keypoints, # Keypoints in COCO format
                "num_keypoints": 4 # Number of keypoints (4 corners)
            })

            annotation_id += 1
            image_id += 1

# Save the JSON data to the output file
Path(output_json).parent.mkdir(parents=True, exist_ok=True)
with open(output_json, "w") as f:
    json.dump(json_data, f, indent=2)

print(f"✅ Export Finished : {output_json}")
print(f"🖼️ {len(json_data['images'])} images, 🧷 {len(json_data['annotations'])} annotations.")
