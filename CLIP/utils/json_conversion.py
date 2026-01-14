import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_DIR = "/data/joshi/utils/junks/visualizations"
OUTPUT_FILE = "/data/joshi/utils/junks/visualizations/embeddings_data.json"
BALANCED_DATA_FILE = "/data/joshi/utils/junks/visualizations/balanced_data.json"

print("=" * 80)
print("CONVERTING NPY TO RICH JSON")
print("=" * 80)

image_emb_3d = np.load(Path(INPUT_DIR) / "image_embeddings_3d.npy")
text_emb_3d = np.load(Path(INPUT_DIR) / "text_embeddings_3d.npy")

print(f"✓ Loaded image embeddings: {image_emb_3d.shape}")
print(f"✓ Loaded text embeddings: {text_emb_3d.shape}")

with open(BALANCED_DATA_FILE, 'r') as f:
    balanced_data = json.load(f)

print(f"✓ Loaded balanced data: {len(balanced_data)} samples")

class_ids_list = [sample['class_id'] for sample in balanced_data]
unique_classes = sorted(set(class_ids_list))
n_classes = len(unique_classes)

colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
class_to_idx = {cls_id: i for i, cls_id in enumerate(unique_classes)}
class_to_color = {}

for cls_id in unique_classes:
    idx = class_to_idx[cls_id]
    rgb = colors[idx][:3]
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    class_to_color[cls_id] = hex_color

data = {
    "metadata": {
        "num_classes": n_classes,
        "num_samples": len(balanced_data),
        "embedding_dim": 3,
        "classes": {}
    },
    "points": []
}

class_names = {}
for sample in balanced_data:
    class_id = sample['class_id']
    if class_id not in class_names:
        class_names[class_id] = sample['label']

for class_id in unique_classes:
    data["metadata"]["classes"][class_id] = {
        "name": class_names[class_id],
        "color": class_to_color[class_id],
        "index": class_to_idx[class_id]
    }

for i, sample in enumerate(balanced_data):
    class_id = sample['class_id']
    class_idx = class_to_idx[class_id]
    
    data["points"].append({
        "id": f"img_{i}",
        "type": "image",
        "position": [float(image_emb_3d[i, 0]), float(image_emb_3d[i, 1]), float(image_emb_3d[i, 2])],
        "class_id": class_id,
        "class_name": class_names[class_id],
        "class_index": class_idx,
        "color": class_to_color[class_id],
        "image_path": sample['image_path'],
        "caption": sample['caption'],
        "label": sample['label']
    })
    
    data["points"].append({
        "id": f"text_{i}",
        "type": "text",
        "position": [float(text_emb_3d[i, 0]), float(text_emb_3d[i, 1]), float(text_emb_3d[i, 2])],
        "class_id": class_id,
        "class_name": class_names[class_id],
        "class_index": class_idx,
        "color": class_to_color[class_id],
        "image_path": sample['image_path'],
        "caption": sample['caption'],
        "label": sample['label']
    })

with open(OUTPUT_FILE, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n✓ Saved JSON: {OUTPUT_FILE}")
print(f"  File size: {Path(OUTPUT_FILE).stat().st_size / 1024 / 1024:.2f} MB")
print(f"  Total points: {len(data['points'])} ({len(balanced_data)} images + {len(balanced_data)} texts)")
print(f"  Classes: {n_classes}")

print("\n" + "=" * 80)
print("✅ CONVERSION COMPLETE!")
print("=" * 80)
print("\nJSON structure:")
print(f"  metadata.num_classes: {n_classes}")
print(f"  metadata.num_samples: {len(balanced_data)}")
print(f"  points[0].id: {data['points'][0]['id']}")
print(f"  points[0].position: {data['points'][0]['position']}")
print(f"  points[0].color: {data['points'][0]['color']}")
