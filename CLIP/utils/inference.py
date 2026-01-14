import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import open_clip

from model import CLIPModel
from config import config


CHECKPOINT_PATH = "/data/joshi/utils/junks/best_model.pt"
DATA_PATH = "/data/joshi/utils/junks/imagenet100_clip/test.json"
LABELS_PATH = "/data/joshi/utils/junks/imagenet100_clip/labels.json"
OUTPUT_DIR = "/home/joshi/experiments/CLIP_reimplementation/viz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES_PER_CLASS = 30
MAX_CLASSES = 100

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_model(checkpoint_path, config, device):
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    
    model = CLIPModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded checkpoint from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Accuracy: {checkpoint['val_acc']*100:.2f}%")
    print(f"  Device: {device}")
    
    return model


@torch.no_grad()
def extract_embeddings(model, dataloader, device, max_samples=None):
    print("\n" + "=" * 80)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 80)
    
    image_embeddings = []
    text_embeddings = []
    
    model.eval()
    
    count = 0
    for images, texts in tqdm(dataloader, desc="Extracting"):
        images = images.to(device)
        texts = texts.to(device)
        
        image_features, text_features = model(images, texts)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        image_embeddings.append(image_features.cpu().numpy())
        text_embeddings.append(text_features.cpu().numpy())
        
        count += len(images)
        if max_samples and count >= max_samples:
            break
    
    image_embeddings = np.concatenate(image_embeddings, axis=0)
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    
    print(f"\n✓ Extracted embeddings:")
    print(f"  Image embeddings: {image_embeddings.shape}")
    print(f"  Text embeddings: {text_embeddings.shape}")
    
    return image_embeddings, text_embeddings


def create_balanced_dataset(data_path, labels_path, samples_per_class=20, max_classes=20):
    print("\n" + "=" * 80)
    print("CREATING BALANCED DATASET")
    print("=" * 80)
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    with open(labels_path, 'r') as f:
        all_labels = json.load(f)
    
    class_samples = {}
    for sample in data:
        class_id = sample['class_id']
        if class_id not in class_samples:
            class_samples[class_id] = []
        class_samples[class_id].append(sample)
    
    selected_classes = list(class_samples.keys())[:max_classes]
    
    balanced_data = []
    class_names = []
    
    for class_id in selected_classes:
        samples = class_samples[class_id][:samples_per_class]
        balanced_data.extend(samples)
        class_names.append(all_labels[class_id])
    
    print(f"✓ Created balanced dataset:")
    print(f"  Classes: {len(selected_classes)}")
    print(f"  Samples per class: {samples_per_class}")
    print(f"  Total samples: {len(balanced_data)}")
    print(f"\nSelected classes:")
    for i, (class_id, name) in enumerate(zip(selected_classes, class_names)):
        print(f"  {i+1}. {class_id}: {name}")
    
    return balanced_data, selected_classes, class_names


def reduce_dimensions(embeddings, method='pca', n_components=3):
    print(f"\n" + "=" * 80)
    print(f"DIMENSIONALITY REDUCTION ({method.upper()})")
    print("=" * 80)
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=RANDOM_SEED)
        reduced = reducer.fit_transform(embeddings)
        variance = reducer.explained_variance_ratio_
        print(f"✓ PCA complete")
        print(f"  Explained variance: {variance.sum()*100:.2f}%")
        print(f"  Per component: {variance * 100}")
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=RANDOM_SEED, perplexity=30)
        reduced = reducer.fit_transform(embeddings)
        print(f"✓ t-SNE complete")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reduced


def plot_3d_embeddings(image_emb_3d, text_emb_3d, class_ids, class_names, output_path, title="CLIP Embeddings (3D PCA)"):
    print(f"\n" + "=" * 80)
    print("CREATING 3D VISUALIZATION")
    print("=" * 80)
    
    fig = plt.figure(figsize=(20, 10))
    
    n_classes = len(set(class_ids))
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    class_to_color = {cls_id: colors[i] for i, cls_id in enumerate(sorted(set(class_ids)))}
    
    ax1 = fig.add_subplot(121, projection='3d')
    for i, class_id in enumerate(sorted(set(class_ids))):
        mask = np.array(class_ids) == class_id
        ax1.scatter(image_emb_3d[mask, 0], image_emb_3d[mask, 1], image_emb_3d[mask, 2],
                   c=[class_to_color[class_id]], label=class_names[i], alpha=0.6, s=50)
    
    ax1.set_xlabel('PC1', fontsize=12)
    ax1.set_ylabel('PC2', fontsize=12)
    ax1.set_zlabel('PC3', fontsize=12)
    ax1.set_title('Image Embeddings', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    ax2 = fig.add_subplot(122, projection='3d')
    for i, class_id in enumerate(sorted(set(class_ids))):
        mask = np.array(class_ids) == class_id
        ax2.scatter(text_emb_3d[mask, 0], text_emb_3d[mask, 1], text_emb_3d[mask, 2],
                   c=[class_to_color[class_id]], label=class_names[i], alpha=0.6, s=50, marker='^')
    
    ax2.set_xlabel('PC1', fontsize=12)
    ax2.set_ylabel('PC2', fontsize=12)
    ax2.set_zlabel('PC3', fontsize=12)
    ax2.set_title('Text Embeddings', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    return fig


def plot_combined_3d(image_emb_3d, text_emb_3d, class_ids, class_names, output_path):
    print(f"\nCreating combined 3D visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    n_classes = len(set(class_ids))
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    class_to_color = {cls_id: colors[i] for i, cls_id in enumerate(sorted(set(class_ids)))}
    
    for i, class_id in enumerate(sorted(set(class_ids))):
        mask = np.array(class_ids) == class_id
        ax.scatter(image_emb_3d[mask, 0], image_emb_3d[mask, 1], image_emb_3d[mask, 2],
                  c=[class_to_color[class_id]], label=f"{class_names[i]} (img)",
                  alpha=0.6, s=100, marker='o', edgecolors='black', linewidths=0.5)
    
    for i, class_id in enumerate(sorted(set(class_ids))):
        mask = np.array(class_ids) == class_id
        ax.scatter(text_emb_3d[mask, 0], text_emb_3d[mask, 1], text_emb_3d[mask, 2],
                  c=[class_to_color[class_id]], label=f"{class_names[i]} (txt)",
                  alpha=0.8, s=150, marker='^', edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
    ax.set_zlabel('PC3', fontsize=14, fontweight='bold')
    ax.set_title('Image-Text Embedding Alignment (3D PCA)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    return fig


def plot_2d_projections(image_emb_3d, text_emb_3d, class_ids, class_names, output_path):
    print(f"\nCreating 2D projection plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    
    n_classes = len(set(class_ids))
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    class_to_color = {cls_id: colors[i] for i, cls_id in enumerate(sorted(set(class_ids)))}
    
    projections = [(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')]
    
    for col, (dim1, dim2, label1, label2) in enumerate(projections):
        ax = axes[0, col]
        for i, class_id in enumerate(sorted(set(class_ids))):
            mask = np.array(class_ids) == class_id
            ax.scatter(image_emb_3d[mask, dim1], image_emb_3d[mask, dim2],
                      c=[class_to_color[class_id]], label=class_names[i], alpha=0.6, s=50)
        ax.set_xlabel(label1, fontsize=12, fontweight='bold')
        ax.set_ylabel(label2, fontsize=12, fontweight='bold')
        ax.set_title(f'Image Embeddings: {label1} vs {label2}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if col == 2:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        ax = axes[1, col]
        for i, class_id in enumerate(sorted(set(class_ids))):
            mask = np.array(class_ids) == class_id
            ax.scatter(text_emb_3d[mask, dim1], text_emb_3d[mask, dim2],
                      c=[class_to_color[class_id]], label=class_names[i], alpha=0.6, s=50, marker='^')
        ax.set_xlabel(label1, fontsize=12, fontweight='bold')
        ax.set_ylabel(label2, fontsize=12, fontweight='bold')
        ax.set_title(f'Text Embeddings: {label1} vs {label2}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if col == 2:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.suptitle('CLIP Embeddings - 2D PCA Projections', fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    return fig


def compute_similarity_stats(image_emb, text_emb, class_ids):
    print(f"\n" + "=" * 80)
    print("COMPUTING SIMILARITY STATISTICS")
    print("=" * 80)
    
    similarities = image_emb @ text_emb.T
    
    within_class_sims = []
    between_class_sims = []
    
    for i in range(len(class_ids)):
        for j in range(len(class_ids)):
            sim = similarities[i, j]
            if class_ids[i] == class_ids[j]:
                within_class_sims.append(sim)
            else:
                between_class_sims.append(sim)
    
    print(f"Within-class similarity: {np.mean(within_class_sims):.4f} ± {np.std(within_class_sims):.4f}")
    print(f"Between-class similarity: {np.mean(between_class_sims):.4f} ± {np.std(between_class_sims):.4f}")
    print(f"Separation: {np.mean(within_class_sims) - np.mean(between_class_sims):.4f}")
    
    return within_class_sims, between_class_sims


def export_for_react_three_fiber(image_emb_3d, text_emb_3d, balanced_data, class_ids_list, class_names, output_dir):
    print(f"\n" + "=" * 80)
    print("EXPORTING FOR REACT THREE FIBER")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    
    np.save(output_dir / "image_embeddings_3d.npy", image_emb_3d)
    np.save(output_dir / "text_embeddings_3d.npy", text_emb_3d)
    
    n_classes = len(set(class_ids_list))
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    
    class_to_idx = {cls_id: i for i, cls_id in enumerate(sorted(set(class_ids_list)))}
    class_to_color = {}
    for cls_id in sorted(set(class_ids_list)):
        idx = class_to_idx[cls_id]
        rgb = colors[idx][:3]
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        class_to_color[cls_id] = hex_color
    
    data = {
        "metadata": {
            "num_classes": n_classes,
            "num_samples": len(balanced_data),
            "samples_per_class": len(balanced_data) // n_classes,
            "embedding_dim": image_emb_3d.shape[1],
            "classes": {}
        },
        "points": []
    }
    
    for i, (class_id, class_name) in enumerate(zip(sorted(set(class_ids_list)), class_names)):
        data["metadata"]["classes"][class_id] = {
            "name": class_name,
            "color": class_to_color[class_id],
            "index": i
        }
    
    for i, sample in enumerate(balanced_data):
        class_id = sample['class_id']
        class_idx = class_to_idx[class_id]
        
        data["points"].append({
            "id": f"img_{i}",
            "type": "image",
            "position": image_emb_3d[i].tolist(),
            "class_id": class_id,
            "class_name": class_names[class_idx],
            "class_index": class_idx,
            "color": class_to_color[class_id],
            "image_path": sample['image_path'],
            "caption": sample['caption'],
            "label": sample['label']
        })
        
        data["points"].append({
            "id": f"text_{i}",
            "type": "text",
            "position": text_emb_3d[i].tolist(),
            "class_id": class_id,
            "class_name": class_names[class_idx],
            "class_index": class_idx,
            "color": class_to_color[class_id],
            "image_path": sample['image_path'],
            "caption": sample['caption'],
            "label": sample['label']
        })
    
    json_path = output_dir / "embeddings_data.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved embeddings data: {json_path}")
    print(f"  Total points: {len(data['points'])} ({len(balanced_data)} images + {len(balanced_data)} texts)")
    print(f"  Classes: {n_classes}")
    
    thumbnails_dir = output_dir / "thumbnails"
    thumbnails_dir.mkdir(exist_ok=True)
    
    print(f"\n✓ Creating thumbnails...")
    for i, sample in enumerate(tqdm(balanced_data, desc="Thumbnails")):
        try:
            img = Image.open(sample['image_path']).convert('RGB')
            img.thumbnail((128, 128))
            thumbnail_path = thumbnails_dir / f"img_{i}.jpg"
            img.save(thumbnail_path, quality=85)
        except Exception as e:
            print(f"  Warning: Failed to create thumbnail for {sample['image_path']}: {e}")
    
    print(f"✓ Saved {len(balanced_data)} thumbnails to {thumbnails_dir}")
    print("\n" + "=" * 80)
    print("✅ REACT THREE FIBER EXPORT COMPLETE!")
    print("=" * 80)


class BalancedCLIPDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)
        text = self.tokenizer([sample['caption']])
        return image, text.squeeze(0)


def main():
    import os
    from torchvision import transforms
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("CLIP EMBEDDING VISUALIZATION")
    print("=" * 80)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    
    balanced_data, selected_classes, class_names = create_balanced_dataset(
        DATA_PATH, LABELS_PATH, 
        samples_per_class=NUM_SAMPLES_PER_CLASS,
        max_classes=MAX_CLASSES
    )
    
    balanced_path = Path(OUTPUT_DIR) / "balanced_data.json"
    with open(balanced_path, 'w') as f:
        json.dump(balanced_data, f, indent=2)
    
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    dataset = BalancedCLIPDataset(balanced_data, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    class_ids_list = [sample['class_id'] for sample in balanced_data]
    
    model = load_model(CHECKPOINT_PATH, config, DEVICE)
    
    image_emb, text_emb = extract_embeddings(model, dataloader, DEVICE)
    
    np.save(Path(OUTPUT_DIR) / "image_embeddings.npy", image_emb)
    np.save(Path(OUTPUT_DIR) / "text_embeddings.npy", text_emb)
    print(f"\n✓ Saved embeddings to {OUTPUT_DIR}")
    
    image_emb_3d = reduce_dimensions(image_emb, method='pca', n_components=3)
    text_emb_3d = reduce_dimensions(text_emb, method='pca', n_components=3)
    
    plot_3d_embeddings(
        image_emb_3d, text_emb_3d, class_ids_list, class_names,
        Path(OUTPUT_DIR) / "embeddings_3d_separate.png",
        title="CLIP Image-Text Embeddings (3D PCA)"
    )
    
    plot_combined_3d(
        image_emb_3d, text_emb_3d, class_ids_list, class_names,
        Path(OUTPUT_DIR) / "embeddings_3d_combined.png"
    )
    
    plot_2d_projections(
        image_emb_3d, text_emb_3d, class_ids_list, class_names,
        Path(OUTPUT_DIR) / "embeddings_2d_projections.png"
    )
    
    within_sims, between_sims = compute_similarity_stats(image_emb, text_emb, class_ids_list)
    
    plt.figure(figsize=(10, 6))
    plt.hist(within_sims, bins=50, alpha=0.6, label='Within-class', color='green')
    plt.hist(between_sims, bins=50, alpha=0.6, label='Between-class', color='red')
    plt.xlabel('Cosine Similarity', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Image-Text Similarity Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "similarity_distribution.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {Path(OUTPUT_DIR) / 'similarity_distribution.png'}")
    
    export_for_react_three_fiber(image_emb_3d, text_emb_3d, balanced_data, class_ids_list, class_names, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated files in {OUTPUT_DIR}:")
    print(f"  1. embeddings_3d_separate.png - Side-by-side 3D plots")
    print(f"  2. embeddings_3d_combined.png - Combined 3D plot")
    print(f"  3. embeddings_2d_projections.png - 2D projections")
    print(f"  4. similarity_distribution.png - Similarity histogram")
    print(f"  5. image_embeddings.npy - Raw image embeddings")
    print(f"  6. text_embeddings.npy - Raw text embeddings")
    print(f"  7. balanced_data.json - Balanced dataset used")
    print(f"  8. embeddings_data.json - React Three Fiber data")
    print(f"  9. thumbnails/ - Thumbnail images")


if __name__ == "__main__":
    main()
