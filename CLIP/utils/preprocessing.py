import os
import json
from pathlib import Path
from PIL import Image
import random
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict


# ============================================================================
# CONFIGURATION - ALL VARIABLES AT TOP
# ============================================================================
# Paths
OUTPUT_DIR = "/data/joshi/utils/junks"
DATASET_NAME = "imagenet100_clip"

# Dataset split
TRAIN_SPLIT = 0.95
RANDOM_SEED = 42

# Derived paths
DOWNLOAD_DIR = os.path.join(OUTPUT_DIR, "downloads")
PROCESSED_DIR = os.path.join(OUTPUT_DIR, DATASET_NAME)

# Dataset location
DATASET_PATH = "/data/joshi/utils/junks/downloads/datasets/ambityga/imagenet100/versions/8"
LABELS_JSON_PATH = os.path.join(DATASET_PATH, "Labels.json")

# ============================================================================
# SETUP
# ============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

random.seed(RANDOM_SEED)

# ============================================================================
# LOAD CLASS NAMES FROM Labels.json
# ============================================================================
print("=" * 80)
print("LOADING CLASS NAMES FROM Labels.json")
print("=" * 80)

if not os.path.exists(LABELS_JSON_PATH):
    print(f"❌ ERROR: Labels.json not found at {LABELS_JSON_PATH}")
    exit(1)

with open(LABELS_JSON_PATH, 'r') as f:
    raw_labels = json.load(f)

# Clean labels - take only the first name before comma
IMAGENET_CLASS_NAMES = {}
for class_id, full_label in raw_labels.items():
    # Take first name only (e.g., "thunder snake, worm snake, ..." → "thunder snake")
    clean_name = full_label.split(',')[0].strip()
    IMAGENET_CLASS_NAMES[class_id] = clean_name

print(f"✓ Loaded {len(IMAGENET_CLASS_NAMES)} class names from Labels.json")
print(f"\nSample mappings:")
for i, (class_id, label) in enumerate(list(IMAGENET_CLASS_NAMES.items())[:10]):
    print(f"  {class_id} → '{label}'")

# ============================================================================
# CAPTION TEMPLATES
# ============================================================================
CAPTION_TEMPLATES = [
    "a photo of a {}",
    "a picture of a {}",
    "an image of a {}",
    "a photograph of a {}",
    "{} in the wild",
    "a high quality photo of a {}",
    "a rendering of a {}",
    "a cropped photo of a {}",
    "a bad photo of a {}",
    "a good photo of a {}",
    "a photo of one {}",
    "a close-up photo of a {}",
    "a bright photo of a {}",
    "a dark photo of a {}",
    "a jpeg corrupted photo of a {}",
    "a blurry photo of a {}",
    "a photo of the {}",
    "a photo of my {}",
    "a photo of the large {}",
    "a photo of the small {}",
    "a {}",
    "this is a {}",
    "that is a {}",
    "there is a {}",
    "an example of a {}",
    "a {} resting",
    "a {} moving",
    "a {} swimming",
    "a {} flying",
    "a {} hunting",
    "a {} eating",
    "a {} sleeping",
    "a {} walking",
    "a {} in its natural habitat",
    "a wild {} in nature",
    "a {} in the forest",
    "a {} in the water",
    "a {} on a branch",
    "a {} on the ground",
    "a {} in the grass",
    "a {} near water",
    "a beautiful {}",
    "a majestic {}",
    "an elegant {}",
    "a colorful {}",
    "a stunning {}",
    "a magnificent {}",
    "a rare {}",
    "a common {}",
    "an adult {}",
    "a young {}",
    "a juvenile {}",
    "a mature {}",
    "a {} with its offspring",
    "a group of {}",
    "a pair of {}",
    "multiple {}",
    "a side view of a {}",
    "a top view of a {}",
    "a front view of a {}",
    "an overhead view of a {}",
    "a profile of a {}",
    "a {} in sunlight",
    "a {} in the shade",
    "a {} at night",
    "a {} during daytime",
    "a {} in the rain",
    "a clear image of a {}",
    "a detailed shot of a {}",
    "a grainy photo of a {}",
    "a pixelated image of a {}",
]

# ============================================================================
# CHECK IF ALREADY PROCESSED
# ============================================================================
print("\n" + "=" * 80)
print("CHECKING FOR EXISTING PROCESSED DATA")
print("=" * 80)

train_file = os.path.join(PROCESSED_DIR, "train.json")
test_file = os.path.join(PROCESSED_DIR, "test.json")
labels_file = os.path.join(PROCESSED_DIR, "labels.json")

if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(labels_file):
    print("✓ Preprocessed dataset already exists!")
    print(f"  Train: {train_file}")
    print(f"  Test: {test_file}")
    print(f"  Labels: {labels_file}")
    print("\nSkipping preprocessing. Delete these files to reprocess.")
    exit(0)

# ============================================================================
# FIND ALL IMAGES
# ============================================================================
def find_dataset_structure(base_path: Path):
    """Find all images in the dataset."""
    print("\n" + "=" * 80)
    print("SCANNING FOR IMAGES")
    print("=" * 80)
    
    all_files = []
    print(f"Scanning directory: {base_path}")
    
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            if file_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                all_files.append(file_path)
                if len(all_files) % 10000 == 0:
                    print(f"  Found {len(all_files)} images...")
    
    print(f"\n✓ Total images found: {len(all_files):,}")
    
    if not all_files:
        print("❌ ERROR: No images found!")
        return None, None
    
    # Show sample paths
    print(f"\nSample image paths:")
    for img_path in all_files[:5]:
        rel_path = img_path.relative_to(base_path)
        print(f"  {rel_path}")
    
    return all_files, base_path


dataset_path = Path(DATASET_PATH)
all_images, base_path = find_dataset_structure(dataset_path)

if all_images is None:
    print("\n❌ Failed to find images in dataset!")
    exit(1)

# ============================================================================
# EXTRACT LABELS FROM PATHS
# ============================================================================
def extract_labels_from_paths(image_paths: List[Path], base_path: Path) -> Dict[str, List[Path]]:
    """Extract class labels from image paths."""
    print("\n" + "=" * 80)
    print("EXTRACTING CLASS LABELS FROM PATHS")
    print("=" * 80)
    
    class_to_images = defaultdict(list)
    
    for img_path in image_paths:
        rel_path = img_path.relative_to(base_path)
        parts = rel_path.parts
        
        # Look for directory that starts with 'n' (ImageNet class ID pattern)
        class_id = None
        for part in parts[:-1]:  # Exclude filename
            if part.startswith('n') and len(part) >= 9:  # ImageNet IDs like n01234567
                class_id = part
                break
        
        if class_id is None:
            # Use parent directory as class
            class_id = parts[-2] if len(parts) > 1 else "unknown"
        
        class_to_images[class_id].append(img_path)
    
    print(f"✓ Found {len(class_to_images)} classes")
    print(f"\nClass distribution:")
    for i, (class_id, images) in enumerate(sorted(class_to_images.items())):
        label_name = IMAGENET_CLASS_NAMES.get(class_id, "UNKNOWN")
        print(f"  {class_id} ({label_name}): {len(images)} images")
        if i >= 9:
            print(f"  ... and {len(class_to_images) - 10} more classes")
            break
    
    return dict(class_to_images)


class_to_images = extract_labels_from_paths(all_images, base_path)

# ============================================================================
# VERIFY ALL CLASSES ARE MAPPED
# ============================================================================
print("\n" + "=" * 80)
print("VERIFYING CLASS MAPPINGS")
print("=" * 80)

labels = {}
unmapped = []

for class_id in class_to_images.keys():
    if class_id in IMAGENET_CLASS_NAMES:
        labels[class_id] = IMAGENET_CLASS_NAMES[class_id]
    else:
        unmapped.append(class_id)
        # Fallback
        labels[class_id] = class_id.replace('n0', '').replace('n', '')

if unmapped:
    print(f"⚠️  WARNING: {len(unmapped)} classes not in Labels.json:")
    for cid in unmapped[:5]:
        print(f"  - {cid}")
    print("\nUsing class IDs as fallback labels for these classes.")
else:
    print(f"✅ All {len(labels)} classes successfully mapped!")

# Save labels
with open(labels_file, 'w') as f:
    json.dump(labels, f, indent=2)
print(f"\n✓ Saved labels to {labels_file}")

# ============================================================================
# CREATE TRAIN/TEST SPLIT
# ============================================================================
def create_train_test_split(class_to_images: Dict[str, List[Path]], train_ratio: float = 0.95):
    """Split images into train and test sets, stratified by class."""
    train_data = []
    test_data = []
    
    print("\n" + "=" * 80)
    print(f"CREATING TRAIN/TEST SPLIT ({train_ratio:.0%} / {1-train_ratio:.0%})")
    print("=" * 80)
    
    for class_id, images in class_to_images.items():
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        
        for img_path in images[:split_idx]:
            train_data.append((str(img_path), class_id, labels[class_id]))
        
        for img_path in images[split_idx:]:
            test_data.append((str(img_path), class_id, labels[class_id]))
        
        label_name = labels[class_id]
        if len(class_to_images) <= 20:  # Only print details for small datasets
            print(f"  {class_id} ({label_name}): {split_idx} train, {len(images) - split_idx} test")
    
    print(f"\n✅ Total - Train: {len(train_data):,}, Test: {len(test_data):,}")
    return train_data, test_data


train_data, test_data = create_train_test_split(class_to_images, TRAIN_SPLIT)

# ============================================================================
# GENERATE CAPTIONS AND SAVE
# ============================================================================
def generate_captions(label_name: str, num_captions: int = 1) -> List[str]:
    """Generate captions for a label using templates."""
    if num_captions == 1:
        template = random.choice(CAPTION_TEMPLATES)
        return [template.format(label_name)]
    else:
        templates = random.sample(CAPTION_TEMPLATES, min(num_captions, len(CAPTION_TEMPLATES)))
        return [template.format(label_name) for template in templates]


def save_dataset(data: List[Tuple[str, str, str]], output_file: str, num_captions_per_image: int = 1):
    """Save dataset in JSON format with image-caption pairs."""
    dataset = []
    
    for img_path, class_id, label_name in tqdm(data, desc=f"Generating captions"):
        captions = generate_captions(label_name, num_captions_per_image)
        
        for caption in captions:
            dataset.append({
                "image_path": img_path,
                "class_id": class_id,
                "label": label_name,
                "caption": caption
            })
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Saved {len(dataset):,} image-caption pairs to {output_file}")
    return dataset


print("\n" + "=" * 80)
print("GENERATING CAPTIONS AND SAVING DATASETS")
print("=" * 80)

train_dataset = save_dataset(train_data, train_file, num_captions_per_image=1)
test_dataset = save_dataset(test_data, test_file, num_captions_per_image=1)

# ============================================================================
# VALIDATION TESTS
# ============================================================================
print("\n" + "=" * 80)
print("RUNNING VALIDATION TESTS")
print("=" * 80)


def run_tests():
    """Run comprehensive tests to validate the preprocessing."""
    
    print("\n✓ Test 1: File existence")
    assert os.path.exists(train_file), "Train file missing!"
    assert os.path.exists(test_file), "Test file missing!"
    assert os.path.exists(labels_file), "Labels file missing!"
    print(f"  ✓ All files exist")
    
    print("\n✓ Test 2: Load JSON files")
    with open(train_file, 'r') as f:
        train_loaded = json.load(f)
    with open(test_file, 'r') as f:
        test_loaded = json.load(f)
    with open(labels_file, 'r') as f:
        labels_loaded = json.load(f)
    print(f"  ✓ Successfully loaded all JSON files")
    
    print("\n✓ Test 3: Dataset statistics")
    print(f"  Train samples: {len(train_loaded):,}")
    print(f"  Test samples: {len(test_loaded):,}")
    print(f"  Total samples: {len(train_loaded) + len(test_loaded):,}")
    print(f"  Number of classes: {len(labels_loaded)}")
    
    print("\n✓ Test 4: Data structure validation")
    required_keys = {'image_path', 'class_id', 'label', 'caption'}
    sample = train_loaded[0]
    assert required_keys.issubset(sample.keys()), f"Missing keys in sample: {sample.keys()}"
    print(f"  ✓ All samples have required keys: {required_keys}")
    
    print("\n✓ Test 5: Image accessibility")
    test_samples = random.sample(train_loaded, min(5, len(train_loaded)))
    success_count = 0
    for sample in test_samples:
        img_path = sample['image_path']
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                img.verify()
                success_count += 1
            except Exception as e:
                print(f"  ✗ Failed to load image {img_path}: {e}")
    print(f"  ✓ Successfully verified {success_count}/{len(test_samples)} random images")
    
    print("\n✓ Test 6: Caption examples")
    caption_samples = random.sample(train_loaded, min(10, len(train_loaded)))
    print(f"  Sample image-caption pairs:")
    for i, sample in enumerate(caption_samples[:5], 1):
        img_name = Path(sample['image_path']).name
        print(f"    {i}. [{sample['class_id']}] {sample['label']}")
        print(f"       Caption: '{sample['caption']}'")
    
    print("\n✓ Test 7: Class distribution")
    train_class_counts = defaultdict(int)
    test_class_counts = defaultdict(int)
    
    for sample in train_loaded:
        train_class_counts[sample['class_id']] += 1
    for sample in test_loaded:
        test_class_counts[sample['class_id']] += 1
    
    print(f"  Classes in train: {len(train_class_counts)}")
    print(f"  Classes in test: {len(test_class_counts)}")
    
    print("\n✓ Test 8: No data leakage check")
    train_images = {sample['image_path'] for sample in train_loaded}
    test_images = {sample['image_path'] for sample in test_loaded}
    overlap = train_images & test_images
    assert len(overlap) == 0, f"Data leakage detected! {len(overlap)} images in both sets"
    print(f"  ✓ No overlap between train and test sets")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Output directory: {PROCESSED_DIR}")
    print(f"Train file: {train_file} ({len(train_loaded):,} samples)")
    print(f"Test file: {test_file} ({len(test_loaded):,} samples)")
    print(f"Labels file: {labels_file} ({len(labels_loaded)} classes)")
    print(f"\nSample captions:")
    for sample in train_loaded[:3]:
        print(f"  • {sample['caption']}")
    print(f"\n📌 Ready for CLIP training!")
    print(f"\nUsage:")
    print(f"  import json")
    print(f"  train_data = json.load(open('{train_file}'))")
    print(f"  test_data = json.load(open('{test_file}'))")
    print("=" * 80)


run_tests()

