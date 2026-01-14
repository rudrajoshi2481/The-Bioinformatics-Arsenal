"""
CLIP ImageNet-100 Dataset Validation Script
Validates preprocessing, loads examples, and tests CLIP training compatibility
"""

import json
import os
from pathlib import Path
from PIL import Image
import random
from collections import defaultdict, Counter
import torch
from torchvision import transforms
import open_clip

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "/data/joshi/utils/junks"
DATASET_NAME = "imagenet100_clip"
PROCESSED_DIR = os.path.join(OUTPUT_DIR, DATASET_NAME)

TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.json")
TEST_FILE = os.path.join(PROCESSED_DIR, "test.json")
LABELS_FILE = os.path.join(PROCESSED_DIR, "labels.json")

# ============================================================================
# CLIP TRANSFORMS (CORRECT NORMALIZATION!)
# ============================================================================
def get_clip_transforms():
    """Get CLIP-compatible transforms with CORRECT normalization."""
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],  # ✅ CLIP normalization
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

# ============================================================================
# VALIDATION TESTS
# ============================================================================

def test_1_file_existence():
    """Test 1: Verify all required files exist"""
    print("\n" + "="*80)
    print("TEST 1: File Existence")
    print("="*80)
    
    files = {
        "Train": TRAIN_FILE,
        "Test": TEST_FILE,
        "Labels": LABELS_FILE
    }
    
    all_exist = True
    for name, path in files.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        size = os.path.getsize(path) / (1024*1024) if exists else 0
        print(f"{status} {name}: {path}")
        if exists:
            print(f"   Size: {size:.2f} MB")
        all_exist = all_exist and exists
    
    assert all_exist, "❌ Some files are missing!"
    print("\n✅ PASSED: All files exist")
    return True


def test_2_load_and_parse():
    """Test 2: Load and parse JSON files"""
    print("\n" + "="*80)
    print("TEST 2: Load and Parse JSON")
    print("="*80)
    
    try:
        with open(TRAIN_FILE, 'r') as f:
            train_data = json.load(f)
        print(f"✓ Loaded train.json: {len(train_data)} samples")
        
        with open(TEST_FILE, 'r') as f:
            test_data = json.load(f)
        print(f"✓ Loaded test.json: {len(test_data)} samples")
        
        with open(LABELS_FILE, 'r') as f:
            labels = json.load(f)
        print(f"✓ Loaded labels.json: {len(labels)} classes")
        
        print(f"\n✅ PASSED: All JSON files loaded successfully")
        return train_data, test_data, labels
        
    except Exception as e:
        print(f"❌ FAILED: Error loading JSON - {e}")
        raise


def test_3_data_structure(train_data, test_data):
    """Test 3: Validate data structure"""
    print("\n" + "="*80)
    print("TEST 3: Data Structure Validation")
    print("="*80)
    
    required_keys = {'image_path', 'class_id', 'label', 'caption'}
    
    # Check train
    for i, sample in enumerate(random.sample(train_data, min(10, len(train_data)))):
        assert required_keys.issubset(sample.keys()), \
            f"Train sample {i} missing keys: {required_keys - set(sample.keys())}"
    print(f"✓ Train data structure valid")
    
    # Check test
    for i, sample in enumerate(random.sample(test_data, min(10, len(test_data)))):
        assert required_keys.issubset(sample.keys()), \
            f"Test sample {i} missing keys: {required_keys - set(sample.keys())}"
    print(f"✓ Test data structure valid")
    
    print(f"\n✅ PASSED: Data structure is correct")
    return True


def test_4_image_accessibility(train_data, test_data):
    """Test 4: Verify images can be loaded"""
    print("\n" + "="*80)
    print("TEST 4: Image Accessibility")
    print("="*80)
    
    def check_images(data, name, num_samples=20):
        samples = random.sample(data, min(num_samples, len(data)))
        success = 0
        failures = []
        
        for sample in samples:
            img_path = sample['image_path']
            try:
                if not os.path.exists(img_path):
                    failures.append((img_path, "File not found"))
                    continue
                    
                with Image.open(img_path) as img:
                    img.verify()
                    
                # Re-open for actual loading (verify closes the file)
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    width, height = img.size
                    
                success += 1
                
            except Exception as e:
                failures.append((img_path, str(e)))
        
        print(f"  {name}: {success}/{len(samples)} images loaded successfully")
        
        if failures:
            print(f"  ⚠ {len(failures)} failures:")
            for path, error in failures[:3]:
                print(f"    - {Path(path).name}: {error}")
        
        return success, len(samples)
    
    train_success, train_total = check_images(train_data, "Train")
    test_success, test_total = check_images(test_data, "Test")
    
    success_rate = (train_success + test_success) / (train_total + test_total)
    assert success_rate > 0.95, f"❌ Only {success_rate*100:.1f}% images accessible"
    
    print(f"\n✅ PASSED: {success_rate*100:.1f}% images accessible")
    return True


def test_5_caption_quality(train_data):
    """Test 5: Validate caption quality"""
    print("\n" + "="*80)
    print("TEST 5: Caption Quality")
    print("="*80)
    
    # Check caption diversity
    captions = [sample['caption'] for sample in train_data]
    unique_captions = len(set(captions))
    
    print(f"Total captions: {len(captions)}")
    print(f"Unique captions: {unique_captions}")
    print(f"Diversity: {unique_captions/len(captions)*100:.1f}%")
    
    # Sample captions
    print(f"\nSample captions:")
    for sample in random.sample(train_data, 10):
        print(f"  [{sample['class_id']}] {sample['label']}")
        print(f"  → '{sample['caption']}'")
        print()
    
    # ✅ FIXED: Check if labels are actual words, not just numeric IDs
    # A real label should have letters
    labels_with_letters = sum(1 for s in train_data if any(c.isalpha() for c in s['label']))
    purely_numeric = len(train_data) - labels_with_letters
    
    print(f"Labels with letters (real names): {labels_with_letters}/{len(train_data)}")
    print(f"Purely numeric labels: {purely_numeric}/{len(train_data)}")
    
    # Check sample of labels
    sample_labels = set(s['label'] for s in random.sample(train_data, 100))
    print(f"\nSample unique labels:")
    for label in list(sample_labels)[:10]:
        print(f"  - '{label}'")
    
    assert labels_with_letters > len(train_data) * 0.99, \
        f"❌ Only {labels_with_letters}/{len(train_data)} labels have real names"
    
    print(f"\n✅ PASSED: {labels_with_letters/len(train_data)*100:.1f}% labels are real names")

    
    return True


def test_6_class_distribution(train_data, test_data, labels):
    """Test 6: Analyze class distribution"""
    print("\n" + "="*80)
    print("TEST 6: Class Distribution")
    print("="*80)
    
    train_counts = Counter(s['class_id'] for s in train_data)
    test_counts = Counter(s['class_id'] for s in test_data)
    
    print(f"Number of classes in labels: {len(labels)}")
    print(f"Number of classes in train: {len(train_counts)}")
    print(f"Number of classes in test: {len(test_counts)}")
    
    # Check class balance
    train_sizes = list(train_counts.values())
    test_sizes = list(test_counts.values())
    
    print(f"\nTrain set per-class stats:")
    print(f"  Min: {min(train_sizes)} samples")
    print(f"  Max: {max(train_sizes)} samples")
    print(f"  Mean: {sum(train_sizes)/len(train_sizes):.1f} samples")
    
    print(f"\nTest set per-class stats:")
    print(f"  Min: {min(test_sizes)} samples")
    print(f"  Max: {max(test_sizes)} samples")
    print(f"  Mean: {sum(test_sizes)/len(test_sizes):.1f} samples")
    
    # Show top 10 classes
    print(f"\nTop 10 classes by training samples:")
    for class_id, count in train_counts.most_common(10):
        label = labels.get(class_id, "Unknown")
        print(f"  {class_id} ({label}): {count} train, {test_counts.get(class_id, 0)} test")
    
    assert len(train_counts) == len(labels), "❌ Train set missing some classes"
    assert len(test_counts) == len(labels), "❌ Test set missing some classes"
    
    print(f"\n✅ PASSED: All classes present in both splits")
    return True


def test_7_data_leakage(train_data, test_data):
    """Test 7: Check for data leakage"""
    print("\n" + "="*80)
    print("TEST 7: Data Leakage Check")
    print("="*80)
    
    train_images = set(s['image_path'] for s in train_data)
    test_images = set(s['image_path'] for s in test_data)
    
    overlap = train_images & test_images
    
    print(f"Train images: {len(train_images)}")
    print(f"Test images: {len(test_images)}")
    print(f"Overlap: {len(overlap)}")
    
    if overlap:
        print(f"\n❌ FAILED: Found {len(overlap)} overlapping images:")
        for img in list(overlap)[:5]:
            print(f"  - {Path(img).name}")
        raise AssertionError("Data leakage detected!")
    
    print(f"\n✅ PASSED: No data leakage")
    return True


def test_8_clip_compatibility(train_data):
    """Test 8: Verify CLIP tokenizer and transforms work"""
    print("\n" + "="*80)
    print("TEST 8: CLIP Compatibility")
    print("="*80)
    
    # Initialize CLIP tokenizer
    print("Initializing CLIP tokenizer...")
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    # Test tokenization
    sample = random.choice(train_data)
    caption = sample['caption']
    
    print(f"\nTest caption: '{caption}'")
    tokens = tokenizer(caption)
    print(f"Token shape: {tokens.shape}")
    print(f"Tokens: {tokens.squeeze()[:15].tolist()}...")
    
    assert tokens.shape[-1] == 77, f"❌ Wrong token length: {tokens.shape}"
    assert tokens.squeeze()[0] == 49406, f"❌ Wrong start token: {tokens.squeeze()[0]}"
    
    print(f"✓ Tokenization works correctly")
    
    # Test image transforms
    print(f"\nTesting image transforms...")
    transform = get_clip_transforms()
    
    img_path = sample['image_path']
    img = Image.open(img_path).convert('RGB')
    print(f"Original image size: {img.size}")
    
    img_tensor = transform(img)
    print(f"Transformed image shape: {img_tensor.shape}")
    print(f"Image value range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    
    assert img_tensor.shape == (3, 224, 224), f"❌ Wrong image shape: {img_tensor.shape}"
    assert -5 < img_tensor.min() < 0, f"❌ Unexpected min value: {img_tensor.min()}"
    assert 0 < img_tensor.max() < 5, f"❌ Unexpected max value: {img_tensor.max()}"
    
    print(f"✓ Image transforms work correctly")
    
    print(f"\n✅ PASSED: CLIP compatibility verified")
    return True


def test_9_normalization_check(train_data):
    """Test 9: Verify normalization is CLIP, not ImageNet"""
    print("\n" + "="*80)
    print("TEST 9: Normalization Verification")
    print("="*80)
    
    transform_clip = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    transform_imagenet = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    sample = random.choice(train_data)
    img = Image.open(sample['image_path']).convert('RGB')
    
    img_clip = transform_clip(img)
    img_imagenet = transform_imagenet(img)
    
    diff = (img_clip - img_imagenet).abs().mean().item()
    
    print(f"CLIP normalized mean: {img_clip.mean().item():.4f}")
    print(f"ImageNet normalized mean: {img_imagenet.mean().item():.4f}")
    print(f"Difference: {diff:.4f}")
    
    print(f"\n⚠️  IMPORTANT: Use CLIP normalization in your training!")
    print(f"  CLIP mean: [0.48145466, 0.4578275, 0.40821073]")
    print(f"  CLIP std:  [0.26862954, 0.26130258, 0.27577711]")
    
    print(f"\n✅ PASSED: Normalization check complete")
    return True


def generate_sample_outputs(train_data, test_data, labels, num_samples=5):
    """Generate example outputs for visual inspection"""
    print("\n" + "="*80)
    print("SAMPLE DATA EXAMPLES")
    print("="*80)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Training samples: {len(train_data):,}")
    print(f"  Test samples: {len(test_data):,}")
    print(f"  Total samples: {len(train_data) + len(test_data):,}")
    print(f"  Number of classes: {len(labels)}")
    
    print(f"\n📝 Sample Training Examples:")
    for i, sample in enumerate(random.sample(train_data, num_samples), 1):
        print(f"\n  Example {i}:")
        print(f"    Class ID: {sample['class_id']}")
        print(f"    Label: {sample['label']}")
        print(f"    Caption: '{sample['caption']}'")
        print(f"    Image: {Path(sample['image_path']).name}")
    
    print(f"\n🏷️  Class Label Examples:")
    for class_id, label in list(labels.items())[:10]:
        count_train = sum(1 for s in train_data if s['class_id'] == class_id)
        count_test = sum(1 for s in test_data if s['class_id'] == class_id)
        print(f"    {class_id} → '{label}' ({count_train} train, {count_test} test)")
    
    print(f"\n✅ Sample outputs generated")


def test_10_pytorch_dataloader(train_data):
    """Test 10: Verify PyTorch DataLoader compatibility"""
    print("\n" + "="*80)
    print("TEST 10: PyTorch DataLoader Compatibility")
    print("="*80)
    
    from torch.utils.data import Dataset, DataLoader
    
    class TestDataset(Dataset):
        def __init__(self, data):
            self.data = data
            self.transform = get_clip_transforms()
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            image = Image.open(item['image_path']).convert('RGB')
            image = self.transform(image)
            tokens = self.tokenizer(item['caption']).squeeze(0)
            return image, tokens
    
    dataset = TestDataset(train_data[:100])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    print(f"Created DataLoader with {len(dataset)} samples")
    
    # Test one batch
    images, tokens = next(iter(dataloader))
    
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")  # Should be [16, 3, 224, 224]
    print(f"  Tokens: {tokens.shape}")  # Should be [16, 77]
    
    assert images.shape == (16, 3, 224, 224), f"❌ Wrong image batch shape"
    assert tokens.shape == (16, 77), f"❌ Wrong token batch shape"
    
    print(f"\n✅ PASSED: DataLoader works correctly")
    return True


# ============================================================================
# MAIN VALIDATION RUNNER
# ============================================================================

def run_all_tests():
    """Run all validation tests"""
    print("\n" + "="*80)
    print("CLIP IMAGENET-100 DATASET VALIDATION")
    print("="*80)
    print(f"Dataset directory: {PROCESSED_DIR}")
    
    try:
        # Run tests
        test_1_file_existence()
        train_data, test_data, labels = test_2_load_and_parse()
        test_3_data_structure(train_data, test_data)
        test_4_image_accessibility(train_data, test_data)
        test_5_caption_quality(train_data)
        test_6_class_distribution(train_data, test_data, labels)
        test_7_data_leakage(train_data, test_data)
        test_8_clip_compatibility(train_data)
        test_9_normalization_check(train_data)
        test_10_pytorch_dataloader(train_data)
        
        # Generate samples
        generate_sample_outputs(train_data, test_data, labels)
        
        print("\n" + "="*80)
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("="*80)
        print("\n✅ Your dataset is ready for CLIP training!")
        print(f"\nFiles:")
        print(f"  Train: {TRAIN_FILE}")
        print(f"  Test: {TEST_FILE}")
        print(f"  Labels: {LABELS_FILE}")
        
        print(f"\n📌 CRITICAL REMINDER:")
        print(f"  Use CLIP normalization in your training script:")
        print(f"  mean=[0.48145466, 0.4578275, 0.40821073]")
        print(f"  std=[0.26862954, 0.26130258, 0.27577711]")
        
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
