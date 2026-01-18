# download_classification_data.py - FIXED
from datasets import load_dataset
import json
import os

def download_classification_data(output_dir='/data/joshi/utils/ESM_junk'):
    """
    Download protein classification dataset
    Using beta_lactamase (antibiotic resistance) - binary classification
    """
    print("=" * 60)
    print("Downloading Protein Classification Dataset")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nDownloading beta-lactamase dataset (antibiotic resistance)...")
    print("This is a binary classification task similar to AAV")
    
    # Use beta_lactamase - it's available and similar task
    dataset = load_dataset("InstaDeepAI/true-cds-protein-tasks", "beta_lactamase_complete")
    
    print("✅ Dataset loaded!")
    print(f"Available splits: {list(dataset.keys())}")
    
    # Check the first example to see field names
    first_example = dataset['train'][0]
    print(f"\nDataset fields: {list(first_example.keys())}")
    print(f"First example preview: {first_example}")
    
    # Extract data from all splits
    print("\nProcessing train split...")
    train_data = []
    for example in dataset['train']:
        # Use the correct field name
        if 'sequence' in example:
            seq = example['sequence']
        elif 'seq' in example:
            seq = example['seq']
        elif 'protein_sequence' in example:
            seq = example['protein_sequence']
        else:
            # Use first string field
            seq = str(list(example.values())[0])
        
        # Get label
        if 'target' in example:
            label = int(example['target'])
        elif 'label' in example:
            label = int(example['label'])
        elif 'y' in example:
            label = int(example['y'])
        else:
            label = int(list(example.values())[1])
        
        train_data.append({
            'sequence': seq,
            'label': label
        })
    
    print(f"✅ Train: {len(train_data):,} samples")
    
    # Process validation split
    print("Processing validation split...")
    val_data = []
    for example in dataset['validation']:
        if 'sequence' in example:
            seq = example['sequence']
        elif 'seq' in example:
            seq = example['seq']
        elif 'protein_sequence' in example:
            seq = example['protein_sequence']
        else:
            seq = str(list(example.values())[0])
        
        if 'target' in example:
            label = int(example['target'])
        elif 'label' in example:
            label = int(example['label'])
        elif 'y' in example:
            label = int(example['y'])
        else:
            label = int(list(example.values())[1])
        
        val_data.append({
            'sequence': seq,
            'label': label
        })
    
    print(f"✅ Val: {len(val_data):,} samples")
    
    # Process test split
    print("Processing test split...")
    test_data = []
    for example in dataset['test']:
        if 'sequence' in example:
            seq = example['sequence']
        elif 'seq' in example:
            seq = example['seq']
        elif 'protein_sequence' in example:
            seq = example['protein_sequence']
        else:
            seq = str(list(example.values())[0])
        
        if 'target' in example:
            label = int(example['target'])
        elif 'label' in example:
            label = int(example['label'])
        elif 'y' in example:
            label = int(example['y'])
        else:
            label = int(list(example.values())[1])
        
        test_data.append({
            'sequence': seq,
            'label': label
        })
    
    print(f"✅ Test: {len(test_data):,} samples")
    
    # Save files
    train_file = os.path.join(output_dir, 'classification_train.json')
    print(f"\nSaving {train_file}...")
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    val_file = os.path.join(output_dir, 'classification_val.json')
    print(f"Saving {val_file}...")
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    test_file = os.path.join(output_dir, 'classification_test.json')
    print(f"Saving {test_file}...")
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Classification Dataset Downloaded")
    print("=" * 60)
    print(f"Task: Beta-lactamase (antibiotic resistance)")
    print(f"Type: Binary classification (0 or 1)")
    print(f"\nTrain samples: {len(train_data):,}")
    print(f"Val samples: {len(val_data):,}")
    print(f"Test samples: {len(test_data):,}")
    
    # Label distribution
    train_labels = [d['label'] for d in train_data]
    print(f"\nTrain label distribution:")
    print(f"  Class 0: {train_labels.count(0):,} ({train_labels.count(0)/len(train_labels)*100:.1f}%)")
    print(f"  Class 1: {train_labels.count(1):,} ({train_labels.count(1)/len(train_labels)*100:.1f}%)")
    
    # Sequence length
    seq_lengths = [len(d['sequence']) for d in train_data]
    print(f"\nSequence length statistics:")
    print(f"  Min: {min(seq_lengths)} aa")
    print(f"  Max: {max(seq_lengths)} aa")
    print(f"  Mean: {sum(seq_lengths)/len(seq_lengths):.1f} aa")
    
    # Show example
    print(f"\nExample sequence:")
    print(f"  Sequence: {train_data[0]['sequence'][:50]}...")
    print(f"  Label: {train_data[0]['label']}")
    print(f"  Length: {len(train_data[0]['sequence'])} aa")
    
    print(f"\nFiles saved:")
    print(f"  📁 {train_file}")
    print(f"  📁 {val_file}")
    print(f"  📁 {test_file}")
    
    print("\n✅ Ready for classification fine-tuning!")
    print("\nNote: Beta-lactamase classification:")
    print("  - Binary classification (resistant/not resistant)")
    print("  - Predicts antibiotic resistance")
    print("  - Same workflow as AAV!")
    
    return train_file, val_file, test_file

if __name__ == "__main__":
    download_classification_data()
