# verify_split.py
import json

def verify_split():
    # Load both files
    with open('/data/joshi/utils/ESM_junk/mlm_train.json', 'r') as f:
        train_data = json.load(f)
    
    with open('/data/joshi/utils/ESM_junk/mlm_val.json', 'r') as f:
        val_data = json.load(f)
    
    print(f"Train sequences: {len(train_data):,}")
    print(f"Val sequences: {len(val_data):,}")
    print(f"Total: {len(train_data) + len(val_data):,}")
    
    # Check no overlap
    train_ids = set(s['id'] for s in train_data)
    val_ids = set(s['id'] for s in val_data)
    overlap = train_ids & val_ids
    
    if len(overlap) == 0:
        print("\n✅ No overlap between train and val!")
    else:
        print(f"\n⚠️ Found {len(overlap)} overlapping sequences")
    
    # Show examples
    print("\nTrain example:")
    print(f"  ID: {train_data[0]['id']}")
    print(f"  Length: {train_data[0]['length']}")
    print(f"  Sequence: {train_data[0]['sequence'][:50]}...")
    
    print("\nVal example:")
    print(f"  ID: {val_data[0]['id']}")
    print(f"  Length: {val_data[0]['length']}")
    print(f"  Sequence: {val_data[0]['sequence'][:50]}...")

if __name__ == "__main__":
    verify_split()
