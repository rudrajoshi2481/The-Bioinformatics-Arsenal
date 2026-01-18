# preprocess_and_split.py
from Bio import SeqIO
import json
from tqdm import tqdm
import os
import random

def preprocess_and_split_uniref50(
    input_fasta='/data/joshi/utils/ESM_junk/uniref50.fasta',
    output_dir='/data/joshi/utils/ESM_junk',
    max_sequences=1_000_000,
    min_length=50,
    max_length=1024,
    val_ratio=0.01,  # 1% for validation (10K sequences)
    seed=42
):
    """
    Filter UniRef50 AND split into train/val in one script
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Preprocessing & Splitting UniRef50")
    print("=" * 60)
    print(f"Input: {input_fasta}")
    print(f"Output dir: {output_dir}")
    print(f"Target: {max_sequences:,} sequences")
    print(f"Length range: {min_length}-{max_length} amino acids")
    print(f"Validation ratio: {val_ratio*100}%")
    print()
    
    # ============================================
    # STEP 1: Filter sequences
    # ============================================
    print("STEP 1: Filtering sequences...")
    print("-" * 60)
    
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    
    sequences = []
    skipped_length = 0
    skipped_invalid = 0
    
    with open(input_fasta, 'r') as f:
        for record in tqdm(SeqIO.parse(f, 'fasta'), desc="Filtering"):
            seq = str(record.seq).upper()
            seq_len = len(seq)
            
            # Filter by length
            if not (min_length <= seq_len <= max_length):
                skipped_length += 1
                continue
            
            # Filter by valid amino acids
            if not all(aa in valid_amino_acids for aa in seq):
                skipped_invalid += 1
                continue
            
            # Keep this sequence
            sequences.append({
                'sequence': seq,
                'length': seq_len,
                'id': record.id,
                'description': record.description[:100]
            })
            
            # Stop when we reach target
            if len(sequences) >= max_sequences:
                print(f"\n✅ Reached target of {max_sequences:,} sequences!")
                break
    
    print(f"\n✅ Kept: {len(sequences):,} sequences")
    print(f"❌ Skipped (length): {skipped_length:,}")
    print(f"❌ Skipped (invalid chars): {skipped_invalid:,}")
    
    # ============================================
    # STEP 2: Split into train/val
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 2: Splitting into train/val")
    print("-" * 60)
    
    # Shuffle sequences
    random.shuffle(sequences)
    
    # Calculate split point
    val_size = int(len(sequences) * val_ratio)
    train_size = len(sequences) - val_size
    
    # Split
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:]
    
    print(f"Train sequences: {len(train_sequences):,}")
    print(f"Val sequences: {len(val_sequences):,}")
    
    # ============================================
    # STEP 3: Save files
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 3: Saving files")
    print("-" * 60)
    
    # Save train
    train_file = os.path.join(output_dir, 'mlm_train.json')
    print(f"Saving train data to {train_file}...")
    with open(train_file, 'w') as f:
        json.dump(train_sequences, f, indent=2)
    
    # Save validation
    val_file = os.path.join(output_dir, 'mlm_val.json')
    print(f"Saving validation data to {val_file}...")
    with open(val_file, 'w') as f:
        json.dump(val_sequences, f, indent=2)
    
    # ============================================
    # STEP 4: Statistics
    # ============================================
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    
    # Length statistics
    all_lengths = [s['length'] for s in sequences]
    train_lengths = [s['length'] for s in train_sequences]
    val_lengths = [s['length'] for s in val_sequences]
    
    print("\nOverall Dataset:")
    print(f"  Total sequences: {len(sequences):,}")
    print(f"  Min length: {min(all_lengths)} aa")
    print(f"  Max length: {max(all_lengths)} aa")
    print(f"  Mean length: {sum(all_lengths)/len(all_lengths):.1f} aa")
    print(f"  Median length: {sorted(all_lengths)[len(all_lengths)//2]} aa")
    
    print("\nTrain Set:")
    print(f"  Sequences: {len(train_sequences):,}")
    print(f"  Mean length: {sum(train_lengths)/len(train_lengths):.1f} aa")
    
    print("\nValidation Set:")
    print(f"  Sequences: {len(val_sequences):,}")
    print(f"  Mean length: {sum(val_lengths)/len(val_lengths):.1f} aa")
    
    print("\nFile Sizes:")
    train_size_gb = os.path.getsize(train_file) / (1024**3)
    val_size_mb = os.path.getsize(val_file) / (1024**2)
    print(f"  Train: {train_size_gb:.2f} GB")
    print(f"  Val: {val_size_mb:.1f} MB")
    
    print("\nFiles saved:")
    print(f"  📁 {train_file}")
    print(f"  📁 {val_file}")
    
    print("\n🎉 Preprocessing complete! Ready for MLM training.")
    
    return train_file, val_file

if __name__ == "__main__":
    train_file, val_file = preprocess_and_split_uniref50()
