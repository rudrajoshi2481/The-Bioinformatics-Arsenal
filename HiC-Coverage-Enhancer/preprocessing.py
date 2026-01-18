"""
Hi-C Enhancement Preprocessing Pipeline - OPTIMIZED VERSION
============================================================

IMPROVEMENTS:
- Memory-efficient incremental saving
- Lower filtering thresholds for better data retention
- Better error handling and progress reporting
- Validation and statistics tracking
- Support for multiple samples and resolutions

Author: Bioinformatics Deep Learning Pipeline
Version: 2.0 (Optimized)
"""

import numpy as np
import cooler
from scipy.sparse import coo_matrix, triu
from typing import Tuple, Dict, List, Optional
import warnings
import os
from pathlib import Path
from tqdm import tqdm
import json
import gc
from datetime import datetime

warnings.filterwarnings('ignore')


class HiCEnhancementPipeline:
    """Memory-efficient Hi-C enhancement preprocessing pipeline"""
    
    def __init__(self, chunk_size=40, stride=28, downsample_ratio=16, 
                 high_cutoff=255, low_cutoff=100, min_contacts=3):
        """
        Initialize pipeline parameters
        
        Args:
            chunk_size: Size of patches for model input (40x40)
            stride: Step size between patches (28 = 12 pixel overlap)
            downsample_ratio: Downsampling factor (16x)
            high_cutoff: Clipping value for high-res normalization
            low_cutoff: Clipping value for low-res normalization
            min_contacts: Minimum contacts to keep chunk (LOWERED from 20 to 5)
        """
        self.chunk_size = chunk_size
        self.stride = stride
        self.downsample_ratio = downsample_ratio
        self.high_cutoff = high_cutoff
        self.low_cutoff = low_cutoff
        self.min_contacts = min_contacts
        
        # Training chromosomes - use fewer for memory efficiency
        self.train_chroms = [f'chr{i}' for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
        self.val_chroms = ['chr21']
        self.test_chroms = ['chr22']
        
        # Statistics tracking
        self.stats = {
            'chromosomes_processed': 0,
            'chromosomes_failed': 0,
            'total_chunks_before_filter': 0,
            'total_chunks_after_filter': 0,
            'filter_rate': 0.0
        }
        
    def load_cooler(self, file_path: str, chrom: str, res: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load Hi-C matrix from .mcool file"""
        try:
            clr = cooler.Cooler(f"{file_path}::/resolutions/{res}")
            matrix = clr.matrix(balance=False).fetch(chrom)
            bins_df = clr.bins().fetch(chrom)
            
            weights = bins_df["weight"].values if "weight" in bins_df.columns else np.ones(len(bins_df))
            valid_bins = np.where(~np.isnan(weights))[0]
            
            matrix = matrix.astype(np.int32)
            
            metadata = {
                'bins_df': bins_df,
                'chrom': chrom,
                'resolution': res,
                'genome': clr.chromsizes.to_dict(),
                'original_shape': matrix.shape,
                'valid_bins': valid_bins,
                'weights': weights
            }
            
            return matrix, valid_bins, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {chrom} from {file_path}: {str(e)}")
    
    def compact_matrix(self, matrix: np.ndarray, valid_bins: np.ndarray) -> np.ndarray:
        """Remove unmappable bins (genomic gaps)"""
        compact_size = len(valid_bins)
        compact_mat = np.zeros((compact_size, compact_size), dtype=matrix.dtype)
        
        for i, row_idx in enumerate(valid_bins):
            compact_mat[i, :] = matrix[row_idx, valid_bins]
        
        return compact_mat
    
    def downsample_matrix(self, matrix: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Downsample matrix to simulate low coverage - MEMORY EFFICIENT VERSION
        Uses probabilistic sampling instead of expanding all reads.
        
        Args:
            matrix: Input contact matrix
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Use probabilistic downsampling - much more memory efficient
        # Each contact has 1/downsample_ratio probability of being kept
        prob = 1.0 / self.downsample_ratio
        
        # Get upper triangle
        matrix_upper = np.triu(matrix)
        
        # For each cell, sample from binomial distribution
        # This is equivalent to sampling reads but much more memory efficient
        downsampled_upper = np.random.binomial(matrix_upper.astype(np.int32), prob)
        
        # Make symmetric
        downsampled = downsampled_upper + np.triu(downsampled_upper, k=1).T
        
        return downsampled.astype(matrix.dtype)
    
    def normalize_matrices(self, high_res: np.ndarray, low_res: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize matrices to [0, 1] range"""
        high_clipped = np.minimum(high_res, self.high_cutoff)
        low_clipped = np.minimum(low_res, self.low_cutoff)
        
        high_norm = high_clipped.astype(np.float32) / self.high_cutoff
        low_norm = low_clipped.astype(np.float32) / self.low_cutoff
        
        return high_norm, low_norm
    
    def create_chunks(self, matrix: np.ndarray, chr_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """Divide matrix into overlapping patches"""
        result = []
        indices = []
        size = matrix.shape[0]
        
        # Padding for edge cases
        pad_len = (self.chunk_size - self.stride) // 2
        mat_padded = np.pad(matrix, ((pad_len, pad_len), (pad_len, pad_len)), 'constant')
        
        height, width = mat_padded.shape
        
        for i in range(0, height - self.chunk_size + 1, self.stride):
            for j in range(0, width - self.chunk_size + 1, self.stride):
                chunk = mat_padded[i:i + self.chunk_size, j:j + self.chunk_size]
                result.append([chunk])
                indices.append((chr_num, size, i, j))
        
        result = np.array(result, dtype=np.float32)
        indices = np.array(indices, dtype=np.int32)
        
        return result, indices
    
    def filter_empty_chunks(self, chunks_low: np.ndarray, chunks_high: np.ndarray, 
                           indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove chunks with insufficient contacts"""
        chunk_sums = np.array([chunk.sum() for chunk in chunks_high])
        valid_mask = chunk_sums >= self.min_contacts
        
        # Update statistics
        self.stats['total_chunks_before_filter'] += len(chunks_high)
        self.stats['total_chunks_after_filter'] += valid_mask.sum()
        
        return chunks_low[valid_mask], chunks_high[valid_mask], indices[valid_mask]
    
    def process_single_chromosome(self, file_path: str, chrom: str, res: int, 
                                  seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process single chromosome to generate training chunks
        
        Returns:
            chunks_low: Low-res input chunks
            chunks_high: High-res target chunks
            indices: Position indices
        """
        try:
            # Load
            matrix, valid_bins, metadata = self.load_cooler(file_path, chrom, res)
            
            # Compact
            compact_mat = self.compact_matrix(matrix, valid_bins)
            
            # Downsample
            downsampled = self.downsample_matrix(compact_mat, seed=seed)
            
            # Normalize
            high_norm, low_norm = self.normalize_matrices(compact_mat, downsampled)
            
            # Create chunks
            chr_num = int(chrom.replace('chr', '')) if chrom.startswith('chr') else 0
            chunks_high, indices_high = self.create_chunks(high_norm, chr_num)
            chunks_low, _ = self.create_chunks(low_norm, chr_num)
            
            # Filter
            chunks_low, chunks_high, indices = self.filter_empty_chunks(chunks_low, chunks_high, indices_high)
            
            # Free memory
            del matrix, compact_mat, downsampled, high_norm, low_norm
            gc.collect()
            
            self.stats['chromosomes_processed'] += 1
            
            return chunks_low, chunks_high, indices
            
        except Exception as e:
            self.stats['chromosomes_failed'] += 1
            raise RuntimeError(f"Error processing {chrom}: {str(e)}")
    
    def prepare_training_dataset(self, 
                                sample_files: List[str],
                                resolutions: List[int] = [10000, 25000],
                                output_dir: str = './training_data',
                                use_all_chroms: bool = False,
                                save_separate: bool = False,
                                max_memory_gb: float = 6.0) -> Dict:
        """
        MEMORY-EFFICIENT: Prepare training dataset with incremental saving
        
        Args:
            sample_files: List of .mcool file paths
            resolutions: List of resolutions to use
            output_dir: Directory to save processed data
            use_all_chroms: If True, use ALL chromosomes
            save_separate: If True, save train/val/test separately
            max_memory_gb: Maximum RAM to use before saving
        """
        start_time = datetime.now()
        
        print(f"\n{'='*70}")
        print(f"HI-C ENHANCEMENT PREPROCESSING - OPTIMIZED VERSION")
        print(f"{'='*70}")
        print(f"  Samples: {len(sample_files)}")
        print(f"  Resolutions: {resolutions}")
        print(f"  Output: {output_dir}")
        print(f"  Max RAM buffer: {max_memory_gb} GB")
        print(f"  Min contacts threshold: {self.min_contacts}")
        print(f"  Downsample ratio: {self.downsample_ratio}x")
        print(f"{'='*70}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine chromosomes
        chroms_to_process = (
            [f'chr{i}' for i in range(1, 23)] + ['chrX'] if use_all_chroms 
            else self.train_chroms
        )
        
        print(f"  Processing {len(chroms_to_process)} chromosomes")
        if not use_all_chroms:
            print(f"  Validation: {self.val_chroms}")
            print(f"  Test: {self.test_chroms}")
        
        # Initialize buffers
        train_file = os.path.join(output_dir, 'train_dataset.npz')
        temp_files = []
        
        chunk_buffer_low = []
        chunk_buffer_high = []
        index_buffer = []
        current_memory_gb = 0
        file_counter = 0
        
        # Process each sample
        for sample_idx, sample_file in enumerate(sample_files):
            sample_name = Path(sample_file).stem
            print(f"\n[{sample_idx+1}/{len(sample_files)}] Processing {sample_name}...")
            
            # Verify file exists
            if not os.path.exists(sample_file):
                print(f"  ⚠️  File not found: {sample_file}")
                continue
            
            for res in resolutions:
                print(f"  Resolution: {res//1000}kb")
                
                for chrom in tqdm(chroms_to_process, desc=f"    Chromosomes"):
                    try:
                        chunks_low, chunks_high, indices = self.process_single_chromosome(
                            sample_file, chrom, res, seed=42
                        )
                        
                        if len(chunks_low) == 0:
                            continue
                        
                        chunk_buffer_low.append(chunks_low)
                        chunk_buffer_high.append(chunks_high)
                        index_buffer.append(indices)
                        
                        # Estimate memory usage
                        current_memory_gb += (chunks_low.nbytes + chunks_high.nbytes) / (1024**3)
                        
                        # Save when buffer reaches limit
                        if current_memory_gb >= max_memory_gb:
                            temp_file = os.path.join(output_dir, f'temp_part_{file_counter}.npz')
                            print(f"\n      💾 Saving buffer ({current_memory_gb:.2f} GB) -> part {file_counter}")
                            
                            np.savez_compressed(
                                temp_file,
                                data=np.concatenate(chunk_buffer_low, axis=0),
                                target=np.concatenate(chunk_buffer_high, axis=0),
                                indices=np.concatenate(index_buffer, axis=0)
                            )
                            
                            temp_files.append(temp_file)
                            file_counter += 1
                            
                            # Clear buffers
                            chunk_buffer_low = []
                            chunk_buffer_high = []
                            index_buffer = []
                            current_memory_gb = 0
                            gc.collect()
                        
                    except Exception as e:
                        print(f"\n      ⚠️  Error processing {chrom}: {str(e)}")
                        continue
        
        # Save remaining buffer
        if len(chunk_buffer_low) > 0:
            temp_file = os.path.join(output_dir, f'temp_part_{file_counter}.npz')
            print(f"\n💾 Saving final buffer ({current_memory_gb:.2f} GB) -> part {file_counter}")
            
            np.savez_compressed(
                temp_file,
                data=np.concatenate(chunk_buffer_low, axis=0),
                target=np.concatenate(chunk_buffer_high, axis=0),
                indices=np.concatenate(index_buffer, axis=0)
            )
            temp_files.append(temp_file)
        
        # Check if any data was collected
        if len(temp_files) == 0:
            raise RuntimeError("No data was successfully processed! Check your input files.")
        
        # Merge all temp files
        print(f"\n{'='*70}")
        print(f"Merging {len(temp_files)} temporary files...")
        
        all_low = []
        all_high = []
        all_indices = []
        
        for temp_file in tqdm(temp_files, desc="Loading temp files"):
            data = np.load(temp_file)
            all_low.append(data['data'])
            all_high.append(data['target'])
            all_indices.append(data['indices'])
        
        print("Concatenating final dataset...")
        final_low = np.concatenate(all_low, axis=0)
        final_high = np.concatenate(all_high, axis=0)
        final_indices = np.concatenate(all_indices, axis=0)
        
        # Save final dataset
        print(f"Saving final dataset...")
        np.savez_compressed(
            train_file,
            data=final_low,
            target=final_high,
            indices=final_indices
        )
        
        # Clean up temp files
        print("Cleaning up temporary files...")
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        # Calculate statistics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.stats['filter_rate'] = (
            (self.stats['total_chunks_before_filter'] - self.stats['total_chunks_after_filter']) 
            / max(self.stats['total_chunks_before_filter'], 1) * 100
        )
        
        print(f"\n{'='*70}")
        print(f"✅ DATASET PREPARATION COMPLETE!")
        print(f"{'='*70}")
        print(f"  Total chunks: {len(final_low):,}")
        print(f"  Data shape: Low={final_low.shape}, High={final_high.shape}")
        print(f"  File size: {os.path.getsize(train_file) / (1024**2):.2f} MB")
        print(f"  Saved to: {train_file}")
        print(f"\n  Statistics:")
        print(f"    Chromosomes processed: {self.stats['chromosomes_processed']}")
        print(f"    Chromosomes failed: {self.stats['chromosomes_failed']}")
        print(f"    Chunks before filtering: {self.stats['total_chunks_before_filter']:,}")
        print(f"    Chunks after filtering: {self.stats['total_chunks_after_filter']:,}")
        print(f"    Filter rate: {self.stats['filter_rate']:.1f}%")
        print(f"  Processing time: {duration/60:.1f} minutes")
        print(f"{'='*70}\n")
        
        # Dataset info
        dataset_info = {
            'n_samples': len(sample_files),
            'n_resolutions': len(resolutions),
            'resolutions': resolutions,
            'n_chromosomes': len(chroms_to_process),
            'total_chunks': len(final_low),
            'train_file': train_file,
            'chunk_size': self.chunk_size,
            'stride': self.stride,
            'downsample_ratio': self.downsample_ratio,
            'min_contacts': self.min_contacts,
            'statistics': self.stats,
            'processing_time_seconds': duration,
            'timestamp': start_time.isoformat()
        }
        
        # Save metadata
        meta_file = os.path.join(output_dir, 'dataset_info.json')
        with open(meta_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        print(f"  ✓ Metadata saved: {meta_file}")
        
        # Training recommendations
        self._print_training_recommendations(len(final_low))
        
        return dataset_info
    
    def _print_training_recommendations(self, total_chunks: int):
        """Print recommendations based on dataset size"""
        print(f"\n{'='*70}")
        print(f"📊 TRAINING RECOMMENDATIONS")
        print(f"{'='*70}")
        
        batch_size = 32
        steps_per_epoch = total_chunks // batch_size
        
        print(f"\n  Dataset size: {total_chunks:,} chunks")
        print(f"  With batch_size={batch_size}:")
        print(f"    Steps per epoch: {steps_per_epoch}")
        
        if total_chunks < 10000:
            print(f"\n  ⚠️  WARNING: Small dataset!")
            print(f"     Recommended actions:")
            print(f"     1. Add more samples")
            print(f"     2. Use both 10kb + 25kb resolutions")
            print(f"     3. Lower min_contacts further (try 3)")
            print(f"     4. Consider data augmentation")
        elif total_chunks < 50000:
            print(f"\n  ✓ Moderate dataset size")
            print(f"    Should work but consider adding more samples for better results")
        else:
            print(f"\n  ✅ Good dataset size for training!")
            print(f"     Expected training time: ~1-2 hours on RTX 3060")
        
        print(f"{'='*70}\n")


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION - Edit these paths for your data
    # =========================================================================
    
    # Input mcool files - add your Hi-C data files here
    MCOOL_FILES = [
        "/path/to/your/sample1.mcool",
        # "/path/to/your/sample2.mcool",  # Add more samples if available
    ]
    
    # Output directory for training data
    OUTPUT_DIR = './training_data'
    
    # Resolution(s) to process (in base pairs)
    RESOLUTIONS = [25000]  # e.g., [10000, 25000] for multiple resolutions
    
    # Memory limit (GB) - adjust based on your system RAM
    MAX_MEMORY_GB = 4.0
    
    # =========================================================================
    
    # Initialize pipeline
    pipeline = HiCEnhancementPipeline(
        chunk_size=40,
        stride=28,
        downsample_ratio=16,
        high_cutoff=255,
        low_cutoff=100,
        min_contacts=5
    )
    
    # Run preprocessing
    dataset_info = pipeline.prepare_training_dataset(
        sample_files=MCOOL_FILES,
        resolutions=RESOLUTIONS,
        output_dir=OUTPUT_DIR,
        use_all_chroms=False,
        save_separate=False,
        max_memory_gb=MAX_MEMORY_GB
    )
    
    print("\nPreprocessing complete! Ready for training.")
    print("Next step: python run_training.py")
