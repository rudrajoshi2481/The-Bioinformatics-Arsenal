#!/usr/bin/env python3
"""
Hi-C Enhancement Inference
==========================

Memory-efficient inference script to generate enhanced .mcool files.
Processes chromosomes in spatial tiles to minimize memory usage.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cooler
from pathlib import Path
from tqdm import tqdm
import os
import gc
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple
import pandas as pd

from model import HiCEnhancementModel


class SimpleChunkDataset(Dataset):
    """Simple dataset for inference chunks"""
    def __init__(self, chunks):
        self.chunks = torch.from_numpy(chunks).float()
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]


class HiCEnhancer:
    """
    Hi-C Enhancement Inference Class
    
    Processes Hi-C contact matrices using a trained deep learning model
    to enhance low-coverage data to high-coverage quality.
    
    Uses tiled processing to handle large chromosomes with limited memory.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        chunk_size: int = 40,
        stride: int = 28
    ):
        """
        Initialize the enhancer
        
        Args:
            model_path: Path to trained model checkpoint (.pth file)
            device: Device to use ('auto', 'cuda', 'cpu')
            chunk_size: Size of patches (must match training, default 40)
            stride: Stride between patches (default 28)
        """
        self.chunk_size = chunk_size
        self.stride = stride
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        print(f"Model loaded on {self.device}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get model config from checkpoint
        model_config = checkpoint.get('config', {}).get('model', {})
        
        model = HiCEnhancementModel(
            base_channels=model_config.get('base_channels', 32),
            num_transformer_layers=model_config.get('num_transformer_layers', 2),
            num_heads=model_config.get('num_heads', 4),
            dropout=model_config.get('dropout', 0.1)
        ).to(self.device)
        
        # Handle DataParallel state dict
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        
        del checkpoint
        gc.collect()
        
        return model
    
    @torch.no_grad()
    def _enhance_tile(self, tile: np.ndarray, tile_max: float, batch_size: int = 8) -> np.ndarray:
        """Enhance a single tile"""
        # Normalize (same as training: clip at 100, normalize to 0-1)
        tile_norm = np.minimum(tile, 100) / 100.0
        
        # Create chunks
        pad_len = (self.chunk_size - self.stride) // 2
        mat_padded = np.pad(tile_norm, ((pad_len, pad_len), (pad_len, pad_len)), 'constant')
        
        chunks = []
        indices = []
        h, w = mat_padded.shape
        
        for i in range(0, h - self.chunk_size + 1, self.stride):
            for j in range(0, w - self.chunk_size + 1, self.stride):
                chunk = mat_padded[i:i + self.chunk_size, j:j + self.chunk_size]
                chunks.append([chunk])
                indices.append((i, j))
        
        if len(chunks) == 0:
            return tile_norm * tile_max
        
        chunks = np.array(chunks, dtype=np.float32)
        indices = np.array(indices, dtype=np.int32)
        
        # Enhance chunks
        dataset = SimpleChunkDataset(chunks)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        enhanced_chunks = []
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch)
            enhanced_chunks.append(out.cpu().numpy())
            del batch, out
        
        enhanced_chunks = np.concatenate(enhanced_chunks, axis=0)
        del chunks
        
        # Reconstruct
        original_size = tile.shape[0]
        padded_size = original_size + 2 * pad_len
        reconstructed = np.zeros((padded_size, padded_size), dtype=np.float32)
        counts = np.zeros((padded_size, padded_size), dtype=np.float32)
        
        for chunk, (i, j) in zip(enhanced_chunks, indices):
            if len(chunk.shape) == 3:
                chunk = chunk[0]
            reconstructed[i:i + self.chunk_size, j:j + self.chunk_size] += chunk
            counts[i:i + self.chunk_size, j:j + self.chunk_size] += 1
        
        counts[counts == 0] = 1
        reconstructed /= counts
        
        result = reconstructed[pad_len:pad_len + original_size, pad_len:pad_len + original_size]
        result = (result + result.T) / 2
        
        # Scale back to original range
        return result * tile_max
    
    def _process_chromosome(
        self,
        mcool_path: str,
        chrom: str,
        resolution: int,
        output_file: str,
        tile_size: int = 500,
        batch_size: int = 8
    ) -> bool:
        """Process a single chromosome using tiled approach"""
        try:
            clr = cooler.Cooler(f"{mcool_path}::/resolutions/{resolution}")
            matrix = clr.matrix(balance=False).fetch(chrom).astype(np.float32)
            chrom_size = clr.chromsizes[chrom]
            n_bins = matrix.shape[0]
            
            if matrix.sum() == 0:
                print(f"  {chrom}: Skipped (empty)")
                return False
            
            # Process in tiles along diagonal
            enhanced = np.zeros_like(matrix)
            counts = np.zeros_like(matrix)
            
            overlap = tile_size // 4
            n_tiles = (n_bins + tile_size - overlap - 1) // (tile_size - overlap)
            print(f"  {chrom}: {n_bins} bins, {n_tiles} tiles")
            
            for t in range(n_tiles):
                start = t * (tile_size - overlap)
                end = min(start + tile_size, n_bins)
                
                if end - start < self.chunk_size:
                    continue
                
                # Extract tile
                tile = matrix[start:end, start:end].copy()
                tile_max = max(tile.max(), 1.0)
                
                # Enhance tile
                enhanced_tile = self._enhance_tile(tile, tile_max, batch_size)
                
                # Add to result with blending
                tile_h = end - start
                enhanced[start:end, start:end] += enhanced_tile[:tile_h, :tile_h]
                counts[start:end, start:end] += 1
                
                del tile, enhanced_tile
                gc.collect()
            
            # Average overlapping regions
            counts[counts == 0] = 1
            enhanced /= counts
            
            # Make symmetric
            enhanced = (enhanced + enhanced.T) / 2
            
            del matrix, counts
            gc.collect()
            
            # Create bins
            bins = []
            for i in range(n_bins):
                bins.append({
                    'chrom': chrom,
                    'start': i * resolution,
                    'end': min((i + 1) * resolution, chrom_size)
                })
            bins_df = pd.DataFrame(bins)
            
            # Extract pixels (upper triangle)
            rows, cols = np.triu_indices(n_bins)
            values = enhanced[rows, cols]
            mask = values > 0.5
            
            pixels = []
            for r, c, v in zip(rows[mask], cols[mask], values[mask]):
                pixels.append({
                    'bin1_id': int(r),
                    'bin2_id': int(c),
                    'count': int(round(v))
                })
            
            n_pixels = len(pixels)
            
            del enhanced, rows, cols, values, mask
            gc.collect()
            
            if n_pixels == 0:
                print(f"  {chrom}: Skipped (no pixels)")
                return False
            
            pixels_df = pd.DataFrame(pixels)
            del pixels
            gc.collect()
            
            # Save to cooler
            cooler.create_cooler(
                output_file,
                bins_df,
                pixels_df,
                ordered=True,
                symmetric_upper=True
            )
            
            del bins_df, pixels_df
            gc.collect()
            
            print(f"  {chrom}: Done ({n_bins} bins, {n_pixels} pixels)")
            return True
            
        except Exception as e:
            print(f"  {chrom}: Error - {str(e)}")
            return False
    
    def _merge_coolers(self, cooler_files: List[str], output_path: str, resolution: int):
        """Merge chromosome coolers into final mcool"""
        print(f"\nMerging {len(cooler_files)} chromosome files...")
        
        all_bins = []
        all_pixels = []
        bin_offset = 0
        
        for cool_file in cooler_files:
            clr = cooler.Cooler(cool_file)
            bins = clr.bins()[:]
            n_bins = len(bins)
            
            for _, row in bins.iterrows():
                all_bins.append({
                    'chrom': row['chrom'],
                    'start': row['start'],
                    'end': row['end']
                })
            
            pixels = clr.pixels()[:]
            for _, row in pixels.iterrows():
                all_pixels.append({
                    'bin1_id': int(row['bin1_id'] + bin_offset),
                    'bin2_id': int(row['bin2_id'] + bin_offset),
                    'count': int(row['count'])
                })
            
            bin_offset += n_bins
            del bins, pixels
            gc.collect()
        
        print(f"  Total: {len(all_bins)} bins, {len(all_pixels)} pixels")
        
        if os.path.exists(output_path):
            os.remove(output_path)
        
        cooler.create_cooler(
            f"{output_path}::/resolutions/{resolution}",
            pd.DataFrame(all_bins),
            pd.DataFrame(all_pixels),
            ordered=True,
            symmetric_upper=True
        )
        print(f"  Created: {output_path}")
    
    def enhance(
        self,
        input_mcool: str,
        output_mcool: str,
        resolution: int = 25000,
        chromosomes: Optional[List[str]] = None,
        tile_size: int = 500,
        batch_size: int = 8
    ):
        """
        Enhance a Hi-C mcool file
        
        Args:
            input_mcool: Path to input .mcool file
            output_mcool: Path to output enhanced .mcool file
            resolution: Resolution to process (default 25000)
            chromosomes: List of chromosomes to process (None = all autosomes + chrX)
            tile_size: Size of tiles for processing large chromosomes (default 500)
            batch_size: Batch size for inference (default 8)
        """
        print(f"\n{'='*60}")
        print("HI-C ENHANCEMENT")
        print(f"{'='*60}")
        print(f"Input: {input_mcool}")
        print(f"Output: {output_mcool}")
        print(f"Resolution: {resolution}")
        print(f"Tile size: {tile_size}")
        
        clr = cooler.Cooler(f"{input_mcool}::/resolutions/{resolution}")
        
        if chromosomes is None:
            # Process main chromosomes only (skip unplaced contigs)
            chromosomes = [c for c in clr.chromnames 
                         if c.startswith('chr') 
                         and '_' not in c 
                         and c not in ['chrM', 'chrY']]
        
        print(f"Chromosomes: {len(chromosomes)}")
        print(f"{'='*60}\n")
        
        # Create temp directory for intermediate files
        temp_dir = tempfile.mkdtemp(prefix="hic_enhance_")
        
        try:
            cooler_files = []
            
            for chrom in chromosomes:
                print(f"Processing {chrom}...")
                output_file = os.path.join(temp_dir, f"{chrom}.cool")
                
                success = self._process_chromosome(
                    input_mcool, chrom, resolution, output_file, tile_size, batch_size
                )
                
                if success:
                    cooler_files.append(output_file)
                
                gc.collect()
            
            if len(cooler_files) == 0:
                raise RuntimeError("No chromosomes were successfully processed!")
            
            # Merge all chromosome coolers
            self._merge_coolers(cooler_files, output_mcool, resolution)
            
            print(f"\n{'='*60}")
            print("ENHANCEMENT COMPLETE!")
            print(f"{'='*60}")
            print(f"Output saved to: {output_mcool}")
            print(f"{'='*60}\n")
            
        finally:
            # Clean up temp files
            shutil.rmtree(temp_dir, ignore_errors=True)


def enhance_hic(
    model_path: str,
    input_mcool: str,
    output_mcool: str,
    resolution: int = 25000,
    chromosomes: Optional[List[str]] = None,
    tile_size: int = 500,
    batch_size: int = 8,
    device: str = 'auto'
):
    """
    Convenience function to enhance a Hi-C mcool file
    
    Args:
        model_path: Path to trained model checkpoint
        input_mcool: Path to input mcool file
        output_mcool: Path to output enhanced mcool file
        resolution: Resolution to process (default 25000)
        chromosomes: List of chromosomes (None = all autosomes + chrX)
        tile_size: Tile size for memory-efficient processing (default 500)
        batch_size: Batch size for inference (default 8)
        device: Device to use ('auto', 'cuda', 'cpu')
    """
    enhancer = HiCEnhancer(model_path, device=device)
    enhancer.enhance(
        input_mcool,
        output_mcool,
        resolution,
        chromosomes,
        tile_size,
        batch_size
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhance Hi-C contact matrices using deep learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enhance all chromosomes
  python inference.py --model best_model.pth --input data.mcool --output enhanced.mcool
  
  # Enhance specific chromosomes
  python inference.py --model best_model.pth --input data.mcool --output enhanced.mcool --chromosomes chr21 chr22
  
  # Use smaller batch size for limited memory
  python inference.py --model best_model.pth --input data.mcool --output enhanced.mcool --batch-size 4
        """
    )
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--input', required=True, help='Input mcool file')
    parser.add_argument('--output', required=True, help='Output enhanced mcool file')
    parser.add_argument('--resolution', type=int, default=25000, help='Resolution to process (default: 25000)')
    parser.add_argument('--tile-size', type=int, default=500, help='Tile size for processing (default: 500)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference (default: 8)')
    parser.add_argument('--chromosomes', nargs='+', default=None, help='Chromosomes to process (default: all)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    enhance_hic(
        model_path=args.model,
        input_mcool=args.input,
        output_mcool=args.output,
        resolution=args.resolution,
        chromosomes=args.chromosomes,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        device=args.device
    )
