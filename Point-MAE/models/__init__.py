"""
Point-MAE Models Package

This package contains the modular components of Point-MAE:
- utils: Helper functions (FPS, KNN, initialization)
- encoder: Point cloud group encoder
- attention: Multi-head self-attention
- transformer: Transformer encoder and decoder
- point_mae: Main PointMAE model
- objects: Random 3D object generator
"""

from .point_mae import PointMAE
from .objects import generate_random_object

__all__ = ['PointMAE', 'generate_random_object']
