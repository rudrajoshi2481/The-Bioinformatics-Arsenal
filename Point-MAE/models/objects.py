"""
Random 3D Object Generator for Point-MAE testing.

Generates various 3D shapes as point clouds with normals.
"""

import numpy as np
import random


def generate_random_object(num_points=8192, obj_type='random', with_normals=True):
    """
    Generate random 3D objects for testing.
    
    Args:
        num_points: number of points
        obj_type: 'sphere', 'cube', 'cylinder', 'torus', 'random', 'airplane', 'chair'
        with_normals: if True, return (N, 6) with normals, else (N, 3)
        
    Returns:
        points: (N, 6) or (N, 3) point cloud
    """
    if obj_type == 'random':
        obj_type = random.choice(['sphere', 'cube', 'cylinder', 'torus', 'airplane', 'chair'])
    
    if obj_type == 'sphere':
        phi = np.random.uniform(0, 2 * np.pi, num_points)
        costheta = np.random.uniform(-1, 1, num_points)
        theta = np.arccos(costheta)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        points = np.stack([x, y, z], axis=1)
        normals = points.copy()
        
    elif obj_type == 'cube':
        points = []
        normals = []
        
        for _ in range(num_points):
            face = np.random.randint(6)
            u, v = np.random.uniform(-1, 1, 2)
            
            if face == 0:
                p, n = [1, u, v], [1, 0, 0]
            elif face == 1:
                p, n = [-1, u, v], [-1, 0, 0]
            elif face == 2:
                p, n = [u, 1, v], [0, 1, 0]
            elif face == 3:
                p, n = [u, -1, v], [0, -1, 0]
            elif face == 4:
                p, n = [u, v, 1], [0, 0, 1]
            else:
                p, n = [u, v, -1], [0, 0, -1]
            
            points.append(p)
            normals.append(n)
        
        points = np.array(points)
        normals = np.array(normals)
        
    elif obj_type == 'cylinder':
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        z = np.random.uniform(-1, 1, num_points)
        
        x = np.cos(theta)
        y = np.sin(theta)
        
        points = np.stack([x, y, z], axis=1)
        normals = np.stack([np.cos(theta), np.sin(theta), np.zeros(num_points)], axis=1)
        
    elif obj_type == 'torus':
        R, r = 0.7, 0.3
        
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        phi = np.random.uniform(0, 2 * np.pi, num_points)
        
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        
        points = np.stack([x, y, z], axis=1)
        
        nx = np.cos(phi) * np.cos(theta)
        ny = np.cos(phi) * np.sin(theta)
        nz = np.sin(phi)
        normals = np.stack([nx, ny, nz], axis=1)
        
    elif obj_type == 'airplane':
        points = []
        normals = []
        
        # Fuselage
        n_fuse = num_points // 2
        theta = np.random.uniform(0, 2 * np.pi, n_fuse)
        x = np.random.uniform(-1, 1, n_fuse)
        y = 0.15 * np.cos(theta)
        z = 0.15 * np.sin(theta)
        points.extend(np.stack([x, y, z], axis=1).tolist())
        normals.extend(np.stack([np.zeros(n_fuse), np.cos(theta), np.sin(theta)], axis=1).tolist())
        
        # Wings
        n_wing = num_points // 4
        x = np.random.uniform(-0.3, 0.3, n_wing)
        y = np.random.uniform(-1, 1, n_wing)
        z = np.zeros(n_wing) + np.random.uniform(-0.02, 0.02, n_wing)
        points.extend(np.stack([x, y, z], axis=1).tolist())
        normals.extend(np.stack([np.zeros(n_wing), np.zeros(n_wing), np.ones(n_wing)], axis=1).tolist())
        
        # Tail
        n_tail = num_points - n_fuse - n_wing
        x = np.random.uniform(0.7, 1, n_tail)
        y = np.zeros(n_tail) + np.random.uniform(-0.02, 0.02, n_tail)
        z = np.random.uniform(0, 0.4, n_tail)
        points.extend(np.stack([x, y, z], axis=1).tolist())
        normals.extend(np.stack([np.zeros(n_tail), np.ones(n_tail), np.zeros(n_tail)], axis=1).tolist())
        
        points = np.array(points)
        normals = np.array(normals)
        
    elif obj_type == 'chair':
        points = []
        normals = []
        
        # Seat
        n_seat = num_points // 3
        x = np.random.uniform(-0.4, 0.4, n_seat)
        y = np.random.uniform(-0.4, 0.4, n_seat)
        z = np.zeros(n_seat) + np.random.uniform(-0.02, 0.02, n_seat)
        points.extend(np.stack([x, y, z], axis=1).tolist())
        normals.extend(np.stack([np.zeros(n_seat), np.zeros(n_seat), np.ones(n_seat)], axis=1).tolist())
        
        # Back
        n_back = num_points // 3
        x = np.random.uniform(-0.4, 0.4, n_back)
        y = np.ones(n_back) * 0.4 + np.random.uniform(-0.02, 0.02, n_back)
        z = np.random.uniform(0, 0.8, n_back)
        points.extend(np.stack([x, y, z], axis=1).tolist())
        normals.extend(np.stack([np.zeros(n_back), np.ones(n_back), np.zeros(n_back)], axis=1).tolist())
        
        # Legs
        n_legs = num_points - n_seat - n_back
        n_per_leg = n_legs // 4
        
        for lx, ly in [(-0.35, -0.35), (-0.35, 0.35), (0.35, -0.35), (0.35, 0.35)]:
            n_this = n_per_leg if lx != 0.35 or ly != 0.35 else n_legs - 3 * n_per_leg
            theta = np.random.uniform(0, 2 * np.pi, n_this)
            x = lx + 0.05 * np.cos(theta)
            y = ly + 0.05 * np.sin(theta)
            z = np.random.uniform(-0.6, 0, n_this)
            points.extend(np.stack([x, y, z], axis=1).tolist())
            normals.extend(np.stack([np.cos(theta), np.sin(theta), np.zeros(n_this)], axis=1).tolist())
        
        points = np.array(points)
        normals = np.array(normals)
    
    else:
        raise ValueError(f"Unknown object type: {obj_type}")
    
    # Normalize to unit sphere
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    
    # Normalize normals
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1)
    normals = normals / norm
    
    if with_normals:
        return np.hstack([points, normals]).astype(np.float32)
    else:
        return points.astype(np.float32)
