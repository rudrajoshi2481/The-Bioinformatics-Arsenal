# Step 3: Transforms (Movement)

## Overview
In this step, we implemented the ability to move the molecule in 3D space. This is crucial for docking because the goal is to find the position and orientation of the ligand that minimizes the energy.

## Files Created/Modified
*   **`atom.py`**: Modified `Atom` to use a list `[x, y, z]` for coordinates instead of a tuple, making them mutable.
*   **`transform.py`**: New class `Transform`.
    *   `translate`: Adds a vector $(dx, dy, dz)$ to every atom's coordinates.
    *   `rotate_z`: Rotates every atom around the Z-axis using a rotation matrix.
*   **`main.py`**: Updated to demonstrate moving the water molecule.

## The Math: Rotation Matrix (Z-axis)
To rotate a point $(x, y)$ by an angle $\theta$:
$$ x' = x \cos\theta - y \sin\theta $$
$$ y' = x \sin\theta + y \cos\theta $$
$$ z' = z $$

## Important Concept: Invariance
Notice that when we translate or rotate the **entire** molecule, the **internal energy** (Lennard-Jones score) does not change. This is expected! The distance between atoms *within* the molecule stays the same.
In a real docking scenario, we would be measuring the energy between the **Ligand** and the **Receptor**. Moving the ligand *would* change that interaction energy.

## How to Run
```bash
python3 main.py
```

## Output
You will see the coordinates change, but the energy remains constant.
