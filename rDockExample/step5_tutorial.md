# Step 5: Receptor & Inter-molecular Scoring

## Overview
In this step, we introduced the concept of a **Receptor** and implemented **Inter-molecular Scoring**. This allows us to simulate the actual docking process where a ligand searches for the best binding pose on a receptor.

## Changes
1.  **`scoring.py`**: Added `score_inter(ligand, receptor)`.
    *   Iterates through all pairs of (ligand atom, receptor atom).
    *   Calculates Lennard-Jones potential.
    *   This creates an "energy landscape" where the ligand is attracted to the receptor (negative energy) but repelled if it gets too close (positive energy).
2.  **`optimizer.py`**: Updated `run` method to accept a receptor.
    *   The objective function is now `Total Energy = Intra + Inter`.
    *   Since `Intra` is constant for rigid bodies, the optimization is driven entirely by `Inter`.
3.  **`main.py`**:
    *   Created a "Receptor" molecule consisting of 3 Carbon atoms centered at the origin.
    *   Placed the Ligand (Water) far away at (5, 5, 5).
    *   Ran the optimization.

## The Physics of Docking
*   **Attraction**: The Van der Waals force (Lennard-Jones) attracts the ligand to the receptor.
*   **Repulsion**: If atoms overlap, the energy skyrockets, preventing collisions.
*   **Binding Site**: The "pocket" we created (atoms at -2, 0, +2) creates a favorable energy well near the origin.

## How to Run
```bash
python3 main.py
```

## Expected Output
*   **Initial Energy**: Should be close to 0 (atoms are far apart).
*   **Optimization**: You should see the acceptance rate vary.
*   **Final Energy**: Should be negative (indicating binding).
*   **Final Position**: The ligand's Oxygen atom should end up somewhere near the receptor atoms (e.g., distance ~3.0-4.0 Angstroms), showing that it has "docked".
