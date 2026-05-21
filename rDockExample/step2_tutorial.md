# Step 2: Basic Scoring Function

## Overview
In this step, we added a physics-based scoring function to evaluate the quality of our molecular structure. We implemented the **Lennard-Jones potential**, which approximates the interaction between a pair of neutral atoms or molecules.

## The Math
The Lennard-Jones potential is defined as:
$$ V_{LJ} = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right] $$

*   **$r$**: Distance between two atoms.
*   **$\epsilon$ (epsilon)**: Depth of the potential well (strength of attraction).
*   **$\sigma$ (sigma)**: Distance at which the potential is zero (collision diameter).

## Files Created/Modified
*   **`scoring.py`**: New class `ScoringFunction`.
    *   `calculate_distance`: Euclidean distance between two atoms.
    *   `lennard_jones`: Implements the formula above.
    *   `score_molecule`: Loops through all pairs of atoms in a molecule and sums their interaction energies.
*   **`main.py`**: Updated to instantiate `ScoringFunction` and calculate the energy of the water molecule.

## How to Run
1.  Navigate to the `implementation` directory.
2.  Run the following command:
    ```bash
    python3 main.py
    ```

## Output Explanation
You will see pairwise calculations for the atoms in Water.
*   **O-H interactions**: These are bonded, so in a real forcefield we would ignore them or treat them as springs. Here, they show a very high positive energy (repulsion) because 0.96 A is much closer than the default Van der Waals radius ($\sigma=3.0$). This correctly simulates that atoms "clash" if they get too close without a chemical bond.
*   **H-H interaction**: The two hydrogens are further apart, showing a different energy.

## Next Steps
Real molecular docking involves **moving** the ligand to find the minimum energy. The next step is to implement a **Transform** class to translate and rotate the molecule.
