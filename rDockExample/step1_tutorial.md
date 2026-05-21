# Step 1: Basic Data Structures (Python)

## Overview
In this step, we implemented the fundamental building blocks of a molecular modeling application: Atoms and Molecules.
**Note**: We are using Python for this implementation because a C++ compiler was not available in the current environment.

## Files
*   **`atom.py`**: Defines the `Atom` class.
    *   Stores element type (string) and 3D coordinates (x, y, z).
    *   Provides methods to print details.
*   **`molecule.py`**: Defines the `Molecule` class.
    *   Acts as a container for `Atom` objects.
    *   Uses a list to store atoms.
    *   Provides methods to add atoms and print the entire molecule structure.
*   **`main.py`**: A test harness.
    *   Creates a `Molecule` object representing Water (H2O).
    *   Adds three `Atom` objects (1 Oxygen, 2 Hydrogens) with approximate coordinates.
    *   Prints the structure to the console.

## How to Run
1.  Navigate to the `implementation` directory.
2.  Run the following command:
    ```bash
    python3 main.py
    ```

## Output
```
=== rDock Reconstruction Step 1: Basic Structures (Python) ===
Molecule: Water
  1: Atom: O (0.0, 0.0, 0.0)
  2: Atom: H (0.96, 0.0, 0.0)
  3: Atom: H (-0.24, 0.93, 0.0)
```
