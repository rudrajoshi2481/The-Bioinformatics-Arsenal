# Step 6: Parsing Real Data (PDB)

## Overview
In this step, we moved away from hardcoded atoms and implemented a **PDB Parser**. This allows us to load real molecular structures from Protein Data Bank (PDB) files, which is the standard format for structural biology.

## Files Created/Modified
*   **`pdb_parser.py`**: New class `PDBParser`.
    *   Reads standard PDB format (fixed width columns).
    *   Extracts Atom Name and Coordinates (X, Y, Z).
    *   Creates `Molecule` and `Atom` objects.
*   **`ligand.pdb`**: A sample ligand file (created manually).
*   **`receptor.pdb`**: A sample receptor file (created manually).
*   **`main.py`**: Updated to load these files instead of creating atoms manually.

## Why PDB?
The PDB format is ancient but ubiquitous.
```
ATOM      1  C   LIG     1       0.000   0.000   0.000  1.00  0.00           C
```
*   **Cols 31-38**: X coordinate
*   **Cols 39-46**: Y coordinate
*   **Cols 47-54**: Z coordinate

## How to Run
```bash
python3 main.py
```

## Expected Output
The program will:
1.  Read `ligand.pdb` and `receptor.pdb`.
2.  Translate the ligand away from the origin.
3.  Run the docking optimization to bring them back together.
