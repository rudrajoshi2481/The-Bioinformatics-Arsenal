# Step 4: Optimization Loop

## Overview
In this step, we implemented the engine of the docking program: the **Optimization Loop**. We used a **Monte Carlo** approach, which is a randomized search algorithm.

## The Algorithm
1.  **Perturb**: Make a small random change to the molecule (translation/rotation).
2.  **Score**: Calculate the energy of the new state.
3.  **Decide**:
    *   If the energy is lower (better), **accept** the new state.
    *   If the energy is higher (worse), **accept** it with a probability $P = e^{-\Delta E / T}$. This allows the system to escape local minima.
    *   Otherwise, **reject** and revert to the previous state.

## Files Created/Modified
*   **`optimizer.py`**: New class `Optimizer`.
    *   `run`: Implements the Monte Carlo loop.
    *   Uses `copy.deepcopy` to save and restore coordinates.
*   **`main.py`**: Updated to instantiate `Optimizer` and run the simulation.

## A Note on this Demo
In this simplified example, we are only moving a single rigid water molecule. As we learned in Step 3, rigid body movement does not change the internal Lennard-Jones energy. Therefore, this optimization is effectively a **Random Walk**—it will accept every move because $\Delta E = 0$.
In a real docking scenario, there would be a **Receptor** (protein) that stays fixed. Moving the ligand would change the distance to the receptor atoms, thus changing the energy and driving the optimization toward the binding pocket.

## How to Run
```bash
python3 main.py
```

## Output
You will see the loop running and the coordinates changing randomly. The acceptance rate should be 100% (or close to it) because the energy is constant.
