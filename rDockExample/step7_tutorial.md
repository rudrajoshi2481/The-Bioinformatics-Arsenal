# Step 7: Gradient Descent (Using Calculus!)

## Overview
In this step, we added **Gradient Descent**, a calculus-based optimization method similar to what AutoDock Vina uses. This allows us to find **local minima** much faster than Monte Carlo alone.

## The Key Difference

### Monte Carlo (No Calculus)
*   **Random Search**: Try random moves and keep good ones.
*   **Pro**: Can escape local minima (finds global minimum).
*   **Con**: Slow to converge.

### Gradient Descent (Uses Calculus)
*   **Directed Search**: Calculate which direction is "downhill" and move that way.
*   **Pro**: Very fast convergence to nearest minimum.
*   **Con**: Gets stuck in local minima.

## The Math: Force = Derivative of Energy

The **Force** on an atom is the negative derivative of the potential energy:

$$
\vec{F} = -\nabla V = -\frac{dV}{dr}
$$

For the Lennard-Jones potential:
$$
V(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]
$$

The force magnitude is:
$$
F(r) = \frac{24\epsilon}{r} \left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]
$$

## Files Created/Modified
*   **`scoring.py`**: Added `lennard_jones_force()` and `calculate_gradient()`.
*   **`gradient_descent.py`**: New class `GradientDescent`.
*   **`main.py`**: Updated to use **Hybrid Approach**:
    1.  **Phase 1**: Monte Carlo (global search).
    2.  **Phase 2**: Gradient Descent (local refinement).

## How to Run
```bash
python3 main.py
```

## Expected Output
You should see:
1.  Monte Carlo finding a rough solution.
2.  Gradient Descent refining it to a precise local minimum.

## Why Hybrid is Best
*   **Monte Carlo** explores the entire landscape (avoids getting stuck).
*   **Gradient Descent** polishes the final answer (fast convergence).

This is exactly what **AutoDock Vina** does!
