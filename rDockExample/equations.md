# Mathematical Equations for rDock Reconstruction

This document summarizes the key mathematical equations used in our implementation of the molecular docking engine.

## 1. Geometry

### Euclidean Distance
The distance $r_{ij}$ between two atoms $i$ and $j$ with coordinates $(x_i, y_i, z_i)$ and $(x_j, y_j, z_j)$ is calculated as:

$$
r_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2}
$$

## 2. Physics (Scoring Function)

### Lennard-Jones Potential
We use the Lennard-Jones 12-6 potential to approximate the interaction between a pair of neutral atoms. This models both the attractive Van der Waals force and the repulsive Pauli exclusion principle.

$$
V_{LJ}(r_{ij}) = 4\epsilon \left[ \left(\frac{\sigma}{r_{ij}}\right)^{12} - \left(\frac{\sigma}{r_{ij}}\right)^6 \right]
$$

Where:
*   $V_{LJ}$: Potential energy.
*   $r_{ij}$: Distance between the atoms.
*   $\epsilon$ (epsilon): Depth of the potential well (strength of attraction).
*   $\sigma$ (sigma): Finite distance at which the inter-particle potential is zero.

### Total Energy
The total energy of the system is the sum of all pairwise interactions. In our docking simulation, we separate this into intra-molecular (ligand-ligand) and inter-molecular (ligand-receptor) terms.

$$
E_{total} = \sum_{i \in Ligand} \sum_{j \in Ligand, j > i} V_{LJ}(r_{ij}) + \sum_{i \in Ligand} \sum_{k \in Receptor} V_{LJ}(r_{ik})
$$

## 3. Transformations (Movement)

### Translation
To translate a molecule by a vector $\vec{v} = (dx, dy, dz)$, we update the coordinates of every atom $i$:

$$
\begin{bmatrix} x'_i \\ y'_i \\ z'_i \end{bmatrix} = \begin{bmatrix} x_i \\ y_i \\ z_i \end{bmatrix} + \begin{bmatrix} dx \\ dy \\ dz \end{bmatrix}
$$

### Rotation (Z-Axis)
To rotate a molecule by an angle $\theta$ around the Z-axis, we apply the following rotation matrix to every atom $i$:

$$
\begin{bmatrix} x'_i \\ y'_i \\ z'_i \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_i \\ y_i \\ z_i \end{bmatrix}
$$

Which simplifies to:
$$
x'_i = x_i \cos\theta - y_i \sin\theta
$$
$$
y'_i = x_i \sin\theta + y_i \cos\theta
$$
$$
z'_i = z_i
$$

## 4. Optimization (Monte Carlo)

### Metropolis Criterion
In the Metropolis-Hastings algorithm, a new state with energy $E_{new}$ is accepted or rejected based on the change in energy $\Delta E = E_{new} - E_{old}$ and the temperature $T$.

The probability of acceptance $P_{accept}$ is:

$$
P_{accept} = \begin{cases} 
1 & \text{if } \Delta E < 0 \\
e^{-\frac{\Delta E}{k_B T}} & \text{if } \Delta E \ge 0 
\end{cases}
$$

Where:
*   $\Delta E$: Change in energy.
*   $T$: Temperature (controls the probability of accepting bad moves).
*   $k_B$: Boltzmann constant (often set to 1 in simplified simulations).
