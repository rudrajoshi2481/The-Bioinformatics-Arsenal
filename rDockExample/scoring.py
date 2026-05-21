import math

class ScoringFunction:
    def __init__(self):
        pass

    def calculate_distance(self, atom1, atom2):
        dx = atom1.coords[0] - atom2.coords[0]
        dy = atom1.coords[1] - atom2.coords[1]
        dz = atom1.coords[2] - atom2.coords[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def lennard_jones(self, r, epsilon=0.1, sigma=3.0):
        """
        Calculate Lennard-Jones potential: 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
        """
        if r == 0:
            return float('inf')
        
        term6 = (sigma / r) ** 6
        term12 = term6 * term6
        return 4 * epsilon * (term12 - term6)

    def lennard_jones_force(self, r, epsilon=0.1, sigma=3.0):
        """
        Calculate the FORCE (derivative of LJ potential).
        Force = -dV/dr = (24*epsilon/r) * [2*(sigma/r)^12 - (sigma/r)^6]
        Returns the magnitude of force (positive = repulsive, negative = attractive)
        """
        if r == 0:
            return 0.0
        
        term6 = (sigma / r) ** 6
        term12 = term6 * term6
        force = (24 * epsilon / r) * (2 * term12 - term6)
        return force

    def score_molecule(self, molecule):
        """
        Calculate intra-molecular score (sum of all pairwise interactions)
        """
        total_energy = 0.0
        atoms = molecule.atoms
        num_atoms = len(atoms)
        
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                atom1 = atoms[i]
                atom2 = atoms[j]
                
                dist = self.calculate_distance(atom1, atom2)
                energy = self.lennard_jones(dist)
                total_energy += energy
                
        return total_energy

    def score_inter(self, ligand, receptor):
        """
        Calculate inter-molecular score (Ligand vs Receptor)
        """
        total_energy = 0.0
        
        for l_atom in ligand.atoms:
            for r_atom in receptor.atoms:
                dist = self.calculate_distance(l_atom, r_atom)
                
                if dist > 10.0:
                    continue
                    
                energy = self.lennard_jones(dist)
                total_energy += energy
                
        return total_energy

    def calculate_gradient(self, ligand, receptor):
        """
        Calculate the GRADIENT (force) on each ligand atom.
        Returns a list of [fx, fy, fz] for each atom.
        This is the CALCULUS part!
        """
        gradients = []
        
        for l_atom in ligand.atoms:
            fx, fy, fz = 0.0, 0.0, 0.0
            
            # Calculate force from receptor atoms
            for r_atom in receptor.atoms:
                dx = l_atom.coords[0] - r_atom.coords[0]
                dy = l_atom.coords[1] - r_atom.coords[1]
                dz = l_atom.coords[2] - r_atom.coords[2]
                
                r = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                if r > 10.0 or r == 0:
                    continue
                
                # Get force magnitude
                force_mag = self.lennard_jones_force(r)
                
                # Convert to force vector (direction)
                fx += force_mag * (dx / r)
                fy += force_mag * (dy / r)
                fz += force_mag * (dz / r)
            
            gradients.append([fx, fy, fz])
        
        return gradients
