import random
import math
import copy

class Optimizer:
    def __init__(self, scoring_function, transform):
        self.sf = scoring_function
        self.tf = transform

    def run(self, ligand, receptor, steps=1000, temp=1.0, step_size=0.1):
        """
        Run a simple Monte Carlo optimization docking Ligand to Receptor.
        """
        print(f"Starting optimization: {steps} steps, T={temp}")
        
        # Calculate initial score (Intra + Inter)
        # For rigid docking, Intra is constant, so we only really care about Inter.
        # But let's track both for completeness.
        current_intra = self.sf.score_molecule(ligand)
        current_inter = self.sf.score_inter(ligand, receptor)
        current_total = current_intra + current_inter
        
        best_score = current_total
        best_coords = copy.deepcopy([atom.coords for atom in ligand.atoms])
        
        accepted = 0
        
        for i in range(steps):
            # 1. Save current state
            old_coords = copy.deepcopy([atom.coords for atom in ligand.atoms])
            
            # 2. Mutate (Random Translation + Rotation)
            # Translation
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            dz = random.uniform(-step_size, step_size)
            self.tf.translate(ligand, dx, dy, dz)
            
            # Rotation (occasionally)
            if random.random() < 0.5:
                angle = random.uniform(-10.0, 10.0) # Small rotation
                self.tf.rotate_z(ligand, angle)
            
            # 3. Score new state
            # Optimization: We know intra score doesn't change for rigid body moves
            # new_intra = self.sf.score_molecule(ligand) 
            new_inter = self.sf.score_inter(ligand, receptor)
            new_total = current_intra + new_inter
            
            # 4. Metropolis Criterion
            delta = new_total - current_total
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_total = new_total
                current_inter = new_inter
                accepted += 1
                
                # Update best found
                if current_total < best_score:
                    best_score = current_total
                    best_coords = copy.deepcopy([atom.coords for atom in ligand.atoms])
            else:
                # Reject: Revert to old coordinates
                for j, atom in enumerate(ligand.atoms):
                    atom.coords = copy.deepcopy(old_coords[j])
                    
        print(f"Optimization finished. Accepted: {accepted}/{steps} ({accepted/steps*100:.1f}%)")
        print(f"Best Score: {best_score:.3f}")
        
        # Restore best coordinates
        for j, atom in enumerate(ligand.atoms):
            atom.coords = best_coords[j]
            
        return best_score
