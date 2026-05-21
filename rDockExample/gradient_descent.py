import copy

class GradientDescent:
    """
    Gradient Descent optimizer - uses CALCULUS to find local minimum.
    This is similar to what AutoDock Vina uses.
    """
    def __init__(self, scoring_function):
        self.sf = scoring_function

    def run(self, ligand, receptor, steps=100, learning_rate=0.01):
        """
        Run gradient descent optimization.
        Uses the GRADIENT (force) to move atoms downhill.
        """
        print(f"Starting Gradient Descent: {steps} steps, lr={learning_rate}")
        
        initial_energy = self.sf.score_inter(ligand, receptor)
        print(f"Initial Energy: {initial_energy:.3f}")
        
        for i in range(steps):
            # Calculate gradient (CALCULUS!)
            gradients = self.sf.calculate_gradient(ligand, receptor)
            
            # Move each atom in the direction of the negative gradient
            for j, atom in enumerate(ligand.atoms):
                fx, fy, fz = gradients[j]
                
                # Gradient Descent: x_new = x_old - learning_rate * gradient
                atom.coords[0] -= learning_rate * fx
                atom.coords[1] -= learning_rate * fy
                atom.coords[2] -= learning_rate * fz
            
            # Check energy every 10 steps
            if (i + 1) % 10 == 0:
                current_energy = self.sf.score_inter(ligand, receptor)
                print(f"Step {i+1}: Energy = {current_energy:.3f}")
        
        final_energy = self.sf.score_inter(ligand, receptor)
        print(f"Final Energy: {final_energy:.3f}")
        
        return final_energy
