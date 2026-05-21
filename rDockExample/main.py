from atom import Atom
from molecule import Molecule
from scoring import ScoringFunction
from transform import Transform
from optimizer import Optimizer
from gradient_descent import GradientDescent
from pdb_parser import PDBParser

def main():
    print("=== rDock + Gradient Descent (Calculus!) ===")

    parser = PDBParser()

    # 1. Load Ligand
    print("\n--- Loading Ligand ---")
    ligand = parser.parse("ligand.pdb", "Ligand")
    if not ligand:
        return

    # 2. Load Receptor
    print("\n--- Loading Receptor ---")
    receptor = parser.parse("receptor.pdb", "Receptor")
    if not receptor:
        return

    # Move ligand away
    tf = Transform()
    tf.translate(ligand, 5.0, 5.0, 5.0)

    sf = ScoringFunction()

    print("\n--- PHASE 1: Monte Carlo (Global Search) ---")
    opt = Optimizer(sf, tf)
    opt.run(ligand, receptor, steps=500, temp=1.0, step_size=0.5)

    print("\n--- PHASE 2: Gradient Descent (Local Refinement) ---")
    # Save the state before gradient descent
    ligand_copy = copy.deepcopy(ligand)
    
    gd = GradientDescent(sf)
    gd.run(ligand, receptor, steps=50, learning_rate=0.01)

    print("\n--- Final Comparison ---")
    print(f"After Monte Carlo: {sf.score_inter(ligand_copy, receptor):.3f}")
    print(f"After Gradient Descent: {sf.score_inter(ligand, receptor):.3f}")

if __name__ == "__main__":
    import copy
    main()
