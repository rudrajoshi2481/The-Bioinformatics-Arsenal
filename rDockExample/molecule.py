from atom import Atom

class Molecule:
    def __init__(self, name):
        self.name = name
        self.atoms = []

    def add_atom(self, atom):
        self.atoms.append(atom)

    def print_details(self):
        print(f"Molecule: {self.name}")
        for i, atom in enumerate(self.atoms):
            print(f"  {i + 1}: ", end="")
            atom.print_details()
