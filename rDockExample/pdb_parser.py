from atom import Atom
from molecule import Molecule

class PDBParser:
    def __init__(self):
        pass

    def parse(self, filename, molecule_name="Unknown"):
        """
        Parses a PDB file and returns a Molecule object.
        """
        mol = Molecule(molecule_name)
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        # PDB Fixed Width Format
                        # 13-16: Atom Name
                        # 31-38: X
                        # 39-46: Y
                        # 47-54: Z
                        # 77-78: Element (often missing or in different col, we'll try to guess from name)
                        
                        atom_name = line[12:16].strip()
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        
                        # Simple element guessing
                        element = atom_name[0]
                        if len(atom_name) > 1 and atom_name[1].islower():
                             element = atom_name[0:2]
                             
                        mol.add_atom(Atom(element, x, y, z))
                        
            print(f"Successfully parsed {filename}: {len(mol.atoms)} atoms.")
            return mol
            
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
            return None
        except Exception as e:
            print(f"Error parsing PDB file: {e}")
            return None
