import math

class Transform:
    def __init__(self):
        pass

    def translate(self, molecule, dx, dy, dz):
        """
        Translate the entire molecule by (dx, dy, dz)
        """
        print(f"Translating by ({dx}, {dy}, {dz})...")
        for atom in molecule.atoms:
            atom.coords[0] += dx
            atom.coords[1] += dy
            atom.coords[2] += dz

    def rotate_z(self, molecule, angle_degrees):
        """
        Rotate the molecule around the Z-axis by angle_degrees
        """
        print(f"Rotating by {angle_degrees} degrees around Z-axis...")
        radians = math.radians(angle_degrees)
        cos_theta = math.cos(radians)
        sin_theta = math.sin(radians)

        for atom in molecule.atoms:
            x = atom.coords[0]
            y = atom.coords[1]
            
            # Rotation matrix for Z-axis
            # x' = x*cos - y*sin
            # y' = x*sin + y*cos
            new_x = x * cos_theta - y * sin_theta
            new_y = x * sin_theta + y * cos_theta
            
            atom.coords[0] = new_x
            atom.coords[1] = new_y
