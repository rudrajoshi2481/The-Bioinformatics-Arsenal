class Atom:
    def __init__(self, element, x, y, z):
        self.element = element
        # Use a list for mutable coordinates
        self.coords = [x, y, z]

    def print_details(self):
        print(f"Atom: {self.element} ({self.coords[0]:.3f}, {self.coords[1]:.3f}, {self.coords[2]:.3f})")
