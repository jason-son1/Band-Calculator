"""
Automated Brillouin Zone & K-Path Analyzer.

Determines the reciprocal lattice, identifies the Bravais lattice type based on
crystallographic symmetry, and automatically generates standard high-symmetry K-paths.
"""

import numpy as np

# ─── Standard High Symmetry Fractional Coordinates ────────────────────
# These are given in fractional coordinates of the reciprocal lattice vectors (b1, b2, b3).

BZ_PATHS = {
    "1d": {
        "points": {"-π": [-0.5, 0, 0], "Γ": [0.0, 0, 0], "π": [0.5, 0, 0]},
        "path": ["-π", "Γ", "π"]
    },
    "square": {
        "points": {"Γ": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.0], "M": [0.5, 0.5, 0.0]},
        "path": ["Γ", "X", "M", "Γ"]
    },
    "rectangular": {
        "points": {"Γ": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.0], "M": [0.5, 0.5, 0.0], "Y": [0.0, 0.5, 0.0]},
        "path": ["Γ", "X", "M", "Y", "Γ"]
    },
    "hexagonal": {
        # Standard Hexagonal (angle 120 between a1, a2)
        "points": {"Γ": [0.0, 0.0, 0.0], "M": [0.5, 0.0, 0.0], "K": [1/3, 1/3, 0.0]},
        "path": ["Γ", "M", "K", "Γ"]
    },
    "cubic": {
        "points": {"Γ": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.0], "M": [0.5, 0.5, 0.0], "R": [0.5, 0.5, 0.5]},
        "path": ["Γ", "X", "M", "Γ", "R", "X"]
    },
}


class BrillouinZoneAnalyzer:
    """
    Analyzes the Lattice to compute the Reciprocal Lattice and automatically
    determines the Brillouin Zone high-symmetry path.
    """

    def __init__(self, lattice):
        self.dim = lattice.dimension
        
        # 1. Parse primitive vectors
        a1 = np.array(lattice.a1, dtype=float)
        a2 = np.array(lattice.a2, dtype=float)
        a3 = np.array(lattice.a3, dtype=float)
        
        # For 1D/2D, ensure orthogonal padding to allow 3D cross products
        if self.dim == 1:
            a2 = np.array([0.0, 1.0, 0.0])
            a3 = np.array([0.0, 0.0, 1.0])
        elif self.dim == 2:
            a3 = np.array([0.0, 0.0, 1.0])

        self.a1, self.a2, self.a3 = a1, a2, a3

        # 2. Compute Volume
        self.volume = np.dot(a1, np.cross(a2, a3))
        if np.isclose(self.volume, 0):
            # Fallback for degenerate lattices
            self.b1, self.b2, self.b3 = a1, a2, a3
            self.lattice_type = "unknown"
            return

        # 3. Compute Reciprocal Lattice Vectors (b_i = 2π * (a_j x a_k) / V)
        self.b1 = 2 * np.pi * np.cross(a2, a3) / self.volume
        self.b2 = 2 * np.pi * np.cross(a3, a1) / self.volume
        self.b3 = 2 * np.pi * np.cross(a1, a2) / self.volume

        # 4. Determine Lattice Symmetry
        self.lattice_type = self._classify_lattice()

    def _classify_lattice(self) -> str:
        """Classify the Bravais lattice based on metric parameters."""
        if self.dim == 1:
            return "1d"

        norm1 = np.linalg.norm(self.a1)
        norm2 = np.linalg.norm(self.a2)
        norm3 = np.linalg.norm(self.a3)
        
        dot12 = np.dot(self.a1, self.a2)
        cos_gamma = dot12 / (norm1 * norm2)

        if self.dim == 2:
            # 2D Classification
            if np.isclose(abs(cos_gamma), 0.5, atol=0.05):  # 60 or 120 degrees
                return "hexagonal"
            elif np.isclose(cos_gamma, 0, atol=0.05):  # 90 degrees
                if np.isclose(norm1, norm2, atol=0.05):
                    return "square"
                else:
                    return "rectangular"
            else:
                return "oblique"
                
        if self.dim == 3:
            # Simple 3D Classification (Cubic vs others)
            cos_alpha = np.dot(self.a2, self.a3) / (norm2 * norm3)
            cos_beta = np.dot(self.a3, self.a1) / (norm3 * norm1)
            
            if np.isclose(cos_alpha, 0) and np.isclose(cos_beta, 0) and np.isclose(cos_gamma, 0):
                if np.isclose(norm1, norm2) and np.isclose(norm2, norm3):
                    return "cubic"
            return "custom_3d"

        return "unknown"

    def get_auto_k_path(self, n_points: int = 200):
        """
        Generates the k-values, k-points (Cartesian), k-ticks, and k-labels
        for the identified Brillouin Zone.
        """
        ltype = self.lattice_type
        
        # Fallback to oblique or custom if not standard
        if ltype not in BZ_PATHS:
            if self.dim == 2:
                # Use rectangular logic as a fallback for oblique
                ltype = "rectangular"
            else:
                # Basic 3D fallback
                from core.utils import get_k_path_square
                return get_k_path_square(n_points)

        config = BZ_PATHS[ltype]
        path_labels = config["path"]
        frac_points = config["points"]

        # Convert fractional points to Cartesian points
        cartesian_points = {}
        for lbl, frac in frac_points.items():
            cart_vec = frac[0] * self.b1 + frac[1] * self.b2 + frac[2] * self.b3
            cartesian_points[lbl] = cart_vec

        # Build segments
        segments = []
        for i in range(len(path_labels) - 1):
            start_lbl = path_labels[i]
            end_lbl = path_labels[i+1]
            segments.append((
                cartesian_points[start_lbl],
                cartesian_points[end_lbl],
                start_lbl,
                end_lbl
            ))

        from core.utils import _build_k_path_from_segments
        return _build_k_path_from_segments(segments, n_points)

    def summary(self) -> str:
        """Returns a human-readable summary of the BZ analysis."""
        return (f"**Lattice Analysis**: Detected as `{self.lattice_type.capitalize()}` "
                f"({self.dim}D).\n"
                f"Reciprocal Vectors:\n"
                f"  b₁ = [{self.b1[0]:.2f}, {self.b1[1]:.2f}, {self.b1[2]:.2f}]\n"
                f"  b₂ = [{self.b2[0]:.2f}, {self.b2[1]:.2f}, {self.b2[2]:.2f}]")
