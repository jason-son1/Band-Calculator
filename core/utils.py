"""
k-path and k-grid utilities for Brillouin zone sampling.
"""

import numpy as np


def _build_k_path_from_segments(segments, n_points):
    """
    Generic k-path builder from a list of (start, end, label_start, label_end) tuples.

    Returns:
        k_values, k_points, k_ticks, k_labels
    """
    k_points_list = []
    k_values_list = []
    k_ticks = [0.0]
    k_labels = [segments[0][2]]
    cumulative_dist = 0.0

    n_per_segment = max(n_points // len(segments), 2)

    for start, end, label_start, label_end in segments:
        t = np.linspace(0, 1, n_per_segment, endpoint=False)
        points = start[None, :] + t[:, None] * (end - start)[None, :]
        segment_length = np.linalg.norm(end - start)
        distances = cumulative_dist + t * segment_length

        k_points_list.append(points)
        k_values_list.append(distances)

        cumulative_dist += segment_length
        k_ticks.append(cumulative_dist)
        k_labels.append(label_end)

    k_points = np.vstack(k_points_list)
    k_values = np.concatenate(k_values_list)

    return k_values, k_points, k_ticks, k_labels


def get_k_path_square(n_points: int = 200):
    """
    Generate k-path along Γ → X → M → Γ for a 2D square lattice (z=0).
    """
    Gamma = np.array([0.0, 0.0, 0.0])
    X = np.array([np.pi, 0.0, 0.0])
    M = np.array([np.pi, np.pi, 0.0])

    segments = [
        (Gamma, X, "Γ", "X"),
        (X, M, "X", "M"),
        (M, Gamma, "M", "Γ"),
    ]
    return _build_k_path_from_segments(segments, n_points)


def get_k_path_1d(n_points: int = 200):
    """
    Generate 1D k-path: -π → 0 → π.
    """
    k_x = np.linspace(-np.pi, np.pi, n_points)
    k_points = np.column_stack([k_x, np.zeros_like(k_x), np.zeros_like(k_x)])  # shape (N, 3)
    k_ticks = [-np.pi, 0.0, np.pi]
    k_labels = ["-π", "0", "π"]

    return k_x, k_points, k_ticks, k_labels


def get_k_path_hexagonal(n_points: int = 200):
    """
    Generate k-path along Γ → M → K → Γ for a 2D hexagonal lattice (z=0).
    """
    Gamma = np.array([0.0, 0.0, 0.0])
    M_pt = np.array([np.pi, np.pi / np.sqrt(3), 0.0])
    K_pt = np.array([4 * np.pi / 3, 0.0, 0.0])

    segments = [
        (Gamma, M_pt, "Γ", "M"),
        (M_pt, K_pt, "M", "K"),
        (K_pt, Gamma, "K", "Γ"),
    ]
    return _build_k_path_from_segments(segments, n_points)


# ─── High-symmetry point library ─────────────────────────────────────
HIGH_SYMMETRY_POINTS = {
    "Γ": (0.0, 0.0, 0.0),
    "X": (np.pi, 0.0, 0.0),
    "Y": (0.0, np.pi, 0.0),
    "Z": (0.0, 0.0, np.pi),
    "M": (np.pi, np.pi, 0.0),
    "K": (4 * np.pi / 3, 0.0, 0.0),
    "K'": (2 * np.pi / 3, 2 * np.pi / np.sqrt(3), 0.0),
    "M_hex": (np.pi, np.pi / np.sqrt(3), 0.0),
    "R": (np.pi, np.pi, np.pi),
    "S": (np.pi / 2, np.pi / 2, 0.0),
}


def get_k_path_custom(point_sequence: list, n_points: int = 200):
    """
    Generate a k-path from a user-defined sequence of high-symmetry points.

    Args:
        point_sequence: List of dicts, each with:
            - "label": str (point label)
            - "kx": float (kx coordinate)
            - "ky": float (ky coordinate)
        n_points: Total number of k-points

    Returns:
        k_values, k_points, k_ticks, k_labels
    """
    if len(point_sequence) < 2:
        return get_k_path_1d(n_points)

    segments = []
    for i in range(len(point_sequence) - 1):
        p1 = point_sequence[i]
        p2 = point_sequence[i + 1]
        start = np.array([p1["kx"], p1["ky"], p1.get("kz", 0.0)])
        end = np.array([p2["kx"], p2["ky"], p2.get("kz", 0.0)])
        segments.append((start, end, p1["label"], p2["label"]))

    return _build_k_path_from_segments(segments, n_points)


def get_k_grid_2d(n_points: int = 80, kx_range=(-np.pi, np.pi), ky_range=(-np.pi, np.pi)):
    """
    Generate a 2D k-grid for surface plots.

    Returns:
        kx_mesh: 2D meshgrid of kx values (n_points x n_points)
        ky_mesh: 2D meshgrid of ky values (n_points x n_points)
    """
    kx = np.linspace(kx_range[0], kx_range[1], n_points)
    ky = np.linspace(ky_range[0], ky_range[1], n_points)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    return kx_mesh, ky_mesh

def get_auto_k_path_type(lattice) -> str:
    """
    Determine the best default k-path based on lattice dimension and vectors.
    Returns: "1d", "square", "hexagonal", or "custom".
    """
    if lattice.dimension == 1:
        return "1d"
    elif lattice.dimension == 2:
        a1 = np.array(lattice.a1[:2])
        a2 = np.array(lattice.a2[:2])
        norm1 = np.linalg.norm(a1)
        norm2 = np.linalg.norm(a2)
        
        if norm1 == 0 or norm2 == 0:
            return "square"
            
        # Check angle
        dot_product = np.dot(a1, a2)
        cos_theta = dot_product / (norm1 * norm2)
        
        # Hexagonal if angle is 60° (0.5) or 120° (-0.5)
        if np.isclose(abs(cos_theta), 0.5, atol=0.1):
            return "hexagonal"
        # Square if angle is 90° (0) and norms are close
        elif np.isclose(cos_theta, 0, atol=0.1) and np.isclose(norm1, norm2, atol=0.1):
            return "square"
        
        return "custom"
    else:
        return "custom"
