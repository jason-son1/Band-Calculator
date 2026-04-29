"""
Band structure calculator: Hamiltonian construction and eigenvalue computation.
"""

import numpy as np


# Pauli matrices (2x2)
SIGMA_0 = np.array([[1, 0], [0, 1]], dtype=complex)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def compute_eigenvalues_1d(d_funcs, k_points, param_values, free_params):
    """
    Compute eigenvalues along a 1D k-path.

    Args:
        d_funcs: List of 4 callables [d0, dx, dy, dz], each with signature
                 f(k_x, k_y, k_z, *params)
        k_points: (N, D) array of k-coordinates along the path (D=1 or 2)
        param_values: dict mapping parameter names (str) to float values
        free_params: list of sympy Symbols (ordered) for the free parameters

    Returns:
        eigenvalues: (N, 2) array of eigenvalues at each k-point
        band_gap: minimum gap between the two bands
    """
    n_k = len(k_points)

    # Prepare k components
    if k_points.ndim == 1:
        k_points = k_points[:, None]

    kx = k_points[:, 0] if k_points.shape[1] >= 1 else np.zeros(n_k)
    ky = k_points[:, 1] if k_points.shape[1] >= 2 else np.zeros(n_k)
    kz = k_points[:, 2] if k_points.shape[1] >= 3 else np.zeros(n_k)

    # Build parameter argument list (in order of free_params)
    param_args = [float(param_values.get(str(p), 0.0)) for p in free_params]

    # Evaluate d-components (vectorized over k-points)
    d_vals = []
    for func in d_funcs:
        try:
            result = func(kx, ky, kz, *param_args)
            # Handle scalar return (constant function)
            if np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0):
                result = np.full(n_k, float(result))
            d_vals.append(np.asarray(result, dtype=float))
        except Exception:
            d_vals.append(np.zeros(n_k))

    d0, dx, dy, dz = d_vals

    # For 2x2 Pauli Hamiltonian, eigenvalues have analytical form:
    # E± = d0 ± sqrt(dx² + dy² + dz²)
    d_norm = np.sqrt(dx**2 + dy**2 + dz**2)
    e_plus = d0 + d_norm
    e_minus = d0 - d_norm

    eigenvalues = np.column_stack([e_minus, e_plus])

    # Compute band gap (minimum separation between bands)
    band_gap = np.min(2.0 * d_norm)

    return eigenvalues, float(band_gap)


def compute_eigenvalues_2d(d_funcs, kx_mesh, ky_mesh, param_values, free_params):
    """
    Compute eigenvalues on a 2D k-grid.

    Args:
        d_funcs: List of 4 callables [d0, dx, dy, dz]
        kx_mesh: 2D meshgrid of kx values (ny, nx)
        ky_mesh: 2D meshgrid of ky values (ny, nx)
        param_values: dict mapping parameter names to float values
        free_params: list of sympy Symbols (ordered)

    Returns:
        e_plus: 2D array of upper band eigenvalues
        e_minus: 2D array of lower band eigenvalues
        band_gap: minimum gap between the two bands
    """
    kz_mesh = np.zeros_like(kx_mesh)

    # Build parameter argument list
    param_args = [float(param_values.get(str(p), 0.0)) for p in free_params]

    # Evaluate d-components
    d_vals = []
    for func in d_funcs:
        try:
            result = func(kx_mesh, ky_mesh, kz_mesh, *param_args)
            if np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0):
                result = np.full_like(kx_mesh, float(result))
            d_vals.append(np.asarray(result, dtype=float))
        except Exception:
            d_vals.append(np.zeros_like(kx_mesh))

    d0, dx, dy, dz = d_vals

    # Analytical eigenvalues for 2x2 Pauli Hamiltonian
    d_norm = np.sqrt(dx**2 + dy**2 + dz**2)
    e_plus = d0 + d_norm
    e_minus = d0 - d_norm

    band_gap = float(np.min(2.0 * d_norm))

    return e_plus, e_minus, band_gap


def compute_eigenvalues_1d_nxn(func_matrix, k_points, param_values, free_params):
    """
    Compute eigenvalues for an NxN Hamiltonian along a 1D k-path.

    Args:
        func_matrix: NxN list of callables
        k_points: (N_k, D) array of k-coordinates
        param_values: dict mapping parameter names to float values
        free_params: list of sympy Symbols

    Returns:
        eigenvalues: (N_k, N) array of eigenvalues at each k-point (sorted real values)
        band_gap: minimum gap between the middle two bands (if N is even)
    """
    N = len(func_matrix)
    n_k = len(k_points)

    if k_points.ndim == 1:
        k_points = k_points[:, None]

    kx = k_points[:, 0] if k_points.shape[1] >= 1 else np.zeros(n_k)
    ky = k_points[:, 1] if k_points.shape[1] >= 2 else np.zeros(n_k)
    kz = k_points[:, 2] if k_points.shape[1] >= 3 else np.zeros(n_k)

    param_args = [float(param_values.get(str(p), 0.0)) for p in free_params]

    # Evaluate matrix elements
    H = np.zeros((n_k, N, N), dtype=complex)
    for i in range(N):
        for j in range(i, N):
            try:
                result = func_matrix[i][j](kx, ky, kz, *param_args)
                if np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0):
                    result = np.full(n_k, float(result) if np.isreal(result) else complex(result))
                H[:, i, j] = np.asarray(result)
            except Exception:
                pass # remains 0
            
            # Hermitian conjugate for lower triangle
            if i != j:
                H[:, j, i] = np.conj(H[:, i, j])

    # Compute eigenvalues using eigh (guarantees real, sorted eigenvalues for Hermitian matrices)
    eigenvalues = np.linalg.eigvalsh(H)

    # Compute band gap (between N//2 - 1 and N//2 bands)
    if N % 2 == 0:
        band_gap = float(np.min(eigenvalues[:, N//2] - eigenvalues[:, N//2 - 1]))
    else:
        band_gap = 0.0 # No clear "middle" gap for odd N

    return eigenvalues, band_gap


def compute_eigenvalues_2d_nxn(func_matrix, kx_mesh, ky_mesh, param_values, free_params):
    """
    Compute eigenvalues for an NxN Hamiltonian on a 2D k-grid.

    Returns:
        eigenvalues_mesh: List of N 2D arrays, each corresponding to a band
        band_gap: minimum gap between middle bands (if N is even)
    """
    N = len(func_matrix)
    shape = kx_mesh.shape
    kz_mesh = np.zeros_like(kx_mesh)

    param_args = [float(param_values.get(str(p), 0.0)) for p in free_params]

    H = np.zeros(shape + (N, N), dtype=complex)
    for i in range(N):
        for j in range(i, N):
            try:
                result = func_matrix[i][j](kx_mesh, ky_mesh, kz_mesh, *param_args)
                if np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0):
                    result = np.full_like(kx_mesh, float(result) if np.isreal(result) else complex(result))
                H[..., i, j] = np.asarray(result)
            except Exception:
                pass
            
            if i != j:
                H[..., j, i] = np.conj(H[..., i, j])

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(H) # shape is (ny, nx, N)

    # Split into list of 2D arrays for each band
    eigenvalues_mesh = [eigenvalues[..., n] for n in range(N)]

    if N % 2 == 0:
        band_gap = float(np.min(eigenvalues_mesh[N//2] - eigenvalues_mesh[N//2 - 1]))
    else:
        band_gap = 0.0

    return eigenvalues_mesh, band_gap
