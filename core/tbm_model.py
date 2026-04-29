"""
TBM (Tight Binding Model) Data Backend.

Defines the core data classes (Lattice, Site, State, Hopping, TBMModel)
and the automatic Hamiltonian matrix builder.

The output is a SymPy N×N matrix compatible with the existing
build_lambdified_matrix_funcs pipeline in core/parser.py.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import sympy as sp
from sympy import Symbol, symbols, exp, I, conjugate, pi, zeros

# ── Wave-vector symbols (shared with core/parser.py) ─────────────────
k_x = Symbol('k_x', real=True)
k_y = Symbol('k_y', real=True)
k_z = Symbol('k_z', real=True)

# ── Supported orbital labels ──────────────────────────────────────────
ORBITAL_OPTIONS = [
    "s",
    "px", "py", "pz",
    "dxy", "dxz", "dyz", "dx2y2", "dz2",
    "custom",
]

# ── Supported spin labels ─────────────────────────────────────────────
SPIN_OPTIONS = ["↑", "↓", "none"]
SPIN_DISPLAY = {"↑": "Up", "↓": "Down", "none": "—"}


# ══════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Lattice:
    """
    Defines the periodic Bravais lattice via primitive vectors.

    All vectors are in units of Angstrom (or arbitrary length units).
    Components are stored as plain floats; SymPy is not needed here.
    """
    a1: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    a2: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    a3: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    dimension: int = 2  # 1, 2, or 3

    def __post_init__(self):
        # Pad vectors to length-3
        for attr in ("a1", "a2", "a3"):
            v = getattr(self, attr)
            while len(v) < 3:
                v.append(0.0)
            setattr(self, attr, list(v[:3]))


@dataclass
class Site:
    """
    A physical site (atom / node) inside the unit cell.

    position is in fractional coordinates relative to the lattice vectors,
    stored as a 3-component list [x, y, z].
    """
    name: str
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self):
        while len(self.position) < 3:
            self.position.append(0.0)
        self.position = list(self.position[:3])

    def __repr__(self):
        return f"Site({self.name}, pos={self.position})"


@dataclass
class State:
    """
    A single quantum-mechanical basis state.

    State = Site + Orbital + Spin
    This is the fundamental unit (row/column) of the Hamiltonian matrix.
    """
    site: Site
    orbital: str = "s"          # One of ORBITAL_OPTIONS
    spin: str = "none"          # One of SPIN_OPTIONS
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def label(self) -> str:
        """Short human-readable label used in the TBM diagram."""
        spin_sym = {"↑": "↑", "↓": "↓", "none": ""}[self.spin]
        return f"{self.site.name}|{self.orbital}{spin_sym}"

    def __repr__(self):
        return f"State({self.label()})"


@dataclass
class Hopping:
    """
    A hopping term between two States.

    H_ij(k) += amplitude * exp(i * k . delta)
    where delta = r_target - r_source + R

    For on-site energy: source == target, R = [0,0,0]
    For intra-cell hopping: source != target, R = [0,0,0]
    For inter-cell hopping: R != [0,0,0]

    amplitude can be a real/complex number OR a SymPy expression string
    (e.g., "t1", "t*exp(I*phi)") to allow symbolic parameters.
    """
    source_uid: str                         # UID of the source State
    target_uid: str                         # UID of the target State
    amplitude: str = "1.0"                  # String → parsed to SymPy expr
    R: list[int] = field(default_factory=lambda: [0, 0, 0])
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    label: str = ""                         # Optional display label

    def __post_init__(self):
        while len(self.R) < 3:
            self.R.append(0)
        self.R = list(self.R[:3])

    def get_sympy_amplitude(self) -> sp.Expr:
        """Parse amplitude string into a SymPy expression."""
        try:
            return sp.sympify(self.amplitude, locals={
                'I': sp.I, 'pi': sp.pi, 'exp': sp.exp,
                'sqrt': sp.sqrt, 'cos': sp.cos, 'sin': sp.sin,
                'conjugate': sp.conjugate,
            })
        except Exception:
            return sp.Integer(0)

    def is_onsite(self) -> bool:
        return self.source_uid == self.target_uid and all(r == 0 for r in self.R)

    def is_intracell(self) -> bool:
        return self.source_uid != self.target_uid and all(r == 0 for r in self.R)

    def is_intercell(self) -> bool:
        return any(r != 0 for r in self.R)


# ══════════════════════════════════════════════════════════════════════
# TBM Model Container & Matrix Builder
# ══════════════════════════════════════════════════════════════════════

class TBMModel:
    """
    Container for a complete Tight Binding Model.

    Holds lists of Sites, States, and Hoppings, and provides the
    automatic Hamiltonian matrix builder.
    """

    def __init__(self, lattice: Optional[Lattice] = None):
        self.lattice: Lattice = lattice or Lattice()
        self.sites: list[Site] = []
        self.states: list[State] = []
        self.hoppings: list[Hopping] = []

    # ── CRUD helpers ──────────────────────────────────────────────────

    def add_site(self, site: Site) -> Site:
        self.sites.append(site)
        return site

    def remove_site(self, uid: str):
        """Remove a site and all states/hoppings that reference it."""
        self.sites = [s for s in self.sites if s.uid != uid]
        state_uids_to_remove = {st.uid for st in self.states if st.site.uid == uid}
        self.states = [st for st in self.states if st.site.uid != uid]
        self.hoppings = [
            h for h in self.hoppings
            if h.source_uid not in state_uids_to_remove
            and h.target_uid not in state_uids_to_remove
        ]

    def add_state(self, state: State) -> State:
        self.states.append(state)
        return state

    def remove_state(self, uid: str):
        self.states = [st for st in self.states if st.uid != uid]
        self.hoppings = [
            h for h in self.hoppings
            if h.source_uid != uid and h.target_uid != uid
        ]

    def add_hopping(self, hopping: Hopping) -> Hopping:
        self.hoppings.append(hopping)
        return hopping

    def remove_hopping(self, uid: str):
        self.hoppings = [h for h in self.hoppings if h.uid != uid]

    def get_state(self, uid: str) -> Optional[State]:
        for st in self.states:
            if st.uid == uid:
                return st
        return None

    # ── Index map ────────────────────────────────────────────────────

    def _build_index_map(self) -> dict[str, int]:
        """Map state UID → row/column index in the Hamiltonian matrix."""
        return {st.uid: idx for idx, st in enumerate(self.states)}

    # ── Hamiltonian Matrix Builder ────────────────────────────────────

    def build_hamiltonian_matrix(self) -> sp.Matrix:
        """
        Build the symbolic N×N Hamiltonian matrix H(k_x, k_y, k_z).

        H_ij(k) = sum_R  t_ij(R) * exp(i * k . (r_j - r_i + R))

        Hermitian conjugate is added automatically for each Hopping:
          H_ji += conj(t_ij) * exp(-i * k . delta)

        Returns a SymPy Matrix with k_x, k_y, k_z as free symbols.
        The matrix is compatible with build_lambdified_matrix_funcs.
        """
        N = len(self.states)
        if N == 0:
            return sp.Matrix([[0]])

        idx = self._build_index_map()
        H = sp.zeros(N, N)

        for hop in self.hoppings:
            src = self.get_state(hop.source_uid)
            tgt = self.get_state(hop.target_uid)
            if src is None or tgt is None:
                continue

            i = idx[hop.source_uid]
            j = idx[hop.target_uid]

            # Displacement vector delta = r_target - r_source + R
            r_src = sp.Matrix(src.site.position)
            r_tgt = sp.Matrix(tgt.site.position)
            R = sp.Matrix([sp.Integer(r) for r in hop.R])
            delta = r_tgt - r_src + R

            # Phase factor  exp(i k . delta)
            phase_arg = k_x * delta[0] + k_y * delta[1] + k_z * delta[2]
            phase = sp.exp(sp.I * phase_arg)

            t = hop.get_sympy_amplitude()

            H[i, j] = H[i, j] + t * phase

            # Automatically add Hermitian conjugate (skip if on-site to avoid double-count)
            if i != j:
                H[j, i] = H[j, i] + sp.conjugate(t) * sp.conjugate(phase)

        return H

    def build_hamiltonian_expr_matrix(self) -> list[list]:
        """
        Return H(k) as a 2D Python list of SymPy expressions.
        This is the format expected by build_lambdified_matrix_funcs
        and the NxN compute_eigenvalues pipeline.
        """
        H = self.build_hamiltonian_matrix()
        N = H.shape[0]
        return [[H[i, j] for j in range(N)] for i in range(N)]

    def free_parameters(self) -> list[sp.Symbol]:
        """
        Extract free symbolic parameters (excluding k_x, k_y, k_z).
        These will be turned into interactive sliders in the UI.
        """
        H = self.build_hamiltonian_matrix()
        k_reserved = {k_x, k_y, k_z}
        return sorted(H.free_symbols - k_reserved, key=lambda s: s.name)

    def matrix_dimension(self) -> int:
        return len(self.states)

    def summary(self) -> str:
        """Human-readable summary of the model."""
        lines = [
            f"Lattice: {self.lattice.dimension}D",
            f"Sites: {[s.name for s in self.sites]}",
            f"States (basis dim): {self.matrix_dimension()}",
            f"Hoppings: {len(self.hoppings)}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Preset Builders (for testing and quick-start)
# ══════════════════════════════════════════════════════════════════════

def build_graphene_model() -> TBMModel:
    """
    Graphene tight-binding model.
    Two sublattices A and B, nearest-neighbor hopping t on a hexagonal lattice.
    Expected result: two bands touching at Dirac points.
    """
    import math
    a = 1.0
    model = TBMModel(Lattice(
        a1=[a, 0.0, 0.0],
        a2=[a / 2, a * math.sqrt(3) / 2, 0.0],
        a3=[0.0, 0.0, 1.0],
        dimension=2,
    ))

    sA = model.add_site(Site("A", [0.0, 0.0, 0.0]))
    sB = model.add_site(Site("B", [0.5, 0.5 / math.sqrt(3), 0.0]))

    stA = model.add_state(State(sA, orbital="pz", spin="none"))
    stB = model.add_state(State(sB, orbital="pz", spin="none"))

    # Three nearest-neighbor hoppings (only i→j; conjugates auto-added)
    model.add_hopping(Hopping(stA.uid, stB.uid, amplitude="t", R=[0, 0, 0]))
    model.add_hopping(Hopping(stA.uid, stB.uid, amplitude="t", R=[1, 0, 0]))
    model.add_hopping(Hopping(stA.uid, stB.uid, amplitude="t", R=[0, 1, 0]))

    return model


def build_ssh_model() -> TBMModel:
    """
    SSH (Su-Schrieffer-Heeger) chain model.
    Two sites A and B, alternating hoppings v (intra) and w (inter).
    Expected result: topological phase transition at |v| = |w|.
    """
    model = TBMModel(Lattice(
        a1=[1.0, 0.0, 0.0],
        a2=[0.0, 1.0, 0.0],
        a3=[0.0, 0.0, 1.0],
        dimension=1,
    ))

    sA = model.add_site(Site("A", [0.0, 0.0, 0.0]))
    sB = model.add_site(Site("B", [0.5, 0.0, 0.0]))

    stA = model.add_state(State(sA, orbital="s", spin="none"))
    stB = model.add_state(State(sB, orbital="s", spin="none"))

    # Intra-cell hopping (v)
    model.add_hopping(Hopping(stA.uid, stB.uid, amplitude="v", R=[0, 0, 0]))
    # Inter-cell hopping (w) — B in cell 0 to A in cell 1
    model.add_hopping(Hopping(stB.uid, stA.uid, amplitude="w", R=[1, 0, 0]))

    return model


def build_qwz_model() -> TBMModel:
    """
    QWZ (Qi-Wu-Zhang) model as a TBM for verification.
    2x2 matrix. Compare result with the Pauli-mode output.
    """
    model = TBMModel(Lattice(dimension=2))

    sA = model.add_site(Site("A", [0.0, 0.0, 0.0]))
    sB = model.add_site(Site("B", [0.0, 0.0, 0.0]))  # same position, different orbital

    stA = model.add_state(State(sA, orbital="px", spin="none"))
    stB = model.add_state(State(sB, orbital="py", spin="none"))

    # On-site: m + cos(kx) + cos(ky) on A, -(m + cos(kx) + cos(ky)) on B
    # These are implemented via Hoppings with amplitude as k-dependent strings
    # On-site: H[0,0]
    model.add_hopping(Hopping(stA.uid, stA.uid,
                               amplitude="m + cos(k_x) + cos(k_y)", R=[0, 0, 0]))
    # On-site: H[1,1]
    model.add_hopping(Hopping(stB.uid, stB.uid,
                               amplitude="-(m + cos(k_x) + cos(k_y))", R=[0, 0, 0]))
    # Off-diagonal: sin(kx) - I*sin(ky)
    model.add_hopping(Hopping(stA.uid, stB.uid,
                               amplitude="sin(k_x) - I*sin(k_y)", R=[0, 0, 0]))

    return model
