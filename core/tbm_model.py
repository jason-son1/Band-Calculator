"""
TBM (Tight Binding Model) Data Backend.

Defines the core data classes (Lattice, Site, State, Hopping, BasisConfig, TBMModel)
and the automatic Hamiltonian matrix builder.

확장 (Phase 1):
  - BasisConfig: Spin ⊗ Orbital 텐서곱 기저 구성
  - Hopping.amplitude_matrix: Pauli 행렬 포함 수식 지원
  - Block-matrix 기반 해밀토니안 빌더 (SOC 호환)
  - Kane-Mele / Rashba SOC 프리셋 모델

The output is a SymPy N×N matrix compatible with the existing
build_lambdified_matrix_funcs pipeline in core/parser.py.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Optional

import sympy as sp
from sympy import Symbol, symbols, exp, I, conjugate, pi, zeros

from core.spin_matrices import expand_pauli_expression

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
    a1_expr: list[str] = field(default_factory=lambda: ["1", "0", "0"])
    a2_expr: list[str] = field(default_factory=lambda: ["0", "1", "0"])
    a3_expr: list[str] = field(default_factory=lambda: ["0", "0", "1"])
    dimension: int = 2  # 1, 2, or 3

    def __post_init__(self):
        # Pad/truncate float vectors to length-3
        for attr in ("a1", "a2", "a3"):
            v = getattr(self, attr)
            while len(v) < 3:
                v.append(0.0)
            setattr(self, attr, list(v[:3]))
        # Back-fill _expr from float when missing or wrong length
        for attr in ("a1", "a2", "a3"):
            expr_attr = f"{attr}_expr"
            e = getattr(self, expr_attr)
            f = getattr(self, attr)
            if not isinstance(e, list) or len(e) != 3:
                e = [f"{x:g}" for x in f]
            else:
                e = [str(x) for x in e[:3]]
                while len(e) < 3:
                    e.append("0")
            setattr(self, expr_attr, e)


@dataclass
class BasisConfig:
    """
    기저 공간의 내부 자유도 구성.

    spin_enabled=True이면 각 State가 spin ↑/↓ 두 성분을 가지며,
    해밀토니안의 각 Hopping 블록이 spin_dim × spin_dim 행렬로 확장됩니다.
    전체 해밀토니안 크기 = (State 수) × internal_dim.
    """
    spin_enabled: bool = False       # True이면 스핀 자유도 활성화
    spin_dim: int = 2                # 스핀 공간 차원 (기본 2: ↑, ↓)

    @property
    def internal_dim(self) -> int:
        """Hopping 하나가 차지하는 내부 블록 크기."""
        return self.spin_dim if self.spin_enabled else 1


@dataclass
class Site:
    """
    A physical site (atom / node) inside the unit cell.

    position is in fractional coordinates relative to the lattice vectors,
    stored as a 3-component list [x, y, z].
    """
    name: str
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    position_expr: list[str] = field(default_factory=lambda: ["0", "0", "0"])
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self):
        while len(self.position) < 3:
            self.position.append(0.0)
        self.position = list(self.position[:3])
        if not isinstance(self.position_expr, list) or len(self.position_expr) != 3:
            self.position_expr = [f"{x:g}" for x in self.position]
        else:
            self.position_expr = [str(x) for x in self.position_expr[:3]]
            while len(self.position_expr) < 3:
                self.position_expr.append("0")

    def __repr__(self):
        return f"Site({self.name}, pos={self.position})"


@dataclass
class State:
    """
    A single quantum-mechanical basis state.

    State = Site + Orbital + Spin
    This is the fundamental unit (row/column) of the Hamiltonian matrix.
    spin_enabled일 때는 spin 자유도가 internal_dim으로 자동 확장됩니다.
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

    Phase 1 확장:
      - amplitude_matrix: Pauli 행렬을 포함한 행렬 수식 문자열.
        예: "t * sigma_z + lambda_R * sigma_y"
        비어있으면 amplitude(스칼라)를 단위행렬에 곱하여 사용.
      - color_id: Color-coding을 위한 식별 인덱스 (Phase 2에서 활용).
    """
    source_uid: str                         # UID of the source State
    target_uid: str                         # UID of the target State
    amplitude: str = "1.0"                  # 스칼라 수식 (하위 호환)
    amplitude_matrix: str = ""              # 행렬 수식 (Pauli 포함 가능)
    R: list[int] = field(default_factory=lambda: [0, 0, 0])
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    label: str = ""                         # Optional display label
    color_id: int = -1                      # Color-coding용 ID

    def __post_init__(self):
        while len(self.R) < 3:
            self.R.append(0)
        self.R = list(self.R[:3])

    def get_sympy_amplitude(self) -> sp.Expr:
        """Parse amplitude string into a SymPy expression (스칼라)."""
        try:
            return sp.sympify(self.amplitude, locals={
                'I': sp.I, 'pi': sp.pi, 'exp': sp.exp,
                'sqrt': sp.sqrt, 'cos': sp.cos, 'sin': sp.sin,
                'conjugate': sp.conjugate,
                'k_x': k_x, 'k_y': k_y, 'k_z': k_z,
                'kx': k_x, 'ky': k_y, 'kz': k_z,
            })
        except Exception:
            return sp.Integer(0)

    def get_sympy_block(self, internal_dim: int = 1) -> sp.Matrix:
        """
        Hopping의 amplitude를 internal_dim × internal_dim SymPy Matrix로 반환.

        - amplitude_matrix가 비어있지 않으면: Pauli 행렬 전개
        - 비어있으면: amplitude 스칼라를 단위행렬에 곱
        """
        if internal_dim <= 1:
            # 스칼라 모드 — 1×1 행렬로 래핑
            return sp.Matrix([[self.get_sympy_amplitude()]])

        if self.amplitude_matrix.strip():
            # 행렬 수식 모드 — Pauli 행렬 전개
            try:
                return expand_pauli_expression(self.amplitude_matrix, spin_dim=internal_dim)
            except ValueError:
                # 폴백: 스칼라 amplitude × 단위행렬
                return self.get_sympy_amplitude() * sp.eye(internal_dim)
        else:
            # 스칼라 amplitude → 단위행렬에 곱
            return self.get_sympy_amplitude() * sp.eye(internal_dim)

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

    Phase 1 확장: BasisConfig를 통해 Spin ⊗ Orbital 텐서곱 공간 지원.
    """

    def __init__(self, lattice: Optional[Lattice] = None,
                 basis_config: Optional[BasisConfig] = None):
        self.lattice: Lattice = lattice or Lattice()
        self.basis_config: BasisConfig = basis_config or BasisConfig()
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

    def get_hopping(self, uid: str) -> Optional[Hopping]:
        """UID로 Hopping 객체를 조회."""
        for h in self.hoppings:
            if h.uid == uid:
                return h
        return None

    # ── Index map ────────────────────────────────────────────────────

    def _build_index_map(self) -> dict[str, int]:
        """Map state UID → row/column index in the Hamiltonian matrix."""
        return {st.uid: idx for idx, st in enumerate(self.states)}

    # ── Dimension helpers ────────────────────────────────────────────

    def matrix_dimension(self) -> int:
        """State 수 (internal_dim 미적용)."""
        return len(self.states)

    def total_dimension(self) -> int:
        """전체 해밀토니안 행렬 크기 = (State 수) × (내부 자유도 차원)."""
        return len(self.states) * self.basis_config.internal_dim

    # ── Hamiltonian Matrix Builder ────────────────────────────────────

    def build_hamiltonian_matrix(self) -> sp.Matrix:
        """
        Build the symbolic N_total × N_total Hamiltonian matrix H(k).

        N_total = N_states × internal_dim (spin_enabled=True이면 ×2)

        각 Hopping은 internal_dim × internal_dim 블록으로 삽입됨:
          H[i*d+si, j*d+sj] += block[si,sj] * exp(i k·δ)

        Hermitian conjugate is added automatically.
        Returns a SymPy Matrix with k_x, k_y, k_z as free symbols.
        """
        N_states = len(self.states)
        d = self.basis_config.internal_dim
        N_total = N_states * d

        if N_total == 0:
            return sp.Matrix([[0]])

        idx = self._build_index_map()
        H = sp.zeros(N_total, N_total)

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
            R_vec = sp.Matrix([sp.Integer(r) for r in hop.R])
            delta = r_tgt - r_src + R_vec

            # Phase factor  exp(i k · delta)
            phase_arg = k_x * delta[0] + k_y * delta[1] + k_z * delta[2]
            phase = sp.exp(sp.I * phase_arg)

            # Block matrix (d × d) — 스칼라 또는 Pauli 전개 결과
            block = hop.get_sympy_block(d)

            # 블록을 전체 행렬에 삽입
            for bi in range(d):
                for bj in range(d):
                    if block[bi, bj] != 0:
                        H[i * d + bi, j * d + bj] += block[bi, bj] * phase

            # Hermitian conjugate 자동 추가 (on-site 대각 제외)
            if i != j or any(r != 0 for r in hop.R):
                block_hc = block.adjoint()
                phase_hc = sp.conjugate(phase)
                for bi in range(d):
                    for bj in range(d):
                        if block_hc[bi, bj] != 0:
                            H[j * d + bi, i * d + bj] += block_hc[bi, bj] * phase_hc

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

    def trace_element_to_hoppings(self, row: int, col: int) -> list[str]:
        """
        행렬 원소 H[row, col]에 기여하는 Hopping들의 UID 목록을 반환.
        Color-coding에서 사용됩니다 (Phase 2).
        """
        d = self.basis_config.internal_dim
        idx = self._build_index_map()
        contributing_uids: list[str] = []

        for hop in self.hoppings:
            src = self.get_state(hop.source_uid)
            tgt = self.get_state(hop.target_uid)
            if src is None or tgt is None:
                continue

            i = idx.get(hop.source_uid, -1)
            j = idx.get(hop.target_uid, -1)
            if i < 0 or j < 0:
                continue

            # Forward: block at (i*d..i*d+d, j*d..j*d+d)
            if i * d <= row < (i + 1) * d and j * d <= col < (j + 1) * d:
                contributing_uids.append(hop.uid)
                continue
            # Hermitian conjugate: block at (j*d..j*d+d, i*d..i*d+d)
            if (i != j or any(r != 0 for r in hop.R)):
                if j * d <= row < (j + 1) * d and i * d <= col < (i + 1) * d:
                    contributing_uids.append(hop.uid)

        return contributing_uids

    def summary(self) -> str:
        """Human-readable summary of the model."""
        lines = [
            f"Lattice: {self.lattice.dimension}D",
            f"Sites: {[s.name for s in self.sites]}",
            f"States (orbital basis): {self.matrix_dimension()}",
            f"Spin enabled: {self.basis_config.spin_enabled}",
            f"Total H dimension: {self.total_dimension()}",
            f"Hoppings: {len(self.hoppings)}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Preset Builders
# ══════════════════════════════════════════════════════════════════════

def build_graphene_model() -> TBMModel:
    """
    Graphene tight-binding model.
    Two sublattices A and B, nearest-neighbor hopping t on a hexagonal lattice.
    Expected result: two bands touching at Dirac points.
    """
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
    sB = model.add_site(Site("B", [0.0, 0.0, 0.0]))

    stA = model.add_state(State(sA, orbital="px", spin="none"))
    stB = model.add_state(State(sB, orbital="py", spin="none"))

    model.add_hopping(Hopping(stA.uid, stA.uid,
                               amplitude="m + cos(k_x) + cos(k_y)", R=[0, 0, 0]))
    model.add_hopping(Hopping(stB.uid, stB.uid,
                               amplitude="-(m + cos(k_x) + cos(k_y))", R=[0, 0, 0]))
    model.add_hopping(Hopping(stA.uid, stB.uid,
                               amplitude="sin(k_x) - I*sin(k_y)", R=[0, 0, 0]))

    return model


def build_kane_mele_model() -> TBMModel:
    """
    Kane-Mele Model — 2D Z₂ Topological Insulator.

    그래핀 격자 위의 nearest-neighbor hopping (t) +
    next-nearest-neighbor intrinsic SOC (lambda_SO * sigma_z).

    spin_enabled=True → 전체 4×4 해밀토니안 (A↑, A↓, B↑, B↓).
    """
    a = 1.0
    model = TBMModel(
        lattice=Lattice(
            a1=[a, 0.0, 0.0],
            a2=[a / 2, a * math.sqrt(3) / 2, 0.0],
            a3=[0.0, 0.0, 1.0],
            dimension=2,
        ),
        basis_config=BasisConfig(spin_enabled=True, spin_dim=2),
    )

    sA = model.add_site(Site("A", [0.0, 0.0, 0.0]))
    sB = model.add_site(Site("B", [0.5, 0.5 / math.sqrt(3), 0.0]))

    stA = model.add_state(State(sA, orbital="pz", spin="none"))
    stB = model.add_state(State(sB, orbital="pz", spin="none"))

    # NN hopping: t (스핀 보존 → 스칼라 amplitude, 단위행렬로 확장)
    model.add_hopping(Hopping(stA.uid, stB.uid, amplitude="t", R=[0, 0, 0],
                               label="NN t"))
    model.add_hopping(Hopping(stA.uid, stB.uid, amplitude="t", R=[1, 0, 0],
                               label="NN t"))
    model.add_hopping(Hopping(stA.uid, stB.uid, amplitude="t", R=[0, 1, 0],
                               label="NN t"))

    # NNN intrinsic SOC: i*lambda_SO*nu_ij*sigma_z
    # A→A (NNN): nu = +1 for certain R vectors
    model.add_hopping(Hopping(stA.uid, stA.uid,
                               amplitude_matrix="I*lambda_SO*sigma_z",
                               R=[1, 0, 0], label="SOC A→A"))
    model.add_hopping(Hopping(stA.uid, stA.uid,
                               amplitude_matrix="-I*lambda_SO*sigma_z",
                               R=[0, 1, 0], label="SOC A→A"))
    model.add_hopping(Hopping(stA.uid, stA.uid,
                               amplitude_matrix="I*lambda_SO*sigma_z",
                               R=[-1, 1, 0], label="SOC A→A"))

    # B→B (NNN): opposite sign
    model.add_hopping(Hopping(stB.uid, stB.uid,
                               amplitude_matrix="-I*lambda_SO*sigma_z",
                               R=[1, 0, 0], label="SOC B→B"))
    model.add_hopping(Hopping(stB.uid, stB.uid,
                               amplitude_matrix="I*lambda_SO*sigma_z",
                               R=[0, 1, 0], label="SOC B→B"))
    model.add_hopping(Hopping(stB.uid, stB.uid,
                               amplitude_matrix="-I*lambda_SO*sigma_z",
                               R=[-1, 1, 0], label="SOC B→B"))

    return model


def build_rashba_2d_model() -> TBMModel:
    """
    2D Square Lattice with Rashba SOC.

    H = -t Σ c†(R+δ)c(R) + iλ_R Σ c†(R+δ)(σ × d̂)_z c(R)
    즉: x-방향 hopping에 Rashba term ∝ sigma_y, y-방향에 ∝ -sigma_x.

    spin_enabled=True → 2 States × 2 spin = 4×4 해밀토니안.
    """
    model = TBMModel(
        lattice=Lattice(
            a1=[1.0, 0.0, 0.0],
            a2=[0.0, 1.0, 0.0],
            dimension=2,
        ),
        basis_config=BasisConfig(spin_enabled=True, spin_dim=2),
    )

    sA = model.add_site(Site("A", [0.0, 0.0, 0.0]))
    stA = model.add_state(State(sA, orbital="s", spin="none"))

    # On-site energy
    model.add_hopping(Hopping(stA.uid, stA.uid, amplitude="eps", R=[0, 0, 0],
                               label="On-site"))

    # x-방향 NN hopping: -t * sigma_0 + i*lambda_R * sigma_y
    model.add_hopping(Hopping(stA.uid, stA.uid,
                               amplitude_matrix="-t*s0 + I*lambda_R*sy",
                               R=[1, 0, 0], label="x-hop"))

    # y-방향 NN hopping: -t * sigma_0 - i*lambda_R * sigma_x
    model.add_hopping(Hopping(stA.uid, stA.uid,
                               amplitude_matrix="-t*s0 - I*lambda_R*sx",
                               R=[0, 1, 0], label="y-hop"))

    return model
