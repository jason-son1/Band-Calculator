"""
Pauli 행렬 및 스핀 관련 유틸리티 모듈.

Spin-Orbit Coupling (SOC) 지원을 위해 SymPy Matrix 형태의
Pauli 행렬을 정의하고, 사용자 입력 수식에 포함된 Pauli 심볼을
실제 행렬 원소로 전개(expand)하는 기능을 제공합니다.

사용 예시:
    >>> expand_pauli_expression("t * sigma_x + lambda_R * sigma_y", 2)
    Matrix([[0, t - I*lambda_R], [t + I*lambda_R, 0]])
"""

from __future__ import annotations

import sympy as sp
from sympy import Matrix, eye, I, sympify

# ══════════════════════════════════════════════════════════════════════
# SymPy Matrix 형태의 Pauli 행렬
# ══════════════════════════════════════════════════════════════════════

# σ₀ = 2×2 단위행렬
SIGMA_0: sp.Matrix = eye(2)

# σₓ — 스핀 플립 (off-diagonal real)
SIGMA_X: sp.Matrix = Matrix([[0, 1], [1, 0]])

# σᵧ — 스핀 플립 (off-diagonal imaginary)
SIGMA_Y: sp.Matrix = Matrix([[0, -I], [I, 0]])

# σ_z — 스핀 보존 (diagonal)
SIGMA_Z: sp.Matrix = Matrix([[1, 0], [0, -1]])

# ── 파서에 주입할 Pauli 심볼 매핑 ─────────────────────────────────────
# 사용자 수식에서 sigma_x, sx 등을 MatrixSymbol 대신 실제 행렬로 치환
PAULI_MAP: dict[str, sp.Matrix] = {
    'sigma_0': SIGMA_0, 's0': SIGMA_0,
    'sigma_x': SIGMA_X, 'sx': SIGMA_X,
    'sigma_y': SIGMA_Y, 'sy': SIGMA_Y,
    'sigma_z': SIGMA_Z, 'sz': SIGMA_Z,
}

# Pauli 행렬이 수식에 포함되어 있는지 판별하기 위한 키워드 세트
PAULI_KEYWORDS: set[str] = set(PAULI_MAP.keys())


def _contains_pauli(expr_str: str) -> bool:
    """수식 문자열에 Pauli 행렬 키워드가 포함되어 있는지 검사."""
    for kw in PAULI_KEYWORDS:
        if kw in expr_str:
            return True
    return False


def expand_pauli_expression(
    expr_str: str,
    spin_dim: int = 2,
    extra_locals: dict | None = None,
) -> sp.Matrix:
    """
    Pauli 행렬을 포함한 수식 문자열을 파싱하여 spin_dim × spin_dim SymPy Matrix로 전개.

    동작 방식:
    1. expr_str에 Pauli 키워드가 없으면 → 스칼라로 파싱 후 spin_dim × spin_dim 단위행렬에 곱하여 반환.
    2. Pauli 키워드가 있으면 → PAULI_MAP을 locals에 주입하여 sympify 호출.
       sympify 결과가 Matrix이면 그대로 반환, 스칼라이면 단위행렬에 곱.

    Args:
        expr_str: 수식 문자열.
            예: "t", "t * sigma_x", "t * sigma_z + lambda_R * sigma_y"
        spin_dim: 스핀 공간 차원 (기본 2: ↑, ↓)
        extra_locals: 추가로 주입할 심볼 딕셔너리 (parser.SAFE_LOCALS 등)

    Returns:
        spin_dim × spin_dim SymPy Matrix

    Raises:
        ValueError: 수식 파싱 실패 또는 행렬 차원 불일치
    """
    expr_str = expr_str.strip()
    if not expr_str:
        return sp.zeros(spin_dim, spin_dim)

    # sympify에 전달할 로컬 심볼 테이블 구성
    local_dict: dict = {
        'I': sp.I, 'pi': sp.pi, 'exp': sp.exp,
        'sqrt': sp.sqrt, 'cos': sp.cos, 'sin': sp.sin,
        'conjugate': sp.conjugate,
        'abs': sp.Abs, 'Abs': sp.Abs,
    }

    # 파동벡터 심볼 추가 (tbm_model 과의 일관성)
    from core.parser import K_SYMBOLS, K_ALIASES
    local_dict.update(K_SYMBOLS)
    local_dict.update(K_ALIASES)

    if extra_locals:
        local_dict.update(extra_locals)

    # Pauli 행렬을 로컬에 주입
    local_dict.update(PAULI_MAP)

    try:
        result = sympify(expr_str, locals=local_dict)
    except Exception as e:
        raise ValueError(
            f"Pauli 수식 파싱 오류: '{expr_str}'\n오류 내용: {e}"
        )

    # 결과가 Matrix인지 판별
    if isinstance(result, sp.MatrixBase):
        # 차원 검증
        if result.shape != (spin_dim, spin_dim):
            raise ValueError(
                f"행렬 차원 불일치: 기대 ({spin_dim}×{spin_dim}), "
                f"결과 ({result.shape[0]}×{result.shape[1]})\n"
                f"수식: '{expr_str}'"
            )
        return result
    else:
        # 스칼라 → 단위행렬에 곱
        return result * eye(spin_dim)


def tensor_product_insert(
    block: sp.Matrix,
    orbital_i: int,
    orbital_j: int,
    n_orbitals: int,
) -> sp.Matrix:
    """
    orbital ⊗ spin 텐서곱 공간에서 특정 orbital 쌍 (i, j)에
    spin block을 삽입한 전체 행렬을 반환.

    전체 차원 = n_orbitals * spin_dim
    block은 spin_dim × spin_dim 행렬.

    Args:
        block: 삽입할 spin block (spin_dim × spin_dim)
        orbital_i: source orbital 인덱스
        orbital_j: target orbital 인덱스
        n_orbitals: 전체 orbital 수

    Returns:
        (n_orbitals * spin_dim) × (n_orbitals * spin_dim) SymPy Matrix
    """
    spin_dim = block.shape[0]
    N_total = n_orbitals * spin_dim
    result = sp.zeros(N_total, N_total)

    for si in range(spin_dim):
        for sj in range(spin_dim):
            row = orbital_i * spin_dim + si
            col = orbital_j * spin_dim + sj
            result[row, col] = block[si, sj]

    return result
