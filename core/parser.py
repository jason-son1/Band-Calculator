"""
SymPy-based expression parser for Hamiltonian input.
Handles string → symbolic expression → NumPy-callable conversion.
"""

import sympy as sp
from sympy import (
    Symbol, symbols, sympify, lambdify,
    sin, cos, tan, sqrt, exp, log, Abs, pi,
    conjugate, re, im, I,
    asin, acos, atan, atan2,
    sinh, cosh, tanh,
    sign, floor, ceiling,
    Piecewise,
)


# Reserved wave-vector symbols (not treated as free parameters)
K_SYMBOLS = {
    'k_x': Symbol('k_x', real=True),
    'k_y': Symbol('k_y', real=True),
    'k_z': Symbol('k_z', real=True),
}

# Convenience aliases for user input (e.g., "kx" instead of "k_x")
K_ALIASES = {
    'kx': K_SYMBOLS['k_x'],
    'ky': K_SYMBOLS['k_y'],
    'kz': K_SYMBOLS['k_z'],
}

# Safe math functions available to user — expanded set
SAFE_LOCALS = {
    # Trigonometric
    'sin': sin, 'cos': cos, 'tan': tan,
    'asin': asin, 'acos': acos, 'atan': atan, 'atan2': atan2,
    'arcsin': asin, 'arccos': acos, 'arctan': atan,
    # Hyperbolic
    'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
    # Exponential / Logarithmic
    'exp': exp, 'log': log, 'ln': log,
    # Power / Root
    'sqrt': sqrt, 'Abs': Abs, 'abs': Abs,
    # Constants
    'pi': pi, 'PI': pi, 'Pi': pi,
    'I': I, 'i': I,
    'e': sp.E, 'E_const': sp.E,
    # Complex
    'conjugate': conjugate, 'conj': conjugate,
    're': re, 'im': im,
    # Step / Floor / Ceiling
    'sign': sign, 'sgn': sign,
    'floor': floor, 'ceil': ceiling,
    'Heaviside': sp.Heaviside,
    # Wave vectors
    **K_SYMBOLS,
    **K_ALIASES,
}

# Organized reference for UI display
FUNCTION_REFERENCE = {
    "삼각함수 (Trig)": [
        ("sin(x)", "사인"),
        ("cos(x)", "코사인"),
        ("tan(x)", "탄젠트"),
        ("asin(x)", "역사인 (arcsin)"),
        ("acos(x)", "역코사인 (arccos)"),
        ("atan(x)", "역탄젠트 (arctan)"),
    ],
    "쌍곡선함수 (Hyperbolic)": [
        ("sinh(x)", "쌍곡사인"),
        ("cosh(x)", "쌍곡코사인"),
        ("tanh(x)", "쌍곡탄젠트"),
    ],
    "지수/로그 (Exp/Log)": [
        ("exp(x)", "지수함수 eˣ"),
        ("log(x)", "자연로그 ln(x)"),
        ("sqrt(x)", "제곱근 √x"),
        ("abs(x)", "절대값 |x|"),
    ],
    "상수 (Constants)": [
        ("pi", "원주율 π ≈ 3.14159"),
        ("I", "허수 단위 i"),
        ("e", "자연상수 e ≈ 2.71828"),
    ],
    "기타 (Misc)": [
        ("sign(x)", "부호 함수 sgn(x)"),
        ("floor(x)", "내림 함수"),
        ("ceil(x)", "올림 함수"),
        ("Heaviside(x)", "헤비사이드 계단 함수"),
        ("conj(x)", "켤레복소수"),
    ],
    "파동 벡터 (Wave Vector)": [
        ("k_x (또는 kx)", "x-방향 파동벡터"),
        ("k_y (또는 ky)", "y-방향 파동벡터"),
        ("k_z (또는 kz)", "z-방향 파동벡터"),
    ],
}


def parse_expression(expr_str: str) -> sp.Expr:
    """
    Parse a string expression into a SymPy expression.

    Args:
        expr_str: Mathematical expression string (e.g., "sin(k_x) + m * cos(k_y)")

    Returns:
        SymPy expression object

    Raises:
        ValueError: If the expression cannot be parsed
    """
    expr_str = expr_str.strip()
    if not expr_str:
        return sp.Integer(0)

    try:
        expr = sympify(expr_str, locals=SAFE_LOCALS)
        return expr
    except Exception as e:
        raise ValueError(
            f"수식 파싱 오류: '{expr_str}'\n"
            f"오류 내용: {str(e)}\n"
            f"사용 가능한 함수: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, "
            f"exp, log, sqrt, abs, sign, floor, ceil\n"
            f"사용 가능한 변수: k_x (또는 kx), k_y (또는 ky), k_z (또는 kz), 그 외 자유 파라미터"
        )


def extract_free_params(expressions: list) -> list:
    """
    Extract free parameters from a list of SymPy expressions.
    k_x, k_y, k_z are excluded (they are wave-vector variables, not free parameters).

    Args:
        expressions: List of SymPy expressions

    Returns:
        Sorted list of free parameter Symbol objects
    """
    all_symbols = set()
    reserved = set(K_SYMBOLS.values())

    for expr in expressions:
        if expr is not None:
            all_symbols.update(expr.free_symbols)

    free_params = all_symbols - reserved
    # Sort alphabetically for consistent UI ordering
    return sorted(free_params, key=lambda s: str(s))


def detect_dimensionality(expressions: list) -> int:
    """
    Detect whether the model is 1D, 2D, or 3D based on which k-variables appear.

    Returns:
        1, 2, or 3
    """
    all_symbols = set()
    for expr in expressions:
        if expr is not None:
            all_symbols.update(expr.free_symbols)

    dim = 0
    if K_SYMBOLS['k_x'] in all_symbols:
        dim = max(dim, 1)
    if K_SYMBOLS['k_y'] in all_symbols:
        dim = max(dim, 2)
    if K_SYMBOLS['k_z'] in all_symbols:
        dim = max(dim, 3)

    return max(dim, 1)  # At least 1D


def build_lambdified_funcs(expressions: list, free_params: list):
    """
    Convert SymPy expressions to fast NumPy-callable functions via lambdify.

    The returned function signature is:
        f(k_x, k_y, k_z, param1, param2, ...)

    Args:
        expressions: List of 4 SymPy expressions [d0, dx, dy, dz]
        free_params: List of free parameter Symbols

    Returns:
        List of 4 callable functions (one per d-component)
    """
    args = [K_SYMBOLS['k_x'], K_SYMBOLS['k_y'], K_SYMBOLS['k_z']] + list(free_params)

    funcs = []
    for expr in expressions:
        if expr is None or expr == 0:
            # Constant zero function
            funcs.append(lambda *a: 0.0)
        else:
            f = lambdify(args, expr, modules=['numpy'])
            funcs.append(f)

    return funcs


def build_lambdified_matrix_funcs(expr_matrix: list, free_params: list):
    """
    Convert a 2D list of SymPy expressions to a 2D list of fast NumPy-callable functions.

    Args:
        expr_matrix: NxN list of SymPy expressions
        free_params: List of free parameter Symbols

    Returns:
        NxN list of callable functions
    """
    args = [K_SYMBOLS['k_x'], K_SYMBOLS['k_y'], K_SYMBOLS['k_z']] + list(free_params)
    
    N = len(expr_matrix)
    func_matrix = [[None for _ in range(N)] for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            expr = expr_matrix[i][j]
            if expr is None or expr == 0:
                func_matrix[i][j] = lambda *a: 0.0
            else:
                f = lambdify(args, expr, modules=['numpy'])
                func_matrix[i][j] = f
                
    return func_matrix
