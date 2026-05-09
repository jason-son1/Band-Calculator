"""
Synchronized numeric input widgets supporting symbolic expressions.

`numeric_expr_input` renders a trio (expression text + slider + number_input)
where the expression text is the source of truth. Users can type "sqrt(3)",
"pi/4", "2*sin(0.3)", "1+2*I" etc.

복소수 값이 평가되면:
  - 슬라이더/단일 숫자 입력 대신 Re(실수부) + Im(허수부) 숫자 입력으로 전환
  - 수식 텍스트 입력은 계속 유지 (고급 표현식 직접 입력 가능)
  - Re/Im 입력 변경 시 수식을 "{re}+{im}*I" 형태로 자동 재조합
"""
from __future__ import annotations

import streamlit as st

from core.parser import eval_constant_expression


def _fmt_float(v: float) -> str:
    """Compact float → string (drops trailing zeros)."""
    return f"{v:.6g}"


def _complex_to_expr(re: float, im: float) -> str:
    """Re/Im 값 → 수식 문자열. 예: 1.0, 0+2*I, 1.5-0.3*I"""
    if im == 0.0:
        return _fmt_float(re)
    sign = "+" if im >= 0 else ""
    return f"{_fmt_float(re)}{sign}{_fmt_float(im)}*I"


def numeric_expr_input(
    label: str,
    key: str,
    default_expr: str = "0",
    min_v: float = -5.0,
    max_v: float = 5.0,
    step: float = 0.05,
    fmt: str = "%.4f",
    show_slider: bool = False,
    show_label: bool = True,
    help: str | None = None,
) -> tuple[str, "float | complex"]:
    """
    Synchronized expression / slider / number input.

    실수이면: 수식 텍스트 + 슬라이더(선택) + 숫자 입력
    복소수이면: 수식 텍스트 + Re 숫자 입력 + Im 숫자 입력

    Returns:
        (expression_string, evaluated_value)  — value는 float 또는 complex.
    """
    expr_key      = f"{key}__expr"
    slider_key    = f"{key}__slider"
    num_key       = f"{key}__num"
    last_good_key = f"{key}__last_float"
    re_key        = f"{key}__re"
    im_key        = f"{key}__im"

    # ── 초기화 ──────────────────────────────────────────────────────
    if expr_key not in st.session_state:
        st.session_state[expr_key] = default_expr
        try:
            v0 = eval_constant_expression(default_expr)
        except ValueError:
            v0 = 0.0
        st.session_state[last_good_key] = v0
        if isinstance(v0, complex):
            st.session_state[re_key]     = v0.real
            st.session_state[im_key]     = v0.imag
            st.session_state[slider_key] = 0.0
            st.session_state[num_key]    = 0.0
        else:
            fv = float(v0)
            st.session_state[re_key]     = fv
            st.session_state[im_key]     = 0.0
            st.session_state[slider_key] = max(min_v, min(max_v, fv))
            st.session_state[num_key]    = fv

    # ── 콜백 ────────────────────────────────────────────────────────
    def _on_expr_change():
        s = st.session_state[expr_key]
        try:
            v = eval_constant_expression(s)
            st.session_state[last_good_key] = v
            if isinstance(v, complex):
                st.session_state[re_key] = v.real
                st.session_state[im_key] = v.imag
            else:
                fv = float(v)
                st.session_state[re_key]     = fv
                st.session_state[im_key]     = 0.0
                st.session_state[slider_key] = max(min_v, min(max_v, fv))
                st.session_state[num_key]    = fv
        except ValueError:
            pass

    def _on_slider_change():
        v = float(st.session_state[slider_key])
        st.session_state[last_good_key] = v
        st.session_state[expr_key]      = _fmt_float(v)
        st.session_state[num_key]       = v
        st.session_state[re_key]        = v
        st.session_state[im_key]        = 0.0

    def _on_num_change():
        v = float(st.session_state[num_key])
        st.session_state[last_good_key] = v
        st.session_state[expr_key]      = _fmt_float(v)
        st.session_state[slider_key]    = max(min_v, min(max_v, v))
        st.session_state[re_key]        = v
        st.session_state[im_key]        = 0.0

    def _on_re_change():
        re = float(st.session_state.get(re_key, 0.0))
        im = float(st.session_state.get(im_key, 0.0))
        s  = _complex_to_expr(re, im)
        st.session_state[expr_key]      = s
        st.session_state[last_good_key] = complex(re, im) if im != 0.0 else re

    def _on_im_change():
        re = float(st.session_state.get(re_key, 0.0))
        im = float(st.session_state.get(im_key, 0.0))
        s  = _complex_to_expr(re, im)
        st.session_state[expr_key]      = s
        st.session_state[last_good_key] = complex(re, im) if im != 0.0 else re

    # ── 레이블 ──────────────────────────────────────────────────────
    if show_label:
        st.markdown(
            f"**{label}**" + (f"  &nbsp;_{help}_" if help else ""),
            unsafe_allow_html=True,
        )

    last_good      = st.session_state.get(last_good_key, 0.0)
    is_complex_val = isinstance(last_good, complex)

    # ── 렌더링 ──────────────────────────────────────────────────────
    if is_complex_val:
        # 복소수 모드: 수식 입력 + Re/Im 분리 숫자 입력
        st.text_input(
            "수식" if show_label else label,
            key=expr_key,
            on_change=_on_expr_change,
            label_visibility="collapsed" if show_label else "visible",
            placeholder="예: 1+2*I, I*0.5, exp(I*pi/4)",
        )
        col_re, col_im = st.columns(2)
        with col_re:
            st.number_input(
                "Re (실수부)",
                key=re_key,
                step=float(step),
                format=fmt,
                on_change=_on_re_change,
            )
        with col_im:
            st.number_input(
                "Im (허수부)",
                key=im_key,
                step=float(step),
                format=fmt,
                on_change=_on_im_change,
            )

        # 현재 수식 재평가 및 동기화
        expr_str = st.session_state[expr_key]
        try:
            value = eval_constant_expression(expr_str)
            st.session_state[last_good_key] = value
            if isinstance(value, complex):
                st.session_state[re_key] = value.real
                st.session_state[im_key] = value.imag
        except ValueError as e:
            value = st.session_state.get(last_good_key, 0.0)
            st.caption(f":red[수식 오류: {e}]")

        re_v = st.session_state.get(re_key, 0.0)
        im_v = st.session_state.get(im_key, 0.0)
        st.caption(
            f":blue[복소수 모드] — 값: **{re_v:g}{'+' if im_v >= 0 else ''}{im_v:g}i**"
            f"  |  실수 복귀: 수식에 실수값 입력 (예: `1.0`)"
        )
        return expr_str, value

    # 실수 모드: 기존 레이아웃 (텍스트 + 슬라이더 + 숫자)
    if show_slider:
        col_expr, col_slider, col_num = st.columns([3, 4, 2])
    else:
        col_expr, col_num = st.columns([3, 2])
        col_slider = None

    with col_expr:
        st.text_input(
            "수식" if show_label else label,
            key=expr_key,
            on_change=_on_expr_change,
            label_visibility="collapsed" if show_label else "visible",
            placeholder="예: sqrt(3), pi/4, 1+2*I",
        )

    if col_slider is not None:
        with col_slider:
            st.slider(
                "slider",
                min_value=float(min_v),
                max_value=float(max_v),
                step=float(step),
                key=slider_key,
                on_change=_on_slider_change,
                label_visibility="collapsed",
            )

    with col_num:
        st.number_input(
            "num",
            min_value=None,
            max_value=None,
            step=float(step),
            key=num_key,
            format=fmt,
            on_change=_on_num_change,
            label_visibility="collapsed",
        )

    # 현재 수식 재평가 + 인라인 오류 표시
    expr_str = st.session_state[expr_key]
    try:
        value = eval_constant_expression(expr_str)
        st.session_state[last_good_key] = value
        # 이번 렌더에서 복소수로 전환된 경우 다음 사이클에서 자동으로 복소수 모드로 전환됨
    except ValueError as e:
        value = float(st.session_state.get(last_good_key, 0.0))
        st.caption(f":red[수식 오류: {e}]  (이전 값 {value:g} 사용)")

    return expr_str, value


def reset_numeric_expr_input(key: str, new_expr: str) -> None:
    """
    Force-reset a numeric_expr_input back to a new expression.
    프리셋·JSON 로드 시 사용. 위젯 렌더링 이전에 호출해야 합니다.
    """
    expr_key      = f"{key}__expr"
    slider_key    = f"{key}__slider"
    num_key       = f"{key}__num"
    last_good_key = f"{key}__last_float"
    re_key        = f"{key}__re"
    im_key        = f"{key}__im"

    st.session_state[expr_key] = new_expr
    try:
        v = eval_constant_expression(new_expr)
    except ValueError:
        v = 0.0
    st.session_state[last_good_key] = v
    if isinstance(v, complex):
        st.session_state[re_key]     = v.real
        st.session_state[im_key]     = v.imag
        st.session_state[slider_key] = 0.0
        st.session_state[num_key]    = 0.0
    else:
        fv = float(v)
        st.session_state[re_key]     = fv
        st.session_state[im_key]     = 0.0
        st.session_state[slider_key] = fv
        st.session_state[num_key]    = fv
