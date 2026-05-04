"""
Synchronized numeric input widgets supporting symbolic expressions.

`numeric_expr_input` renders a trio (expression text + slider + number_input)
where the expression text is the source of truth. Users can type "sqrt(3)",
"pi/4", "2*sin(0.3)", etc. — these are evaluated to float by the existing
parser. Dragging the slider or editing the number replaces the expression
with a numeric literal.
"""
from __future__ import annotations

import streamlit as st

from core.parser import eval_constant_expression


def _fmt_float(v: float) -> str:
    """Compact float → string (drops trailing zeros)."""
    s = f"{v:.6g}"
    return s


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
) -> tuple[str, float]:
    """
    Synchronized expression / slider / number input.

    Args:
        label: Display label.
        key: Base session_state key. Must be unique per widget instance.
        default_expr: Initial expression string when key not in session_state.
        min_v, max_v, step: Slider/number input range and step.
        fmt: Number input display format.
        show_slider: Whether to render the slider column.
        show_label: Whether to render the text label above the row.
        help: Optional tooltip.

    Returns:
        (expression_string, evaluated_float).
        On parse error, the float falls back to the last successfully evaluated
        value (or 0.0 on first failure), and the expression text is preserved
        verbatim so the user can fix the typo.
    """
    expr_key = f"{key}__expr"
    slider_key = f"{key}__slider"
    num_key = f"{key}__num"
    last_good_key = f"{key}__last_float"

    if expr_key not in st.session_state:
        st.session_state[expr_key] = default_expr
        try:
            v0 = eval_constant_expression(default_expr)
        except ValueError:
            v0 = 0.0
        st.session_state[last_good_key] = v0
        st.session_state[slider_key] = max(min_v, min(max_v, v0))
        st.session_state[num_key] = v0

    def _on_expr_change():
        s = st.session_state[expr_key]
        try:
            v = eval_constant_expression(s)
            st.session_state[last_good_key] = v
            st.session_state[slider_key] = max(min_v, min(max_v, v))
            st.session_state[num_key] = v
        except ValueError:
            # leave slider/num as-is; expression text stays so user can fix
            pass

    def _on_slider_change():
        v = float(st.session_state[slider_key])
        st.session_state[last_good_key] = v
        st.session_state[expr_key] = _fmt_float(v)
        st.session_state[num_key] = v

    def _on_num_change():
        v = float(st.session_state[num_key])
        st.session_state[last_good_key] = v
        st.session_state[expr_key] = _fmt_float(v)
        st.session_state[slider_key] = max(min_v, min(max_v, v))

    if show_label:
        st.markdown(f"**{label}**" + (f"  &nbsp;_{help}_" if help else ""),
                    unsafe_allow_html=True)

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
            placeholder="예: sqrt(3), pi/4, 1.5",
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

    # Validate current expression and surface error inline
    expr_str = st.session_state[expr_key]
    try:
        value = eval_constant_expression(expr_str)
        st.session_state[last_good_key] = value
    except ValueError as e:
        value = float(st.session_state.get(last_good_key, 0.0))
        st.caption(f":red[수식 오류: {e}]  (이전 값 {value:g} 사용)")

    return expr_str, value


def reset_numeric_expr_input(key: str, new_expr: str) -> None:
    """
    Force-reset a numeric_expr_input back to a new expression.

    Useful when a preset is loaded and we need to overwrite all the linked
    widget states (expr / slider / num). Must be called BEFORE the widgets
    render in the same script run.
    """
    expr_key = f"{key}__expr"
    slider_key = f"{key}__slider"
    num_key = f"{key}__num"
    last_good_key = f"{key}__last_float"

    st.session_state[expr_key] = new_expr
    try:
        v = eval_constant_expression(new_expr)
    except ValueError:
        v = 0.0
    st.session_state[last_good_key] = v
    st.session_state[slider_key] = v
    st.session_state[num_key] = v
