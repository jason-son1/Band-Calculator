"""
🔬 Band Dispersion Visualizer
Interactive Streamlit app for visualizing 2x2 Hamiltonian band structures.
"""

import streamlit as st
import numpy as np

from core.parser import (
    parse_expression, extract_free_params,
    detect_dimensionality, build_lambdified_funcs,
    build_lambdified_matrix_funcs,
    FUNCTION_REFERENCE,
)
from core.band_calculator import (
    compute_eigenvalues_1d, compute_eigenvalues_2d,
    compute_eigenvalues_1d_nxn, compute_eigenvalues_2d_nxn
)
from core.visualizer import plot_1d_bands, plot_2d_surface
from core.presets import PRESETS
from core.utils import (
    get_k_path_square, get_k_path_1d, get_k_path_hexagonal,
    get_k_path_custom, get_k_grid_2d, HIGH_SYMMETRY_POINTS,
)

# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Band Dispersion Visualizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State Initialization ────────────────────────────────────
if "d0" not in st.session_state:
    default_preset = PRESETS["QWZ (Qi-Wu-Zhang)"]
    st.session_state["d0"] = default_preset["d0"]
    st.session_state["dx"] = default_preset["dx"]
    st.session_state["dy"] = default_preset["dy"]
    st.session_state["dz"] = default_preset["dz"]
    st.session_state["_last_preset"] = "QWZ (Qi-Wu-Zhang)"
    st.session_state["kpath"] = default_preset.get("k_path", "square")

if "ui_mode" not in st.session_state:
    st.session_state["ui_mode"] = "2x2 (Pauli 방식)"
if "matrix_N" not in st.session_state:
    st.session_state["matrix_N"] = 4

# ─── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #06b6d4, #a855f7, #06b6d4);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem; font-weight: 700; margin: 0;
        animation: gradient-shift 3s ease infinite;
    }
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 100% center; }
    }
    .main-header p { color: #94a3b8; font-size: 0.9rem; margin: 0.3rem 0 0 0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1a1d2e 100%);
    }

    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #06b6d4 !important;
        box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.2) !important;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 27, 75, 0.6));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px; padding: 1rem 1.2rem; margin: 0.3rem 0;
    }
    .metric-card .label {
        color: #94a3b8; font-size: 0.75rem;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .metric-card .value {
        color: #e2e8f0; font-size: 1.4rem; font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card .value.gap-open { color: #06b6d4; }
    .metric-card .value.gap-closed { color: #f43f5e; }

    /* Section dividers */
    .section-divider {
        border: none; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.4), transparent);
        margin: 1rem 0;
    }

    /* Preset description */
    .preset-desc {
        background: rgba(99, 102, 241, 0.1);
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem; font-size: 0.82rem; color: #a5b4fc; margin: 0.5rem 0;
    }

    /* Formula display cards */
    .formula-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px; padding: 0.7rem 1rem; margin: 0.3rem 0;
        font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;
        color: #e2e8f0; cursor: default;
        transition: border-color 0.2s;
    }
    .formula-card:hover {
        border-color: #06b6d4;
    }
    .formula-card .formula-label {
        color: #06b6d4; font-weight: 600; font-size: 0.8rem;
        margin-bottom: 0.2rem;
    }
    .formula-card .formula-expr {
        color: #f0abfc; font-size: 0.85rem; word-break: break-all;
    }

    /* Function ref table */
    .func-ref {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 8px; padding: 0.5rem 0.8rem; margin: 0.3rem 0;
    }
    .func-ref .cat-title {
        color: #06b6d4; font-weight: 600; font-size: 0.78rem;
        margin: 0.3rem 0 0.2rem 0;
    }
    .func-ref .func-item {
        color: #cbd5e1; font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 0.1rem 0;
    }
    .func-ref .func-item code {
        color: #f0abfc; background: rgba(168, 85, 247, 0.15);
        padding: 0.1rem 0.3rem; border-radius: 3px;
    }

    /* Hide defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px 8px 0 0; color: #94a3b8; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: #6366f1 !important; color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 Band Dispersion Visualizer</h1>
    <p>2×2 Hamiltonian Band Structure — Real-time Parameter Tuning</p>
</div>
""", unsafe_allow_html=True)


# ─── Preset Callback ─────────────────────────────────────────────────
def _on_preset_change():
    """Callback: update session_state d-formulas when preset changes."""
    name = st.session_state["preset_select"]
    if name == st.session_state.get("_last_preset"):
        return
    preset = PRESETS[name]
    st.session_state["d0"] = preset["d0"]
    st.session_state["dx"] = preset["dx"]
    st.session_state["dy"] = preset["dy"]
    st.session_state["dz"] = preset["dz"]
    st.session_state["kpath"] = preset.get("k_path", "square")
    # Reset parameter sliders for the new preset
    for key in list(st.session_state.keys()):
        if key.startswith("param_") or key.startswith("paramnum_") or key.startswith("pval_"):
            del st.session_state[key]
    st.session_state["_last_preset"] = name


# ─── Formula Editor Dialog ───────────────────────────────────────────
@st.dialog("📐 Hamiltonian 수식 편집기", width="large")
def _formula_editor_dialog():
    """Full-screen modal dialog for editing formulas."""
    if st.session_state["ui_mode"] == "2x2 (Pauli 방식)":
        st.markdown(r"$H(\mathbf{k}) = d_0 \sigma_0 + d_x \sigma_x + d_y \sigma_y + d_z \sigma_z$")
    else:
        st.markdown(rf"$H(\mathbf{{k}})$: {st.session_state['matrix_N']} × {st.session_state['matrix_N']} Hermitian Matrix")
    
    st.markdown("---")

    # Reference panel at the top
    with st.expander("📚 사용 가능한 함수/변수", expanded=False):
        ref_cols = st.columns(3)
        categories = list(FUNCTION_REFERENCE.items())
        for idx, (cat, funcs) in enumerate(categories):
            with ref_cols[idx % 3]:
                st.markdown(f"**{cat}**")
                for fname, fdesc in funcs:
                    st.markdown(f"`{fname}` — {fdesc}")

    st.markdown("")

    new_vals = {}
    
    if st.session_state["ui_mode"] == "2x2 (Pauli 방식)":
        components = [
            ("d₀(k)", "d0", "σ₀ 성분 — 에너지 오프셋. 양 밴드를 동시에 이동."),
            ("dₓ(k)", "dx", "σₓ 성분 — 해밀토니안의 off-diagonal real part."),
            ("dᵧ(k)", "dy", "σᵧ 성분 — 해밀토니안의 off-diagonal imaginary part."),
            ("d_z(k)", "dz", "σ_z 성분 — 밴드 갭을 결정하는 diagonal 성분."),
        ]
        for label, key, help_text in components:
            st.markdown(f"#### {label}")
            st.caption(help_text)
            new_vals[key] = st.text_area(
                label,
                value=st.session_state.get(key, "0"),
                height=68,
                key=f"dialog_{key}",
                placeholder=f"{label} 수식을 입력하세요",
                label_visibility="collapsed",
            )
    else:
        N = st.session_state["matrix_N"]
        st.markdown("#### 우상단(Upper Triangular) 요소 입력")
        st.caption("대각선 및 우상단 요소를 입력하세요. 좌하단은 켤레복소수로 자동 처리됩니다.")
        
        for i in range(N):
            cols = st.columns(N)
            for j in range(N):
                with cols[j]:
                    if j >= i:
                        key = f"h_{i}_{j}"
                        new_vals[key] = st.text_input(
                            f"H_{i+1},{j+1}",
                            value=st.session_state.get(key, "0" if i!=j else "0"),
                            key=f"dialog_{key}",
                        )
                    else:
                        st.text_input(
                            f"H_{i+1},{j+1}",
                            value=f"(H_{j+1},{i+1})*",
                            disabled=True,
                            key=f"dialog_disabled_{i}_{j}"
                        )

    st.markdown("")
    col_ok, col_cancel = st.columns(2)
    with col_ok:
        if st.button("✅ 적용", use_container_width=True, type="primary"):
            for key, val in new_vals.items():
                st.session_state[key] = val
            # Reset params since formulas changed
            for k in list(st.session_state.keys()):
                if k.startswith("param_") or k.startswith("paramnum_") or k.startswith("pval_") or k.startswith("ni_") or k.startswith("sl_"):
                    del st.session_state[k]
            st.rerun()
    with col_cancel:
        if st.button("❌ 취소", use_container_width=True):
            st.rerun()


# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ UI Mode")
    ui_mode = st.radio(
        "해밀토니안 입력 방식",
        ["2x2 (Pauli 방식)", "N x N (Custom 방식)"],
        key="ui_mode",
    )
    
    if ui_mode == "N x N (Custom 방식)":
        matrix_N = st.number_input("행렬 크기 (N)", min_value=2, max_value=10, value=st.session_state.get("matrix_N", 4), step=1, key="matrix_N")
    
    st.markdown("### 🧩 Model Selection")
    
    if ui_mode == "2x2 (Pauli 방식)":
        preset_name = st.selectbox(
            "프리셋 모델",
            options=list(PRESETS.keys()),
            index=list(PRESETS.keys()).index(st.session_state.get("_last_preset", "QWZ (Qi-Wu-Zhang)")),
            key="preset_select",
            on_change=_on_preset_change,
            help="프리셋을 선택하면 수식과 k-경로가 자동 설정됩니다.",
        )
        preset = PRESETS[preset_name]

        if preset["description"]:
            st.markdown(f'<div class="preset-desc">{preset["description"]}</div>', unsafe_allow_html=True)
    else:
        preset_name = "Custom N x N"
        preset = {"params": {}}

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ─── Hamiltonian Input ────────────────────────────────────────
    st.markdown("### 📐 Hamiltonian H(k)")
    
    if ui_mode == "2x2 (Pauli 방식)":
        st.markdown(r"$H(\mathbf{k}) = d_0 \sigma_0 + d_x \sigma_x + d_y \sigma_y + d_z \sigma_z$")
        d_labels = [("d₀(k)", "d0"), ("dₓ(k)", "dx"), ("dᵧ(k)", "dy"), ("d_z(k)", "dz")]
        for label, key in d_labels:
            current_val = st.session_state.get(key, "0")
            st.markdown(f"""
            <div class="formula-card">
                <div class="formula-label">{label}</div>
                <div class="formula-expr">{current_val if current_val else '0'}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"**{matrix_N} × {matrix_N} Hermitian Matrix**")
        st.markdown("<span style='color:#a5b4fc; font-size: 0.85rem;'>✏️ '수식 편집기 열기'를 눌러 행렬 요소를 입력하세요.</span>", unsafe_allow_html=True)

    # Open the dialog editor
    if st.button("✏️ 수식 편집기 열기", use_container_width=True, type="primary"):
        _formula_editor_dialog()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ─── Parse expressions ────────────────────────────────────────
    parse_error = None
    expressions = []
    flat_expressions = [] # For parameter extraction

    try:
        if ui_mode == "2x2 (Pauli 방식)":
            d0_str = st.session_state.get("d0", "0")
            dx_str = st.session_state.get("dx", "0")
            dy_str = st.session_state.get("dy", "0")
            dz_str = st.session_state.get("dz", "0")
            
            for label, expr_str in [("d₀", d0_str), ("dₓ", dx_str), ("dᵧ", dy_str), ("d_z", dz_str)]:
                expr = parse_expression(expr_str if expr_str else "0")
                expressions.append(expr)
            flat_expressions = expressions
        else:
            N = st.session_state["matrix_N"]
            expressions = [[None for _ in range(N)] for _ in range(N)]
            for i in range(N):
                for j in range(i, N):
                    val = st.session_state.get(f"h_{i}_{j}", "0")
                    expr = parse_expression(val if val else "0")
                    expressions[i][j] = expr
                    flat_expressions.append(expr)
    except ValueError as e:
        parse_error = str(e)

    if parse_error:
        st.error(f"⚠️ {parse_error}")
        st.stop()

    # ─── Extract free parameters and create sliders ───────────────
    free_params = extract_free_params(flat_expressions)
    dimensionality = detect_dimensionality(flat_expressions)
    preset_params = preset.get("params", {})

    param_values = {}
    if free_params:
        st.markdown("### 🎛️ Parameter Tuning")
        for param in free_params:
            pname = str(param)
            p_info = preset_params.get(pname, {})
            p_min = float(p_info.get("min", -5.0))
            p_max = float(p_info.get("max", 5.0))
            p_default = float(p_info.get("default", 0.0))
            p_step = float(p_info.get("step", 0.01))
            p_default = max(p_min, min(p_max, p_default))

            # Use number_input key as the single source of truth
            ni_key = f"ni_{pname}"
            sl_key = f"sl_{pname}"

            if ni_key not in st.session_state:
                st.session_state[ni_key] = p_default

            # When slider changes, update the number_input key
            def _sync_slider_to_ni(s=sl_key, n=ni_key):
                st.session_state[n] = st.session_state[s]

            col_s, col_n = st.columns([3, 1])
            with col_s:
                st.slider(
                    f"{pname}",
                    min_value=p_min,
                    max_value=p_max,
                    value=float(st.session_state[ni_key]),
                    step=p_step,
                    key=sl_key,
                    on_change=_sync_slider_to_ni,
                )
            with col_n:
                st.number_input(
                    f"{pname} value",
                    min_value=p_min,
                    max_value=p_max,
                    step=p_step,
                    key=ni_key,
                    label_visibility="collapsed",
                )

            param_values[pname] = float(st.session_state[ni_key])
    else:
        st.info("ℹ️ 자유 파라미터가 없습니다.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ─── Visualization settings ───────────────────────────────────
    st.markdown("### ⚙️ Settings")

    n_k_1d = st.slider("1D k-points", min_value=50, max_value=500, value=200, step=50, key="nk1d")

    k_path_options = {
        "square": "Square (Γ→X→M→Γ)",
        "hexagonal": "Hexagonal (Γ→M→K→Γ)",
        "1d": "1D (-π→0→π)",
        "custom": "✏️ Custom (직접 입력)",
    }
    k_path_type = st.selectbox(
        "k-경로 타입",
        options=list(k_path_options.keys()),
        format_func=lambda x: k_path_options[x],
        key="kpath",
    )

    # ─── Custom k-path input ──────────────────────────────────────
    custom_points = []
    if k_path_type == "custom":
        st.markdown("#### Custom k-path 정의")
        st.caption("고대칭점을 순서대로 추가하세요. 좌표 단위: π")

        # Show available high-symmetry points
        with st.expander("📍 고대칭점 라이브러리", expanded=False):
            for name, (kx, ky, kz) in HIGH_SYMMETRY_POINTS.items():
                st.markdown(f"**{name}**: ({kx/np.pi:.3f}π, {ky/np.pi:.3f}π, {kz/np.pi:.3f}π)")

        n_custom_pts = st.number_input("경로 점 개수", min_value=2, max_value=10, value=4, step=1, key="n_custom_pts")
        default_labels = ["Γ", "X", "M", "Γ", "Y", "S", "R", "K", "K'", "M_hex"]
        default_coords = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (0.5, 0.5, 0.0), (1.0, 1.0, 1.0), (4/3, 0.0, 0.0),
            (2/3, 2/np.sqrt(3), 0.0), (1.0, 1/np.sqrt(3), 0.0),
        ]

        for i in range(int(n_custom_pts)):
            c1, c2, c3, c4 = st.columns([1, 1.1, 1.1, 1.1])
            dl = default_labels[i] if i < len(default_labels) else f"P{i}"
            dkx = default_coords[i][0] if i < len(default_coords) else 0.0
            dky = default_coords[i][1] if i < len(default_coords) else 0.0
            dkz = default_coords[i][2] if i < len(default_coords) else 0.0
            with c1:
                lbl = st.text_input(f"Label {i+1}", value=dl, key=f"cpt_lbl_{i}")
            with c2:
                ckx = st.number_input(f"kx/π {i+1}", value=dkx, step=0.1, key=f"cpt_kx_{i}", format="%.3f")
            with c3:
                cky = st.number_input(f"ky/π {i+1}", value=dky, step=0.1, key=f"cpt_ky_{i}", format="%.3f")
            with c4:
                ckz = st.number_input(f"kz/π {i+1}", value=dkz, step=0.1, key=f"cpt_kz_{i}", format="%.3f")
            custom_points.append({"label": lbl, "kx": ckx * np.pi, "ky": cky * np.pi, "kz": ckz * np.pi})

    show_2d = st.checkbox("2D Surface Plot 표시", value=(dimensionality >= 2), key="show2d")

    if show_2d:
        n_k_2d = st.slider("2D grid resolution", min_value=30, max_value=150, value=80, step=10, key="nk2d")

    # ─── Function Reference ───────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    with st.expander("📚 사용 가능한 함수/변수 레퍼런스", expanded=False):
        for category, funcs in FUNCTION_REFERENCE.items():
            st.markdown(f'<div class="func-ref"><div class="cat-title">{category}</div>', unsafe_allow_html=True)
            for fname, fdesc in funcs:
                st.markdown(f'<div class="func-item"><code>{fname}</code> — {fdesc}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ─── Computation ─────────────────────────────────────────────────────
if ui_mode == "2x2 (Pauli 방식)":
    d_funcs = build_lambdified_funcs(expressions, free_params)
else:
    d_funcs = build_lambdified_matrix_funcs(expressions, free_params)

# 1D Band Calculation
if k_path_type == "square":
    k_vals_1d, k_pts_1d, k_ticks, k_labels = get_k_path_square(n_k_1d)
elif k_path_type == "hexagonal":
    k_vals_1d, k_pts_1d, k_ticks, k_labels = get_k_path_hexagonal(n_k_1d)
elif k_path_type == "custom" and len(custom_points) >= 2:
    k_vals_1d, k_pts_1d, k_ticks, k_labels = get_k_path_custom(custom_points, n_k_1d)
else:
    k_vals_1d, k_pts_1d, k_ticks, k_labels = get_k_path_1d(n_k_1d)

if ui_mode == "2x2 (Pauli 방식)":
    eigenvalues_1d, band_gap_1d = compute_eigenvalues_1d(d_funcs, k_pts_1d, param_values, free_params)
else:
    eigenvalues_1d, band_gap_1d = compute_eigenvalues_1d_nxn(d_funcs, k_pts_1d, param_values, free_params)

# 2D Surface Calculation (only if enabled)
if show_2d:
    kx_mesh, ky_mesh = get_k_grid_2d(n_k_2d)
    if ui_mode == "2x2 (Pauli 방식)":
        e_plus, e_minus, band_gap_2d = compute_eigenvalues_2d(d_funcs, kx_mesh, ky_mesh, param_values, free_params)
        eigenvalues_2d_list = [e_minus, e_plus]
    else:
        eigenvalues_2d_list, band_gap_2d = compute_eigenvalues_2d_nxn(d_funcs, kx_mesh, ky_mesh, param_values, free_params)


# ─── Info Metrics ────────────────────────────────────────────────────
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    gap_class = "gap-open" if band_gap_1d > 1e-6 else "gap-closed"
    gap_text = f"{band_gap_1d:.4f}" if band_gap_1d > 1e-6 else "CLOSED"
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Band Gap</div>
        <div class="value {gap_class}">{gap_text}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    e_range = f"{eigenvalues_1d.min():.2f} ~ {eigenvalues_1d.max():.2f}"
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Energy Range</div>
        <div class="value">{e_range}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    param_text = ", ".join([f"{k}={v:.2f}" for k, v in param_values.items()]) if param_values else "None"
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Parameters</div>
        <div class="value" style="font-size:0.95rem;">{param_text}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Plots ───────────────────────────────────────────────────────────
if show_2d:
    tab1, tab2 = st.tabs(["📈 1D Band Structure", "🌐 2D Energy Surface"])
else:
    tab1 = st.container()
    tab2 = None

with tab1:
    path_label = k_path_options.get(k_path_type, "k-path")
    fig_1d = plot_1d_bands(
        k_vals_1d, eigenvalues_1d, k_ticks, k_labels,
        title=f"Band Structure — {preset_name} ({path_label})"
    )
    st.plotly_chart(fig_1d, width="stretch", key="plot_1d")

if show_2d and tab2 is not None:
    with tab2:
        fig_2d = plot_2d_surface(
            kx_mesh, ky_mesh, eigenvalues_2d_list,
            title=f"Energy Surface — {preset_name}"
        )
        st.plotly_chart(fig_2d, width="stretch", key="plot_2d")


# ─── Hamiltonian Display ─────────────────────────────────────────────
with st.expander("📝 Current Hamiltonian (SymPy)", expanded=False):
    import sympy as sp
    if ui_mode == "2x2 (Pauli 방식)":
        st.latex(r"H(\mathbf{k}) = d_0 \sigma_0 + d_x \sigma_x + d_y \sigma_y + d_z \sigma_z")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.latex(f"d_0 = {sp.latex(expressions[0])}")
            st.latex(f"d_x = {sp.latex(expressions[1])}")
        with col_h2:
            st.latex(f"d_y = {sp.latex(expressions[2])}")
            st.latex(f"d_z = {sp.latex(expressions[3])}")
    else:
        st.markdown("**N x N Matrix (Upper Triangular):**")
        for i in range(st.session_state["matrix_N"]):
            for j in range(i, st.session_state["matrix_N"]):
                expr = expressions[i][j]
                if expr is not None and expr != 0:
                    st.latex(f"H_{{{i+1},{j+1}}} = {sp.latex(expr)}")
