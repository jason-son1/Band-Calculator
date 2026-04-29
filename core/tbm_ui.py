"""
TBM Visual Builder — Streamlit UI module.

Renders the entire TBM Builder interface (sidebar + main area).
Called from app.py when ui_mode == "🔗 TBM Visual Builder".
"""
from __future__ import annotations
import streamlit as st
import numpy as np
import sympy as sp

from core.tbm_model import (
    TBMModel, Lattice, Site, State, Hopping,
    ORBITAL_OPTIONS, SPIN_OPTIONS,
    build_graphene_model, build_ssh_model, build_qwz_model,
)
from core.tbm_visualizer import plot_tbm_diagram
from core.parser import build_lambdified_matrix_funcs, K_SYMBOLS
from core.band_calculator import compute_eigenvalues_1d_nxn, compute_eigenvalues_2d_nxn
from core.visualizer import plot_1d_bands, plot_2d_surface
from core.utils import (
    get_k_path_square, get_k_path_hexagonal, get_k_path_1d,
    get_k_path_custom, get_k_grid_2d, HIGH_SYMMETRY_POINTS,
)


# ── Session state helpers ─────────────────────────────────────────────

def _init_tbm_state():
    if "tbm_model" not in st.session_state:
        st.session_state["tbm_model"] = TBMModel()
    if "tbm_preset" not in st.session_state:
        st.session_state["tbm_preset"] = "빈 모델"


def _get_model() -> TBMModel:
    return st.session_state["tbm_model"]


def _set_model(model: TBMModel):
    st.session_state["tbm_model"] = model


# ── Preset loader ─────────────────────────────────────────────────────

TBM_PRESETS = {
    "빈 모델": None,
    "SSH Chain (1D 위상)": build_ssh_model,
    "Graphene (Dirac 반금속)": build_graphene_model,
    "QWZ Model (검증용 2×2)": build_qwz_model,
}


# ══════════════════════════════════════════════════════════════════════
# Sidebar UI
# ══════════════════════════════════════════════════════════════════════

def render_tbm_sidebar():
    """Render the TBM Builder sidebar. Returns (k_path_type, k_path_args, n_k, show_2d, n_k_2d)."""
    _init_tbm_state()
    model = _get_model()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📐 Lattice")

    dim = st.selectbox("차원", [1, 2, 3],
                       index=[1, 2, 3].index(model.lattice.dimension),
                       key="tbm_lattice_dim")
    model.lattice.dimension = dim

    with st.expander("격자 벡터 설정", expanded=False):
        for vec_name in (["a1"] if dim == 1 else ["a1", "a2"] if dim == 2 else ["a1", "a2", "a3"]):
            cur = getattr(model.lattice, vec_name)
            cols = st.columns(3)
            new_v = []
            for ci, comp in enumerate(["x", "y", "z"]):
                with cols[ci]:
                    new_v.append(st.number_input(
                        f"{vec_name}_{comp}", value=float(cur[ci]),
                        step=0.1, format="%.2f",
                        key=f"tbm_lat_{vec_name}_{comp}",
                        label_visibility="collapsed" if ci > 0 else "visible",
                    ))
            setattr(model.lattice, vec_name, new_v)

    # ── Sites ─────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 🔵 Sites")

    for site in list(model.sites):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**{site.name}** — `{[round(p,2) for p in site.position]}`")
        with c2:
            if st.button("🗑", key=f"del_site_{site.uid}", help="삭제"):
                model.remove_site(site.uid)
                st.rerun()

    with st.expander("➕ Site 추가", expanded=False):
        sname = st.text_input("이름", value=f"S{len(model.sites)}", key="new_site_name")
        sc = st.columns(3)
        sx = sc[0].number_input("x", value=0.0, step=0.1, format="%.2f", key="new_site_x")
        sy = sc[1].number_input("y", value=0.0, step=0.1, format="%.2f", key="new_site_y")
        sz = sc[2].number_input("z", value=0.0, step=0.1, format="%.2f", key="new_site_z")
        if st.button("Site 추가", key="add_site_btn", use_container_width=True):
            model.add_site(Site(name=sname, position=[sx, sy, sz]))
            st.rerun()

    # ── States ────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f"### ⚛️ States (기저 차원: **{model.matrix_dimension()}**)")

    for state in list(model.states):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"`{state.label()}`")
        with c2:
            if st.button("🗑", key=f"del_state_{state.uid}", help="삭제"):
                model.remove_state(state.uid)
                st.rerun()

    with st.expander("➕ State 추가", expanded=len(model.sites) > 0 and len(model.states) == 0):
        if not model.sites:
            st.caption("먼저 Site를 추가하세요.")
        else:
            site_opts = {s.name: s for s in model.sites}
            sel_site_name = st.selectbox("Site", list(site_opts.keys()), key="new_state_site")
            sel_orbital = st.selectbox("Orbital", ORBITAL_OPTIONS, key="new_state_orbital")
            sel_spin = st.selectbox("Spin", SPIN_OPTIONS, key="new_state_spin")
            if st.button("State 추가", key="add_state_btn", use_container_width=True):
                model.add_state(State(
                    site=site_opts[sel_site_name],
                    orbital=sel_orbital,
                    spin=sel_spin,
                ))
                st.rerun()

    # ── Hoppings ──────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f"### 🔗 Hoppings ({len(model.hoppings)}개)")

    for hop in list(model.hoppings):
        src = model.get_state(hop.source_uid)
        tgt = model.get_state(hop.target_uid)
        if src and tgt:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"`{src.label()}` → `{tgt.label()}` | t=`{hop.amplitude}` R={hop.R}")
            with c2:
                if st.button("🗑", key=f"del_hop_{hop.uid}", help="삭제"):
                    model.remove_hopping(hop.uid)
                    st.rerun()

    with st.expander("➕ Hopping 추가", expanded=False):
        if len(model.states) < 1:
            st.caption("먼저 State를 추가하세요.")
        else:
            state_opts = {st_obj.label(): st_obj.uid for st_obj in model.states}
            labels = list(state_opts.keys())
            h_src = st.selectbox("Source State", labels, key="new_hop_src")
            h_tgt = st.selectbox("Target State", labels, key="new_hop_tgt")
            h_amp = st.text_input("Amplitude (t)", value="1.0",
                                  help="실수, 허수 모두 가능. 예: t, 1.5, t1*exp(I*phi)",
                                  key="new_hop_amp")
            rc = st.columns(3)
            hr0 = int(rc[0].number_input("Rx", value=0, step=1, key="new_hop_Rx"))
            hr1 = int(rc[1].number_input("Ry", value=0, step=1, key="new_hop_Ry"))
            hr2 = int(rc[2].number_input("Rz", value=0, step=1, key="new_hop_Rz"))
            if st.button("Hopping 추가", key="add_hop_btn", use_container_width=True):
                model.add_hopping(Hopping(
                    source_uid=state_opts[h_src],
                    target_uid=state_opts[h_tgt],
                    amplitude=h_amp,
                    R=[hr0, hr1, hr2],
                ))
                st.rerun()

    # ── Settings ──────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### ⚙️ 계산 설정")

    n_k = st.slider("1D k-points", 50, 500, 200, 50, key="tbm_nk1d")
    k_path_options = {
        "square":     "Square (Γ→X→M→Γ)",
        "hexagonal":  "Hexagonal (Γ→M→K→Γ)",
        "1d":         "1D (−π→0→π)",
        "custom":     "✏️ Custom",
    }
    k_path_type = st.selectbox(
        "k-경로 타입",
        list(k_path_options.keys()),
        format_func=lambda x: k_path_options[x],
        key="tbm_kpath",
    )

    custom_points = []
    if k_path_type == "custom":
        with st.expander("📍 고대칭점 라이브러리", expanded=False):
            for name, (kx, ky, kz) in HIGH_SYMMETRY_POINTS.items():
                st.markdown(f"**{name}**: ({kx/np.pi:.2f}π, {ky/np.pi:.2f}π, {kz/np.pi:.2f}π)")
        n_pts = st.number_input("점 개수", 2, 10, 4, key="tbm_n_custom_pts")
        def_lbls = ["Γ", "X", "M", "Γ", "Y", "R"]
        def_coords = [(0,0,0),(1,0,0),(1,1,0),(0,0,0),(0,1,0),(1,1,1)]
        for i in range(int(n_pts)):
            cc = st.columns([1,1,1,1])
            lbl = cc[0].text_input(f"Lbl{i+1}", def_lbls[i] if i < len(def_lbls) else f"P{i}", key=f"tbm_cpt_lbl_{i}")
            ckx = cc[1].number_input(f"kx{i+1}", value=float(def_coords[i][0]) if i < len(def_coords) else 0.0, step=0.1, format="%.2f", key=f"tbm_cpt_kx_{i}")
            cky = cc[2].number_input(f"ky{i+1}", value=float(def_coords[i][1]) if i < len(def_coords) else 0.0, step=0.1, format="%.2f", key=f"tbm_cpt_ky_{i}")
            ckz = cc[3].number_input(f"kz{i+1}", value=float(def_coords[i][2]) if i < len(def_coords) else 0.0, step=0.1, format="%.2f", key=f"tbm_cpt_kz_{i}")
            custom_points.append({"label": lbl, "kx": ckx*np.pi, "ky": cky*np.pi, "kz": ckz*np.pi})

    show_2d = st.checkbox("2D Surface Plot", value=False, key="tbm_show2d")
    n_k_2d = 60
    if show_2d:
        n_k_2d = st.slider("2D grid resolution", 20, 100, 60, 10, key="tbm_nk2d")

    return k_path_type, custom_points, n_k, show_2d, n_k_2d


# ══════════════════════════════════════════════════════════════════════
# Main Area UI
# ══════════════════════════════════════════════════════════════════════

def render_tbm_main(k_path_type, custom_points, n_k, show_2d, n_k_2d):
    """Render TBM diagram, H(k) preview, parameter sliders, and band plots."""
    _init_tbm_state()
    model = _get_model()

    # ── Top row: Preset selector + clear ─────────────────────────────
    col_pre, col_clr = st.columns([4, 1])
    with col_pre:
        preset_choice = st.selectbox(
            "🚀 빠른 시작 (프리셋 로드)",
            list(TBM_PRESETS.keys()),
            key="tbm_preset_sel",
        )
    with col_clr:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📂 로드", key="tbm_load_preset", use_container_width=True):
            fn = TBM_PRESETS[preset_choice]
            _set_model(fn() if fn else TBMModel())
            st.rerun()

    # ── Diagram + H(k) preview ────────────────────────────────────────
    diag_col, matrix_col = st.columns([6, 4])

    with diag_col:
        fig_diag = plot_tbm_diagram(model, title="TBM Diagram")
        st.plotly_chart(fig_diag, use_container_width=True, key="tbm_diagram")

    with matrix_col:
        st.markdown("#### $H(\\mathbf{k})$ — SymPy 행렬")
        if model.matrix_dimension() == 0:
            st.info("State를 추가하면 행렬이 여기에 표시됩니다.")
        else:
            try:
                H_sym = model.build_hamiltonian_matrix()
                N = H_sym.shape[0]
                # Show each non-zero element as LaTeX
                shown = 0
                for i in range(N):
                    for j in range(N):
                        elem = H_sym[i, j]
                        if elem != 0:
                            st.latex(f"H_{{{i+1},{j+1}}} = {sp.latex(sp.simplify(elem))}")
                            shown += 1
                            if shown >= 12:
                                st.caption(f"... ({N*N - shown}개 항목 생략)")
                                break
                    if shown >= 12:
                        break
            except Exception as e:
                st.error(f"행렬 생성 오류: {e}")

    # ── Guard: need states + hoppings ────────────────────────────────
    if model.matrix_dimension() == 0:
        st.info("사이드바에서 Site → State → Hopping 순서로 추가하면 밴드 구조가 계산됩니다.")
        return

    # ── Build lambdified funcs ────────────────────────────────────────
    try:
        H_sym = model.build_hamiltonian_matrix()
        expr_matrix = [[H_sym[i, j] for j in range(H_sym.shape[0])]
                       for i in range(H_sym.shape[0])]

        # Extract free parameters (excluding k symbols)
        k_reserved = set(K_SYMBOLS.values())
        raw_free = sorted(H_sym.free_symbols - k_reserved, key=lambda s: s.name)
    except Exception as e:
        st.error(f"해밀토니안 빌드 오류: {e}")
        return

    # ── Parameter sliders ─────────────────────────────────────────────
    param_values: dict[str, float] = {}
    if raw_free:
        st.markdown("### 🎛️ Parameter Tuning")
        pcols = st.columns(min(len(raw_free), 3))
        for idx, sym in enumerate(raw_free):
            pname = str(sym)
            ni_key = f"tbm_ni_{pname}"
            sl_key = f"tbm_sl_{pname}"
            if ni_key not in st.session_state:
                st.session_state[ni_key] = 1.0

            def _sync(s=sl_key, n=ni_key):
                st.session_state[n] = st.session_state[s]

            with pcols[idx % len(pcols)]:
                st.slider(pname, -5.0, 5.0,
                          value=float(st.session_state[ni_key]),
                          step=0.05, key=sl_key, on_change=_sync)
                st.number_input(f"{pname} 값", -5.0, 5.0,
                                step=0.05, key=ni_key,
                                label_visibility="collapsed")
            param_values[pname] = float(st.session_state[ni_key])

    # ── Build callable matrix functions ───────────────────────────────
    try:
        func_matrix = build_lambdified_matrix_funcs(expr_matrix, raw_free)
    except Exception as e:
        st.error(f"lambdify 오류: {e}")
        return

    # ── k-path ────────────────────────────────────────────────────────
    if k_path_type == "square":
        k_vals, k_pts, k_ticks, k_lbls = get_k_path_square(n_k)
    elif k_path_type == "hexagonal":
        k_vals, k_pts, k_ticks, k_lbls = get_k_path_hexagonal(n_k)
    elif k_path_type == "custom" and len(custom_points) >= 2:
        k_vals, k_pts, k_ticks, k_lbls = get_k_path_custom(custom_points, n_k)
    else:
        k_vals, k_pts, k_ticks, k_lbls = get_k_path_1d(n_k)

    # ── Eigenvalues ───────────────────────────────────────────────────
    try:
        eigenvalues_1d, band_gap = compute_eigenvalues_1d_nxn(
            func_matrix, k_pts, param_values, raw_free
        )
    except Exception as e:
        st.error(f"고유값 계산 오류: {e}")
        return

    # ── Metrics ───────────────────────────────────────────────────────
    mc1, mc2, mc3 = st.columns(3)
    gap_class = "gap-open" if band_gap > 1e-6 else "gap-closed"
    gap_text = f"{band_gap:.4f}" if band_gap > 1e-6 else "CLOSED"
    with mc1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Band Gap</div>
            <div class="value {gap_class}">{gap_text}</div></div>""",
                    unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Energy Range</div>
            <div class="value">{eigenvalues_1d.min():.2f} ~ {eigenvalues_1d.max():.2f}</div></div>""",
                    unsafe_allow_html=True)
    with mc3:
        param_str = ", ".join(f"{k}={v:.2f}" for k, v in param_values.items()) or "None"
        st.markdown(f"""<div class="metric-card">
            <div class="label">Parameters</div>
            <div class="value" style="font-size:0.9rem;">{param_str}</div></div>""",
                    unsafe_allow_html=True)

    # ── Band plots ────────────────────────────────────────────────────
    N_bands = model.matrix_dimension()
    model_name = f"TBM ({N_bands}×{N_bands})"

    if show_2d:
        tab1, tab2 = st.tabs(["📈 1D Band Structure", "🌐 2D Energy Surface"])
    else:
        tab1 = st.container()
        tab2 = None

    k_path_labels = {
        "square": "Square (Γ→X→M→Γ)",
        "hexagonal": "Hexagonal (Γ→M→K→Γ)",
        "1d": "1D",
        "custom": "Custom",
    }
    with tab1:
        fig_1d = plot_1d_bands(
            k_vals, eigenvalues_1d, k_ticks, k_lbls,
            title=f"Band Structure — {model_name} ({k_path_labels.get(k_path_type, '')})"
        )
        st.plotly_chart(fig_1d, use_container_width=True, key="tbm_plot_1d")

    if show_2d and tab2 is not None:
        with tab2:
            try:
                kx_mesh, ky_mesh = get_k_grid_2d(n_k_2d)
                ev_2d, _ = compute_eigenvalues_2d_nxn(
                    func_matrix, kx_mesh, ky_mesh, param_values, raw_free
                )
                fig_2d = plot_2d_surface(kx_mesh, ky_mesh, ev_2d,
                                         title=f"Energy Surface — {model_name}")
                st.plotly_chart(fig_2d, use_container_width=True, key="tbm_plot_2d")
            except Exception as e:
                st.error(f"2D 계산 오류: {e}")
