"""
TBM Visual Builder — Streamlit UI module (Phase 2 & 3 통합).
Renders the entire TBM Builder interface (sidebar + main area).
"""
from __future__ import annotations
import streamlit as st
import numpy as np
import sympy as sp

from core.tbm_model import (
    TBMModel, Lattice, BasisConfig, Site, State, Hopping,
    ORBITAL_OPTIONS, SPIN_OPTIONS,
    build_graphene_model, build_ssh_model, build_qwz_model,
    build_kane_mele_model, build_rashba_2d_model,
)
from core.tbm_visualizer import generate_pyvis_html
from core.color_manager import ColorManager
from core.parser import build_lambdified_matrix_funcs, K_SYMBOLS
from core.widgets import numeric_expr_input, reset_numeric_expr_input
from core.band_calculator import compute_eigenvalues_1d_nxn, compute_eigenvalues_2d_nxn
from core.visualizer import plot_1d_bands, plot_2d_surface
from core.utils import (
    get_k_path_square, get_k_path_hexagonal, get_k_path_1d,
    get_k_path_custom, get_k_grid_2d, HIGH_SYMMETRY_POINTS,
)
from core.brillouin_zone import BrillouinZoneAnalyzer
import streamlit.components.v1 as components
import os

# Initialize PyVis Component
PYVIS_COMPONENT_PATH = os.path.join(os.path.dirname(__file__), "pyvis_component")
try:
    pyvis_interaction = components.declare_component("pyvis_interaction", path=PYVIS_COMPONENT_PATH)
except Exception:
    pyvis_interaction = None

# ── Session state helpers ─────────────────────────────────────────────
def _init_tbm_state():
    if "tbm_model" not in st.session_state:
        st.session_state["tbm_model"] = TBMModel()
    if "tbm_color_mgr" not in st.session_state:
        st.session_state["tbm_color_mgr"] = ColorManager()
    if "selected_hop_uid" not in st.session_state:
        st.session_state["selected_hop_uid"] = None
    if "selected_site_uid" not in st.session_state:
        st.session_state["selected_site_uid"] = None
    if "selected_state_uid" not in st.session_state:
        st.session_state["selected_state_uid"] = None

def _get_model() -> TBMModel:
    return st.session_state["tbm_model"]

def _set_model(model: TBMModel):
    st.session_state["tbm_model"] = model
    st.session_state["tbm_color_mgr"] = ColorManager()
    st.session_state["selected_hop_uid"] = None

def _get_color_mgr() -> ColorManager:
    return st.session_state["tbm_color_mgr"]

# ── Preset loader ─────────────────────────────────────────────────────
TBM_PRESETS = {
    "빈 모델": None,
    "SSH Chain (1D 위상)": build_ssh_model,
    "Graphene (Dirac 반금속)": build_graphene_model,
    "QWZ Model (검증용 2×2)": build_qwz_model,
    "Kane-Mele (Z₂ TI + SOC)": build_kane_mele_model,
    "Rashba 2D (스핀 분리)": build_rashba_2d_model,
}

# ══════════════════════════════════════════════════════════════════════
# Sidebar UI
# ══════════════════════════════════════════════════════════════════════
def render_tbm_sidebar():
    """Render the TBM Builder sidebar. Returns (k_path_type, custom_points, n_k, show_2d, n_k_2d)."""
    _init_tbm_state()
    model = _get_model()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Basis Config ──────────────────────────────────────────────────
    st.markdown("### 🔬 Basis Configuration")
    spin_on = st.checkbox("Enable Spin (↑↓)", value=model.basis_config.spin_enabled,
                          key="tbm_spin_enabled",
                          help="활성화하면 각 State에 spin ↑/↓ 자유도가 추가됩니다.")
    model.basis_config.spin_enabled = spin_on
    if spin_on:
        st.caption(f"Internal dim = {model.basis_config.internal_dim} (spin ⊗ orbital)")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📐 Lattice")
    dim = st.selectbox("차원", [1, 2, 3],
                       index=[1, 2, 3].index(model.lattice.dimension),
                       key="tbm_lattice_dim")
    model.lattice.dimension = dim

    with st.expander("격자 벡터 설정", expanded=False):
        st.caption("수식 입력 가능 — 예: `sqrt(3)/2`, `pi/4`")
        vec_names = ["a1"] if dim == 1 else ["a1", "a2"] if dim == 2 else ["a1", "a2", "a3"]
        for vec_name in vec_names:
            cur_expr = getattr(model.lattice, f"{vec_name}_expr")
            new_floats = []
            new_exprs = []
            for ci, comp in enumerate(["x", "y", "z"]):
                e, v = numeric_expr_input(
                    label=f"{vec_name}_{comp}",
                    key=f"tbm_lat_{vec_name}_{comp}",
                    default_expr=str(cur_expr[ci]),
                    min_v=-3.0, max_v=3.0, step=0.05,
                    fmt="%.4f",
                    show_label=True,
                )
                new_floats.append(v)
                new_exprs.append(e)
            setattr(model.lattice, vec_name, new_floats)
            setattr(model.lattice, f"{vec_name}_expr", new_exprs)

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
        st.caption("Position (수식 가능)")
        ex_x, sx = numeric_expr_input("x", key="new_site_x", default_expr="0",
                                       min_v=-2.0, max_v=2.0, step=0.05,
                                       show_label=True)
        ex_y, sy = numeric_expr_input("y", key="new_site_y", default_expr="0",
                                       min_v=-2.0, max_v=2.0, step=0.05,
                                       show_label=True)
        ex_z, sz = numeric_expr_input("z", key="new_site_z", default_expr="0",
                                       min_v=-2.0, max_v=2.0, step=0.05,
                                       show_label=True)
        if st.button("Site 추가", key="add_site_btn", use_container_width=True):
            model.add_site(Site(name=sname,
                                position=[sx, sy, sz],
                                position_expr=[ex_x, ex_y, ex_z]))
            st.rerun()

    # ── States ────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    total_dim = model.total_dimension()
    st.markdown(f"### ⚛️ States (H dim: **{total_dim}×{total_dim}**)")
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
            if sel_orbital == "custom":
                custom_orbital_name = st.text_input(
                    "Custom Orbital 이름",
                    key="new_state_custom_orbital",
                    placeholder="예: p_+, d_{x^2-y^2}, p_x + ip_y",
                    help="다이어그램 라벨과 행렬 표시에 그대로 사용됩니다.",
                )
                actual_orbital = custom_orbital_name.strip() or "custom"
            else:
                actual_orbital = sel_orbital
            sel_spin = st.selectbox("Spin", SPIN_OPTIONS, key="new_state_spin")
            if st.button("State 추가", key="add_state_btn", use_container_width=True):
                model.add_state(State(
                    site=site_opts[sel_site_name], orbital=actual_orbital, spin=sel_spin,
                ))
                st.rerun()

    # ── Hoppings ──────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f"### 🔗 Hoppings ({len(model.hoppings)}개)")
    color_mgr = _get_color_mgr()

    for hop in list(model.hoppings):
        src = model.get_state(hop.source_uid)
        tgt = model.get_state(hop.target_uid)
        if src and tgt:
            clr = color_mgr.assign_color(hop.uid, is_onsite=hop.is_onsite())
            c1, c2 = st.columns([3, 1])
            with c1:
                amp_disp = hop.amplitude_matrix.strip() if hop.amplitude_matrix.strip() else hop.amplitude
                st.markdown(
                    f'{color_mgr.get_html_colored(hop.uid, "●")} '
                    f'`{src.label()}` → `{tgt.label()}` | t=`{amp_disp}` R={hop.R}',
                    unsafe_allow_html=True)
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
            h_amp = st.text_input("Amplitude (scalar)", value="1.0",
                                  help="실수/복소수. 예: t, 1.5, t1*exp(I*phi)",
                                  key="new_hop_amp")
            h_amp_matrix = ""
            if model.basis_config.spin_enabled:
                h_amp_matrix = st.text_input(
                    "Matrix Amplitude (Pauli)", value="",
                    help="예: t*sigma_z, t*s0 + I*lambda_R*sy. 비어있으면 스칼라×단위행렬 사용.",
                    key="new_hop_amp_matrix")
            rc = st.columns(3)
            hr0 = int(rc[0].number_input("Rx", value=0, step=1, key="new_hop_Rx"))
            hr1 = int(rc[1].number_input("Ry", value=0, step=1, key="new_hop_Ry"))
            hr2 = int(rc[2].number_input("Rz", value=0, step=1, key="new_hop_Rz"))
            if st.button("Hopping 추가", key="add_hop_btn", use_container_width=True):
                model.add_hopping(Hopping(
                    source_uid=state_opts[h_src], target_uid=state_opts[h_tgt],
                    amplitude=h_amp, amplitude_matrix=h_amp_matrix,
                    R=[hr0, hr1, hr2],
                ))
                st.rerun()

    # ── Settings ──────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### ⚙️ 계산 설정")
    n_k = st.slider("1D k-points", 50, 500, 200, 50, key="tbm_nk1d")
    k_path_options = {
        "auto": "Auto (격자 기반 자동 선택)",
        "square": "Square (Γ→X→M→Γ)", "hexagonal": "Hexagonal (Γ→M→K→Γ)",
        "1d": "1D (−π→0→π)", "custom": "✏️ Custom",
    }
    k_path_type = st.selectbox("k-경로 타입", list(k_path_options.keys()),
                               format_func=lambda x: k_path_options[x], key="tbm_kpath")

    custom_points = []
    if k_path_type == "custom":
        with st.expander("📍 고대칭점 라이브러리", expanded=False):
            for name, (kx, ky, kz) in HIGH_SYMMETRY_POINTS.items():
                st.markdown(f"**{name}**: ({kx/np.pi:.2f}π, {ky/np.pi:.2f}π, {kz/np.pi:.2f}π)")
        n_pts = st.number_input("점 개수", 2, 10, 4, key="tbm_n_custom_pts")
        def_lbls = ["Γ", "X", "M", "Γ", "Y", "R"]
        def_coords = [(0,0,0),(1,0,0),(1,1,0),(0,0,0),(0,1,0),(1,1,1)]
        st.caption("좌표는 π 단위. 수식 가능 — 예: `1/2`, `2/3`, `sqrt(3)/3`")
        for i in range(int(n_pts)):
            lbl = st.text_input(
                f"Point {i+1} Label",
                def_lbls[i] if i < len(def_lbls) else f"P{i}",
                key=f"tbm_cpt_lbl_{i}",
            )
            d = def_coords[i] if i < len(def_coords) else (0, 0, 0)
            _, ckx = numeric_expr_input(
                "kx (×π)", key=f"tbm_cpt_kx_{i}", default_expr=str(d[0]),
                min_v=-2.0, max_v=2.0, step=0.05, show_label=True)
            _, cky = numeric_expr_input(
                "ky (×π)", key=f"tbm_cpt_ky_{i}", default_expr=str(d[1]),
                min_v=-2.0, max_v=2.0, step=0.05, show_label=True)
            _, ckz = numeric_expr_input(
                "kz (×π)", key=f"tbm_cpt_kz_{i}", default_expr=str(d[2]),
                min_v=-2.0, max_v=2.0, step=0.05, show_label=True)
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
    color_mgr = _get_color_mgr()

    # ── Handle Scroll-to logic ──────────────────────────────────────────
    scroll_target = st.session_state.pop("scroll_to", None)
    if scroll_target:
        scroll_js = f"""
        <script>
        window.parent.document.getElementById('{scroll_target}').scrollIntoView({{behavior: 'smooth', block: 'start'}});
        </script>
        """
        st.components.v1.html(scroll_js, height=0)

    # ── Preset selector ──────────────────────────────────────────────
    col_pre, col_clr = st.columns([4, 1])
    with col_pre:
        st.selectbox("🚀 빠른 시작 (프리셋 로드)",
                     list(TBM_PRESETS.keys()), key="tbm_preset_sel")
                     
    def load_preset_callback():
        preset_choice = st.session_state["tbm_preset_sel"]
        fn = TBM_PRESETS[preset_choice]
        new_model = fn() if fn else TBMModel()
        _set_model(new_model)

        # 명시적으로 위젯 상태(Session State) 동기화
        st.session_state["tbm_spin_enabled"] = new_model.basis_config.spin_enabled
        st.session_state["tbm_lattice_dim"] = new_model.lattice.dimension
        for ci, comp in enumerate(["x", "y", "z"]):
            for vec in ("a1", "a2", "a3"):
                vec_expr = getattr(new_model.lattice, f"{vec}_expr")
                if ci < len(vec_expr):
                    reset_numeric_expr_input(
                        f"tbm_lat_{vec}_{comp}", str(vec_expr[ci]))
        # Reset parameter widgets so old preset's params don't leak in
        for k in list(st.session_state.keys()):
            if k.startswith("tbm_param_") and (
                k.endswith("__expr") or k.endswith("__slider")
                or k.endswith("__num") or k.endswith("__last_float")
            ):
                del st.session_state[k]

    with col_clr:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("📂 로드", key="tbm_load_preset", use_container_width=True, on_click=load_preset_callback)

    # ── Diagram + H(k) preview ────────────────────────────────────────
    # 배치 모드 선택: 물리 엔진(자유 배치) ↔ 실제 격자(고정 좌표)
    layout_mode = st.radio(
        "다이어그램 배치 모드",
        options=["physics", "real_space"],
        format_func=lambda x: "⚛️ 물리 엔진 (Physics)" if x == "physics" else "🔷 실제 격자 (Real Space)",
        horizontal=True,
        key="tbm_layout_mode",
        help="Physics: 노드가 자유롭게 이동합니다. Real Space: 격자 벡터 기준 실제 좌표에 노드를 고정합니다.",
    )

    diag_col, matrix_col = st.columns([6, 4])

    selected_hop = st.session_state.get("selected_hop_uid")

    with diag_col:
        st.markdown("#### TBM Diagram")
        html_data = generate_pyvis_html(
            model, color_mgr=color_mgr,
            selected_hop_uid=selected_hop,
            layout_mode=layout_mode,
        )
        
        if pyvis_interaction:
            clicked_element = pyvis_interaction(html=html_data, height=520, key="pyvis_diagram")
            if clicked_element:
                ts = clicked_element.get("ts", 0)
                if ts != st.session_state.get("last_pyvis_click_ts"):
                    st.session_state["last_pyvis_click_ts"] = ts
                    uid = clicked_element.get("id")
                    c_type = clicked_element.get("type")
                    
                    if c_type == "node":
                        if any(s.uid == uid for s in model.states):
                            st.session_state["selected_state_uid"] = uid
                            st.session_state["scroll_to"] = "edit-state"
                    elif c_type == "edge":
                        if any(h.uid == uid for h in model.hoppings):
                            st.session_state["selected_hop_uid"] = uid
                            st.session_state["scroll_to"] = "edit-hopping"
                    st.rerun()
        else:
            components.html(html_data, height=520)

    with matrix_col:
        st.markdown("#### $H(\\mathbf{k})$ — Color-coded Matrix")
        if model.matrix_dimension() == 0:
            st.info("State를 추가하면 행렬이 여기에 표시됩니다.")
        else:
            try:
                H_sym = model.build_hamiltonian_matrix()
                
                with st.expander("👀 전체 Hamiltonian 행렬 보기", expanded=False):
                    st.latex(sp.latex(sp.simplify(H_sym)))
                
                N = H_sym.shape[0]
                shown = 0
                for i in range(N):
                    for j in range(N):
                        elem = H_sym[i, j]
                        if elem != 0:
                            # 이 원소에 기여한 Hopping 추적
                            hop_uids = model.trace_element_to_hoppings(i, j)
                            
                            # 수식 내의 모든 실수(Number)를 소수점 3자리로 반올림
                            simplified_elem = sp.simplify(elem)
                            rounded_elem = simplified_elem.xreplace({
                                n: round(n, 3) for n in simplified_elem.atoms(sp.Number)
                            })
                            
                            latex_str = sp.latex(rounded_elem)
                            if hop_uids:
                                primary_uid = hop_uids[0]
                                clr = color_mgr.get_color(primary_uid)
                                st.markdown(
                                    f'<div style="border-left:3px solid {clr}; '
                                    f'padding-left:8px; margin:2px 0 6px 0;">',
                                    unsafe_allow_html=True)
                                st.latex(f"H_{{{i+1},{j+1}}} = {latex_str}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.latex(f"H_{{{i+1},{j+1}}} = {latex_str}")
                            shown += 1
                            if shown >= 12:
                                st.caption(f"... ({N*N - shown}개 항목 생략)")
                                break
                    if shown >= 12:
                        break
            except Exception as e:
                st.error(f"행렬 생성 오류: {e}")

        # ── Color legend ──────────────────────────────────────────────
        if model.hoppings:
            with st.expander("🎨 Hopping ↔ Color Legend", expanded=False):
                entries = []
                for hop in model.hoppings:
                    s = model.get_state(hop.source_uid)
                    t = model.get_state(hop.target_uid)
                    if s and t:
                        amp = hop.amplitude_matrix.strip() if hop.amplitude_matrix.strip() else hop.amplitude
                        entries.append((hop.uid, f"{s.label()}→{t.label()}", amp))
                st.markdown(color_mgr.build_legend_html(entries), unsafe_allow_html=True)

    # ── Select-to-edit Hopping form ────────────────────────────────────
    if model.hoppings:
        st.markdown('<div id="edit-hopping"></div>', unsafe_allow_html=True)
        st.markdown("### ✏️ Edit Hopping")
        hop_options = {}
        for h in model.hoppings:
            src = model.get_state(h.source_uid)
            tgt = model.get_state(h.target_uid)
            if src and tgt:
                lbl = f"{src.label()} → {tgt.label()} (R={h.R})"
                hop_options[h.uid] = lbl
                
        # 현재 선택된 Hopping 인덱스 찾기
        sel_uid = st.session_state.get("selected_hop_uid")
        current_idx = list(hop_options.keys()).index(sel_uid) if sel_uid in hop_options else 0
        
        sel_uid_from_selectbox = st.selectbox(
            "수정할 Hopping을 선택하세요:", 
            options=list(hop_options.keys()), 
            format_func=lambda uid: hop_options[uid],
            index=current_idx,
            key="tbm_edit_hop_selector"
        )
        
        # 선택이 변경되었으면 상태 업데이트 후 리런 (다이어그램 하이라이트 반영)
        if sel_uid_from_selectbox != sel_uid:
            st.session_state["selected_hop_uid"] = sel_uid_from_selectbox
            st.rerun()

        if sel_uid_from_selectbox:
            hop = model.get_hopping(sel_uid_from_selectbox)
            if hop:
                src_st = model.get_state(hop.source_uid)
                tgt_st = model.get_state(hop.target_uid)
                clr = color_mgr.get_color(sel_uid_from_selectbox)
                st.markdown(
                    f'<div style="background:rgba(99,102,241,0.1); border:1px solid {clr}; '
                    f'border-radius:10px; padding:12px; margin:8px 0;">'
                    f'<span style="color:{clr}; font-weight:700;">Selected</span> — '
                    f'{src_st.label() if src_st else "?"} → {tgt_st.label() if tgt_st else "?"}'
                    f'</div>', unsafe_allow_html=True)
                with st.form("edit_hop_form"):
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        new_amp = st.text_input("Amplitude (scalar)", value=hop.amplitude)
                    with ec2:
                        new_amp_m = st.text_input("Matrix Amplitude", value=hop.amplitude_matrix)
                    rc = st.columns(3)
                    new_r0 = int(rc[0].number_input("Rx", value=hop.R[0], step=1))
                    new_r1 = int(rc[1].number_input("Ry", value=hop.R[1], step=1))
                    new_r2 = int(rc[2].number_input("Rz", value=hop.R[2], step=1))
                    if st.form_submit_button("✅ 수정 적용", use_container_width=True):
                        hop.amplitude = new_amp
                        hop.amplitude_matrix = new_amp_m
                        hop.R = [new_r0, new_r1, new_r2]
                        st.rerun()

    # ── Select-to-edit Site form ──────────────────────────────────────
    if model.sites:
        st.markdown('<div id="edit-site"></div>', unsafe_allow_html=True)
        st.markdown("### 🔵 Edit Site")
        site_options = {s.uid: s.name for s in model.sites}
        sel_site_uid_prev = st.session_state.get("selected_site_uid")
        site_idx = (list(site_options.keys()).index(sel_site_uid_prev)
                    if sel_site_uid_prev in site_options else 0)
        sel_site_uid = st.selectbox(
            "수정할 Site를 선택하세요:",
            options=list(site_options.keys()),
            format_func=lambda uid: site_options[uid],
            index=site_idx,
            key="tbm_edit_site_selector",
        )
        if sel_site_uid != sel_site_uid_prev:
            st.session_state["selected_site_uid"] = sel_site_uid
            # Reset position widgets to the newly selected site's expressions
            site_obj = next((s for s in model.sites if s.uid == sel_site_uid), None)
            if site_obj is not None:
                for ci, comp in enumerate(["x", "y", "z"]):
                    reset_numeric_expr_input(
                        f"edit_site_pos_{comp}", str(site_obj.position_expr[ci]))
                st.session_state["edit_site_name"] = site_obj.name
            st.rerun()

        if sel_site_uid:
            site = next((s for s in model.sites if s.uid == sel_site_uid), None)
            if site is not None:
                # Make sure the name input has an initial value on first render
                if "edit_site_name" not in st.session_state:
                    st.session_state["edit_site_name"] = site.name

                ec1, ec2 = st.columns([1, 2])
                with ec1:
                    new_name = st.text_input("이름", key="edit_site_name")
                with ec2:
                    st.caption("Position (수식 가능)")
                    e_x, v_x = numeric_expr_input("x", key="edit_site_pos_x",
                                                   default_expr=str(site.position_expr[0]),
                                                   min_v=-2.0, max_v=2.0, step=0.05)
                    e_y, v_y = numeric_expr_input("y", key="edit_site_pos_y",
                                                   default_expr=str(site.position_expr[1]),
                                                   min_v=-2.0, max_v=2.0, step=0.05)
                    e_z, v_z = numeric_expr_input("z", key="edit_site_pos_z",
                                                   default_expr=str(site.position_expr[2]),
                                                   min_v=-2.0, max_v=2.0, step=0.05)
                if st.button("✅ Site 수정 적용", key="apply_edit_site",
                             use_container_width=True):
                    site.name = new_name
                    site.position = [v_x, v_y, v_z]
                    site.position_expr = [e_x, e_y, e_z]
                    st.rerun()

    # ── Select-to-edit State form ─────────────────────────────────────
    if model.states:
        st.markdown('<div id="edit-state"></div>', unsafe_allow_html=True)
        st.markdown("### ⚛️ Edit State")
        state_options = {s.uid: s.label() for s in model.states}
        sel_state_uid_prev = st.session_state.get("selected_state_uid")
        state_idx = (list(state_options.keys()).index(sel_state_uid_prev)
                     if sel_state_uid_prev in state_options else 0)
        sel_state_uid = st.selectbox(
            "수정할 State를 선택하세요:",
            options=list(state_options.keys()),
            format_func=lambda uid: state_options[uid],
            index=state_idx,
            key="tbm_edit_state_selector",
        )
        if sel_state_uid != sel_state_uid_prev:
            st.session_state["selected_state_uid"] = sel_state_uid
            st.rerun()

        if sel_state_uid:
            state = next((s for s in model.states if s.uid == sel_state_uid), None)
            if state is not None:
                site_opts = {s.name: s for s in model.sites}
                cur_site_idx = list(site_opts.keys()).index(state.site.name) \
                    if state.site.name in site_opts else 0
                # 현재 orbital이 표준 목록에 없으면 "custom" 인덱스로 fallback
                is_custom_orbital = state.orbital not in ORBITAL_OPTIONS
                cur_orb_idx = (ORBITAL_OPTIONS.index("custom")
                               if is_custom_orbital
                               else ORBITAL_OPTIONS.index(state.orbital))
                cur_spin_idx = SPIN_OPTIONS.index(state.spin) \
                    if state.spin in SPIN_OPTIONS else 0
                with st.form("edit_state_form"):
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        new_site_name = st.selectbox(
                            "Site", list(site_opts.keys()), index=cur_site_idx)
                    with sc2:
                        new_orbital = st.selectbox(
                            "Orbital", ORBITAL_OPTIONS, index=cur_orb_idx)
                    with sc3:
                        new_spin = st.selectbox(
                            "Spin", SPIN_OPTIONS, index=cur_spin_idx)
                    # "custom" 선택 시 텍스트 입력창 표시
                    if new_orbital == "custom":
                        # form 내부이므로 기존 custom 이름을 기본값으로 표시
                        init_custom = state.orbital if is_custom_orbital else ""
                        custom_orb_edit = st.text_input(
                            "Custom Orbital 이름",
                            value=init_custom,
                            placeholder="예: p_+, d_{x^2-y^2}, p_x + ip_y",
                            help="다이어그램 라벨과 행렬 표시에 그대로 사용됩니다.",
                        )
                        actual_new_orbital = custom_orb_edit.strip() or "custom"
                    else:
                        actual_new_orbital = new_orbital
                    if st.form_submit_button("✅ State 수정 적용",
                                              use_container_width=True):
                        state.site = site_opts[new_site_name]
                        state.orbital = actual_new_orbital
                        state.spin = new_spin
                        st.rerun()

    # ── Guard: need states ────────────────────────────────────────────
    if model.matrix_dimension() == 0:
        st.info("사이드바에서 Site → State → Hopping 순서로 추가하면 밴드 구조가 계산됩니다.")
        return

    # ── Build lambdified funcs ────────────────────────────────────────
    try:
        H_sym = model.build_hamiltonian_matrix()
        expr_matrix = [[H_sym[i, j] for j in range(H_sym.shape[0])]
                       for i in range(H_sym.shape[0])]
        k_reserved = set(K_SYMBOLS.values())
        raw_free = sorted(H_sym.free_symbols - k_reserved, key=lambda s: s.name)
    except Exception as e:
        st.error(f"해밀토니안 빌드 오류: {e}")
        return

    # ── Parameter sliders ─────────────────────────────────────────────
    param_values: dict[str, float] = {}
    if raw_free:
        st.markdown("### 🎛️ Parameter Tuning")
        st.caption("수식 입력 가능 — 예: `sqrt(3)`, `pi/4`, `2*sin(0.3)`")
        for sym in raw_free:
            pname = str(sym)
            _, val = numeric_expr_input(
                label=pname,
                key=f"tbm_param_{pname}",
                default_expr="1",
                min_v=-5.0, max_v=5.0, step=0.05,
                fmt="%.4f",
                show_label=True,
            )
            param_values[pname] = val

    # ── Build callable matrix functions ───────────────────────────────
    try:
        func_matrix = build_lambdified_matrix_funcs(expr_matrix, raw_free)
    except Exception as e:
        st.error(f"lambdify 오류: {e}")
        return

    # ── k-path ────────────────────────────────────────────────────────
    actual_k_path_type = k_path_type
    
    # K-path 자동화 시스템 (brillouin_zone.py)
    if actual_k_path_type == "auto":
        analyzer = BrillouinZoneAnalyzer(model.lattice)
        actual_k_path_type = analyzer.lattice_type
        st.info(f"💡 **Automated BZ Analyzer**\n{analyzer.summary()}")
        k_vals, k_pts, k_ticks, k_lbls = analyzer.get_auto_k_path(n_k)
    else:
        if actual_k_path_type == "square":
            k_vals, k_pts, k_ticks, k_lbls = get_k_path_square(n_k)
        elif actual_k_path_type == "hexagonal":
            k_vals, k_pts, k_ticks, k_lbls = get_k_path_hexagonal(n_k)
        elif actual_k_path_type == "custom" and len(custom_points) >= 2:
            k_vals, k_pts, k_ticks, k_lbls = get_k_path_custom(custom_points, n_k)
        else:
            k_vals, k_pts, k_ticks, k_lbls = get_k_path_1d(n_k)

    # ── Eigenvalues ───────────────────────────────────────────────────
    try:
        eigenvalues_1d, band_gap = compute_eigenvalues_1d_nxn(
            func_matrix, k_pts, param_values, raw_free)
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
    N_total = H_sym.shape[0]
    model_name = f"TBM ({N_total}×{N_total})"
    k_path_labels = {"square": "Square", "hexagonal": "Hexagonal", "1d": "1D", "custom": "Custom"}

    if show_2d:
        tab1, tab2 = st.tabs(["📈 1D Band Structure", "🌐 2D Energy Surface"])
    else:
        tab1 = st.container()
        tab2 = None

    with tab1:
        fig_1d = plot_1d_bands(k_vals, eigenvalues_1d, k_ticks, k_lbls,
                               title=f"Band Structure — {model_name} ({k_path_labels.get(k_path_type, '')})")
        st.plotly_chart(fig_1d, use_container_width=True, key="tbm_plot_1d")

    if show_2d and tab2 is not None:
        with tab2:
            try:
                kx_mesh, ky_mesh = get_k_grid_2d(n_k_2d)
                ev_2d, _ = compute_eigenvalues_2d_nxn(func_matrix, kx_mesh, ky_mesh, param_values, raw_free)
                fig_2d = plot_2d_surface(kx_mesh, ky_mesh, ev_2d,
                                         title=f"Energy Surface — {model_name}")
                st.plotly_chart(fig_2d, use_container_width=True, key="tbm_plot_2d")
            except Exception as e:
                st.error(f"2D 계산 오류: {e}")
