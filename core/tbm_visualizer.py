"""
TBM Diagram Visualizer — PyVis 기반 유동적 다이어그램 (Phase 4).

Converts a TBMModel into an interactive, physics-based, draggable network graph
using PyVis. This replaces the static Plotly implementation to solve edge overlaps
and provide fluid interactions.

Node = State (Site + Orbital + Spin)
Edge = Hopping term
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network

from core.tbm_model import TBMModel
from core.color_manager import ColorManager, ONSITE_COLOR

# ── Fallback site colors (노드용) ─────────────────────────────────────
SITE_COLORS = [
    "#06b6d4", "#a855f7", "#f59e0b", "#10b981",
    "#f43f5e", "#3b82f6", "#84cc16", "#ec4899",
]

def generate_pyvis_html(
    model: TBMModel,
    color_mgr: Optional[ColorManager] = None,
    selected_hop_uid: Optional[str] = None,
    layout_mode: str = "physics",
) -> str:
    """
    Render the TBM model as a PyVis interactive HTML string.

    Args:
        model: TBM 모델
        color_mgr: Hopping 색상 관리자
        selected_hop_uid: 강조 표시할 Hopping UID
        layout_mode: "physics" (물리 엔진) | "real_space" (실제 격자 좌표 고정)
    """
    if color_mgr is None:
        color_mgr = ColorManager()

    use_real_space = (layout_mode == "real_space")

    # PyVis Network 인스턴스 생성 (다크 테마 호환)
    net = Network(
        height="500px",
        width="100%",
        bgcolor="#0e1117",
        font_color="#e2e8f0",
        directed=True
    )

    # ── Physics & Edge Curve Options ─────────────────────────────────
    # real_space 모드: physics 비활성화 → 노드 좌표 고정
    physics_block = (
        '"physics": { "enabled": false }'
        if use_real_space
        else """
        "physics": {
          "barnesHut": {
            "gravitationalConstant": -4000,
            "centralGravity": 0.1,
            "springLength": 200,
            "springConstant": 0.02,
            "damping": 0.09,
            "avoidOverlap": 0.3
          },
          "minVelocity": 0.75
        }"""
    )

    net.set_options(f"""
    var options = {{
      "nodes": {{
        "shape": "dot",
        "size": 20,
        "font": {{
          "size": 14,
          "color": "#e2e8f0",
          "face": "JetBrains Mono, monospace"
        }},
        "borderWidth": 2
      }},
      "edges": {{
        "smooth": {{
          "type": "dynamic",
          "roundness": 0.5
        }},
        "font": {{
          "size": 11,
          "color": "#94a3b8",
          "background": "#0e1117",
          "strokeWidth": 0
        }},
        "arrows": {{
          "to": {{ "enabled": true, "scaleFactor": 0.8 }}
        }}
      }},
      {physics_block},
      "interaction": {{
        "hover": true,
        "tooltipDelay": 200
      }}
    }}
    """)

    # ── Site → color mapping ──────────────────────────────
    site_color_map: dict[str, str] = {}
    for i, site in enumerate(model.sites):
        site_color_map[site.uid] = SITE_COLORS[i % len(SITE_COLORS)]

    # ── Add Nodes ────────────────────────────────────────────────────
    # Orbital → Shape 매핑 (custom/unknown은 triangle)
    shape_map = {
        "s": "dot",
        "px": "diamond", "py": "diamond", "pz": "diamond",
        "dxy": "square", "dxz": "square", "dyz": "square", "dx2y2": "square", "dz2": "square",
    }

    # Real Space 모드용 격자 벡터 (2D 기준)
    a1 = model.lattice.a1  # [x, y, z]
    a2 = model.lattice.a2  # [x, y, z]
    REAL_SPACE_SCALE = 250  # 픽셀 단위 스케일 팩터

    spin_enabled = model.basis_config.spin_enabled
    for state in model.states:
        color = site_color_map.get(state.site.uid, "#94a3b8")
        lbl = state.label()

        # Spin ⊗ 라벨
        if spin_enabled and state.spin == "none":
            lbl += " ⊗σ"

        # Spin에 따른 테두리 스타일 지정
        border_width = 2
        border_dashes = False
        if state.spin == "↑":
            border_width = 4
            border_dashes = False
        elif state.spin == "↓":
            border_width = 4
            border_dashes = True  # 파선 테두리

        # Orbital 모양 지정 (custom/unknown → triangle)
        shape = shape_map.get(state.orbital, "triangle")

        hover_text = (
            f"<b>{lbl}</b><br>"
            f"Site: {state.site.name}<br>"
            f"Orbital: {state.orbital}<br>"
            f"Spin: {state.spin if state.spin != 'none' else '↑↓ (tensor)'}"
        )

        node_kwargs = dict(
            label=lbl,
            title=hover_text,
            shape=shape,
            borderWidth=border_width,
            shapeProperties={"borderDashes": border_dashes},
            color={"background": color, "border": "#ffffff",
                   "highlight": {"background": color, "border": "#ffeb3b"}},
        )

        if use_real_space:
            # 분수 좌표 → 2D 데카르트 좌표: r = c1*a1 + c2*a2
            pos = state.site.position  # [c1, c2, c3]
            x_cart = (pos[0] * a1[0] + pos[1] * a2[0]) * REAL_SPACE_SCALE
            # PyVis y축은 아래가 양수이므로 부호 반전하여 물리적 좌표계와 일치
            y_cart = -(pos[0] * a1[1] + pos[1] * a2[1]) * REAL_SPACE_SCALE
            node_kwargs["x"] = x_cart
            node_kwargs["y"] = y_cart
            node_kwargs["physics"] = False

        net.add_node(state.uid, **node_kwargs)

    # ── Add Edges ────────────────────────────────────────────────────
    for hop in model.hoppings:
        # 노드가 삭제된 경우 예외 처리
        if model.get_state(hop.source_uid) is None or model.get_state(hop.target_uid) is None:
            continue

        edge_color = color_mgr.assign_color(hop.uid, is_onsite=hop.is_onsite())
        is_selected = (hop.uid == selected_hop_uid)

        amp_display = hop.amplitude_matrix.strip() if hop.amplitude_matrix.strip() else hop.amplitude
        R_str = f"R={hop.R}"
        edge_label = hop.label if hop.label else f"t={amp_display}"

        # Inter-cell 표시는 파선으로 (PyVis dash 지원)
        dashes = True if hop.is_intercell() else False

        # 선택된 Edge는 두껍게
        width = 4 if is_selected else (2 if hop.is_intercell() else 1)

        hover_text = f"<b>{edge_label}</b><br>{R_str}<br>Amplitude: {amp_display}"

        net.add_edge(
            hop.source_uid,
            hop.target_uid,
            label=f"{edge_label} {R_str}",
            title=hover_text,
            color={"color": edge_color, "highlight": edge_color},
            width=width,
            dashes=dashes,
        )

    # 모델이 비어있으면 안내 메시지 노드 하나 추가
    if len(model.states) == 0:
        net.add_node("empty", label="모델이 비어있습니다.\n사이드바에서 Site와 State를 추가하세요.", color="#333333", shape="box")

    # 임시 HTML 생성
    html_data = net.generate_html()

    injection_script = """
    <script>
    window.addEventListener("load", function() {
        setTimeout(() => {
            if (typeof network !== 'undefined') {
                network.on("doubleClick", function(params) {
                    if (params.nodes.length > 0) {
                        window.parent.postMessage({type: "pyvis_double_click", payload: {type: "node", id: params.nodes[0], ts: Date.now()}}, "*");
                    } else if (params.edges.length > 0) {
                        window.parent.postMessage({type: "pyvis_double_click", payload: {type: "edge", id: params.edges[0], ts: Date.now()}}, "*");
                    }
                });
            }
        }, 500);
    });
    </script>
    """
    html_data = html_data.replace("</body>", injection_script + "\n</body>")

    return html_data


# ══════════════════════════════════════════════════════════════════════
# Real Space Lattice 시각화 (Plotly)
# ══════════════════════════════════════════════════════════════════════

# Plotly marker 심볼 매핑 (Orbital → shape)
_ORBITAL_SYMBOL: dict[str, str] = {
    "s":    "circle",
    "px":   "diamond", "py": "diamond", "pz": "diamond",
    "dxy":  "square",  "dxz": "square", "dyz": "square",
    "dx2y2":"square",  "dz2": "square",
}
# Spin → 마커 스타일 (open = hollow 테두리만, solid = 채워진)
_SPIN_OPEN: dict[str, bool] = {"↑": False, "↓": True, "none": False}


def generate_real_space_figure(
    model: TBMModel,
    n_cells: int = 2,
    site_color_map: Optional[dict] = None,
) -> go.Figure:
    """
    실제 격자 구조를 Plotly 산점도로 렌더링.

    원점 단위셀을 크고 선명하게, 주변 n_cells 주기의 이미지 셀을 반투명하게 표시합니다.
    격자 벡터 화살표와 단위셀 경계선을 함께 그립니다.

    Args:
        model:          TBM 모델
        n_cells:        원점에서 ±n_cells 방향으로 표시 (기본 2)
        site_color_map: {site_uid: hex_color} — 미제공 시 SITE_COLORS로 자동 생성
    """
    if site_color_map is None:
        site_color_map = {
            site.uid: SITE_COLORS[i % len(SITE_COLORS)]
            for i, site in enumerate(model.sites)
        }

    a1 = np.array(model.lattice.a1[:2], dtype=float)
    a2 = np.array(model.lattice.a2[:2], dtype=float)
    dim = model.lattice.dimension

    # 1D 모델이면 a2를 순수 수직 보조벡터로 처리 (시각화 전용)
    if dim == 1:
        a2_vis = np.array([0.0, 1.0])
    else:
        a2_vis = a2

    # ── 같은 Site에 있는 State들 간 소규모 오프셋 계산 ─────────────────
    site_state_count: dict[str, int] = {}
    state_site_idx:   dict[str, int] = {}
    for state in model.states:
        sid = state.site.uid
        idx = site_state_count.get(sid, 0)
        state_site_idx[state.uid] = idx
        site_state_count[sid] = idx + 1

    a1_len = float(np.linalg.norm(a1)) or 1.0
    a2v_len = float(np.linalg.norm(a2_vis)) or 1.0
    offset_r = min(a1_len, a2v_len) * 0.13  # 같은 Site 내 State 분리 반경

    fig = go.Figure()

    # ── 각 State를 격자 전체에 그리기 ────────────────────────────────
    r1_range = range(-n_cells, n_cells + 1)
    r2_range = range(-n_cells, n_cells + 1) if dim >= 2 else range(0, 1)

    for state in model.states:
        color  = site_color_map.get(state.site.uid, "#94a3b8")
        base_sym = _ORBITAL_SYMBOL.get(state.orbital, "triangle-up")
        is_open  = _SPIN_OPEN.get(state.spin, False)
        marker_sym = (base_sym + "-open") if is_open else base_sym

        # 같은 Site 내 상태들을 살짝 오프셋
        n_same = site_state_count.get(state.site.uid, 1)
        idx    = state_site_idx[state.uid]
        if n_same > 1:
            angle  = 2.0 * math.pi * idx / n_same
            dx_off = offset_r * math.cos(angle)
            dy_off = offset_r * math.sin(angle)
        else:
            dx_off = dy_off = 0.0

        origin_x, origin_y, origin_tip = [], [], []
        other_x,  other_y,  other_tip  = [], [], []

        pos = state.site.position
        for n1 in r1_range:
            for n2 in r2_range:
                r = (n1 + pos[0]) * a1 + (n2 + pos[1]) * a2_vis
                x = float(r[0]) + dx_off
                y = float(r[1]) + dy_off
                tip = (
                    f"<b>{state.label()}</b><br>"
                    f"Site: {state.site.name}<br>"
                    f"Orbital: {state.orbital}<br>"
                    f"Spin: {state.spin}<br>"
                    f"Cell R=({n1},{n2 if dim >= 2 else 0})<br>"
                    f"({x:.3f}, {y:.3f})"
                )
                if n1 == 0 and n2 == 0:
                    origin_x.append(x); origin_y.append(y); origin_tip.append(tip)
                else:
                    other_x.append(x);  other_y.append(y);  other_tip.append(tip)

        legend_name = state.label()

        # 주변 이미지 셀 (반투명, 작게)
        if other_x:
            fig.add_trace(go.Scatter(
                x=other_x, y=other_y, mode="markers",
                marker=dict(
                    symbol=marker_sym, size=9, color=color, opacity=0.28,
                    line=dict(width=1.2, color="rgba(255,255,255,0.35)"),
                ),
                hovertext=other_tip, hoverinfo="text",
                name=legend_name, legendgroup=legend_name, showlegend=False,
            ))

        # 원점 단위셀 (크고 선명하게 + 레이블)
        if origin_x:
            fig.add_trace(go.Scatter(
                x=origin_x, y=origin_y, mode="markers+text",
                marker=dict(
                    symbol=marker_sym, size=18, color=color, opacity=1.0,
                    line=dict(width=2.5, color="white"),
                ),
                text=[state.label()] * len(origin_x),
                textposition="top center",
                textfont=dict(size=10, color="#e2e8f0"),
                hovertext=origin_tip, hoverinfo="text",
                name=legend_name, legendgroup=legend_name,
            ))

    # ── 단위셀 경계 평행사변형 ─────────────────────────────────────
    uc_x = [0.0, a1[0], a1[0]+a2_vis[0], a2_vis[0], 0.0]
    uc_y = [0.0, a1[1], a1[1]+a2_vis[1], a2_vis[1], 0.0]
    fig.add_trace(go.Scatter(
        x=uc_x, y=uc_y, mode="lines",
        line=dict(color="rgba(255,255,255,0.22)", width=1.2, dash="dash"),
        hoverinfo="skip", showlegend=False, name="Unit Cell",
    ))

    # ── 격자 벡터 화살표 (annotation) ─────────────────────────────
    _arrow_cfg = dict(
        xref="x", yref="y", axref="x", ayref="y",
        ax=0.0, ay=0.0,
        showarrow=True, arrowhead=3, arrowsize=1.6, arrowwidth=2.5,
    )
    fig.add_annotation(**_arrow_cfg,
        x=float(a1[0]), y=float(a1[1]),
        arrowcolor="#f59e0b",
        text="<b>a₁</b>", font=dict(size=13, color="#f59e0b"),
        bgcolor="rgba(0,0,0,0)", borderpad=2,
    )
    if dim >= 2:
        fig.add_annotation(**_arrow_cfg,
            x=float(a2[0]), y=float(a2[1]),
            arrowcolor="#10b981",
            text="<b>a₂</b>", font=dict(size=13, color="#10b981"),
            bgcolor="rgba(0,0,0,0)", borderpad=2,
        )

    # ── 격자 점선 배경 그리드 (이미지 셀 경계) ────────────────────
    # 각 격자 꼭짓점을 십자(+) 모양으로 표시
    grid_x, grid_y = [], []
    for n1 in r1_range:
        for n2 in r2_range:
            r = n1 * a1 + n2 * a2_vis
            grid_x.append(float(r[0]))
            grid_y.append(float(r[1]))
    fig.add_trace(go.Scatter(
        x=grid_x, y=grid_y, mode="markers",
        marker=dict(symbol="cross-thin", size=6, color="rgba(255,255,255,0.12)",
                    line=dict(width=1, color="rgba(255,255,255,0.12)")),
        hoverinfo="skip", showlegend=False, name="lattice pts",
    ))

    # ── 레이아웃 ──────────────────────────────────────────────────
    total_cells = (2 * n_cells + 1) ** (min(dim, 2))
    fig.update_layout(
        title=dict(
            text=f"Real Space Lattice  —  {2*n_cells+1}×{2*n_cells+1 if dim >= 2 else 1} 단위셀  ({total_cells}개)",
            font=dict(size=13, color="#94a3b8"),
        ),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e2e8f0", family="monospace"),
        xaxis=dict(
            title="x  (a.u.)", color="#e2e8f0",
            showgrid=True, gridcolor="rgba(255,255,255,0.07)",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.18)",
        ),
        yaxis=dict(
            title="y  (a.u.)", color="#e2e8f0",
            showgrid=True, gridcolor="rgba(255,255,255,0.07)",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.18)",
            scaleanchor="x", scaleratio=1,   # 등방 비율
        ),
        legend=dict(
            bgcolor="rgba(14,17,23,0.85)",
            bordercolor="rgba(255,255,255,0.18)", borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=20, t=50, b=50),
        height=520,
        hovermode="closest",
    )

    return fig
