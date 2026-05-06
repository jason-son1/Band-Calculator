"""
TBM Diagram Visualizer — PyVis 기반 유동적 다이어그램 (Phase 4).

Converts a TBMModel into an interactive, physics-based, draggable network graph
using PyVis. This replaces the static Plotly implementation to solve edge overlaps
and provide fluid interactions.

Node = State (Site + Orbital + Spin)
Edge = Hopping term
"""

from __future__ import annotations

from typing import Optional
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
