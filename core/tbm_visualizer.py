"""
TBM Diagram Visualizer.

Converts a TBMModel into a NetworkX directed graph and renders it
as an interactive Plotly figure inside Streamlit.

Node = State (Site + Orbital + Spin)
Edge = Hopping term

Edge style encoding:
  - On-site (i==i, R=0)  : red marker (loop, shown as self-arrow)
  - Intra-cell (R=0)     : solid gray arrow
  - Inter-cell (R≠0)     : dashed blue arrow
"""

from __future__ import annotations

import math
from typing import Optional

import networkx as nx
import plotly.graph_objects as go

from core.tbm_model import TBMModel, Hopping, State

# ── Color palette ─────────────────────────────────────────────────────
SITE_COLORS = [
    "#06b6d4",  # cyan
    "#a855f7",  # purple
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#f43f5e",  # rose
    "#3b82f6",  # blue
    "#84cc16",  # lime
    "#ec4899",  # pink
]

EDGE_COLORS = {
    "onsite":    "#f43f5e",   # red
    "intracell": "#94a3b8",   # slate
    "intercell": "#06b6d4",   # cyan
}


# ══════════════════════════════════════════════════════════════════════
# Graph Builder
# ══════════════════════════════════════════════════════════════════════

def build_nx_graph(model: TBMModel) -> nx.DiGraph:
    """
    Convert a TBMModel into a NetworkX directed graph.

    Node attributes  : label, site_name, orbital, spin, site_uid, color
    Edge attributes  : amplitude_str, R, hop_type, hop_uid, label
    """
    G = nx.DiGraph()

    # ── Site → color mapping ──────────────────────────────────────────
    site_color_map: dict[str, str] = {}
    for i, site in enumerate(model.sites):
        site_color_map[site.uid] = SITE_COLORS[i % len(SITE_COLORS)]

    # ── Add nodes ─────────────────────────────────────────────────────
    for state in model.states:
        color = site_color_map.get(state.site.uid, "#94a3b8")
        G.add_node(
            state.uid,
            label=state.label(),
            site_name=state.site.name,
            orbital=state.orbital,
            spin=state.spin,
            site_uid=state.site.uid,
            color=color,
        )

    # ── Add edges ─────────────────────────────────────────────────────
    for hop in model.hoppings:
        if hop.source_uid not in G or hop.target_uid not in G:
            continue

        hop_type = (
            "onsite"    if hop.is_onsite() else
            "intracell" if hop.is_intracell() else
            "intercell"
        )
        R_str = f"R={hop.R}"
        edge_label = f"t={hop.amplitude}" if not hop.label else hop.label

        G.add_edge(
            hop.source_uid,
            hop.target_uid,
            amplitude_str=hop.amplitude,
            R=hop.R,
            hop_type=hop_type,
            hop_uid=hop.uid,
            label=f"{edge_label} {R_str}",
        )

    return G


# ══════════════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════════════

def _compute_layout(G: nx.DiGraph, model: TBMModel) -> dict[str, tuple[float, float]]:
    """
    Compute 2D node positions.

    Strategy: group nodes by Site, arrange Sites in a circle,
    arrange States within each Site in a smaller inner circle.
    """
    if len(G.nodes) == 0:
        return {}

    # Group states by site
    site_groups: dict[str, list[str]] = {}
    for uid, data in G.nodes(data=True):
        site_uid = data.get("site_uid", "")
        site_groups.setdefault(site_uid, []).append(uid)

    n_sites = max(len(site_groups), 1)
    outer_r = 2.0
    inner_r = 0.4

    pos: dict[str, tuple[float, float]] = {}
    for s_idx, (site_uid, state_uids) in enumerate(site_groups.items()):
        # Position of this Site center on the outer circle
        theta_site = 2 * math.pi * s_idx / n_sites
        cx = outer_r * math.cos(theta_site)
        cy = outer_r * math.sin(theta_site)

        n_states = max(len(state_uids), 1)
        for st_idx, uid in enumerate(state_uids):
            theta_st = 2 * math.pi * st_idx / n_states
            x = cx + inner_r * math.cos(theta_st)
            y = cy + inner_r * math.sin(theta_st)
            pos[uid] = (x, y)

    return pos


# ══════════════════════════════════════════════════════════════════════
# Plotly Figure Renderer
# ══════════════════════════════════════════════════════════════════════

def plot_tbm_diagram(model: TBMModel, title: str = "TBM Diagram") -> go.Figure:
    """
    Render the TBM model as an interactive Plotly directed-graph figure.

    Returns a go.Figure ready to be passed to st.plotly_chart().
    """
    G = build_nx_graph(model)
    pos = _compute_layout(G, model)

    fig = go.Figure()

    if len(G.nodes) == 0:
        # Empty model — show placeholder
        fig.add_annotation(
            text="모델이 비어있습니다.<br>사이드바에서 Site와 State를 추가하세요.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#94a3b8"),
        )
        _apply_layout(fig, title)
        return fig

    # ── Draw edges ────────────────────────────────────────────────────
    for src_uid, tgt_uid, data in G.edges(data=True):
        hop_type = data.get("hop_type", "intracell")
        color = EDGE_COLORS.get(hop_type, "#94a3b8")

        x0, y0 = pos.get(src_uid, (0, 0))
        x1, y1 = pos.get(tgt_uid, (0, 0))

        if src_uid == tgt_uid:
            # Self-loop: draw a small circle offset from the node
            _add_self_loop(fig, x0, y0, color, data.get("label", ""))
        else:
            # Slight curve by offsetting the midpoint
            _add_arrow_edge(fig, x0, y0, x1, y1, color, hop_type, data.get("label", ""))

    # ── Draw nodes ────────────────────────────────────────────────────
    node_x, node_y, node_text, node_colors, node_labels = [], [], [], [], []
    for uid, data in G.nodes(data=True):
        x, y = pos.get(uid, (0, 0))
        node_x.append(x)
        node_y.append(y)
        node_colors.append(data.get("color", "#94a3b8"))
        node_labels.append(data.get("label", uid[:6]))
        node_text.append(
            f"<b>{data.get('label', uid)}</b><br>"
            f"Site: {data.get('site_name', '')}<br>"
            f"Orbital: {data.get('orbital', '')}<br>"
            f"Spin: {data.get('spin', '')}"
        )

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=36,
            color=node_colors,
            line=dict(width=2, color="#1e293b"),
            opacity=0.95,
        ),
        text=node_labels,
        textposition="middle center",
        textfont=dict(size=10, color="#0f172a", family="JetBrains Mono, monospace"),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    ))

    # ── Legend ────────────────────────────────────────────────────────
    for hop_type, color in EDGE_COLORS.items():
        labels = {
            "onsite":    "On-site (R=0, i=j)",
            "intracell": "Intra-cell (R=0)",
            "intercell": "Inter-cell (R≠0)",
        }
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=color, width=2,
                      dash="dash" if hop_type == "intercell" else "solid"),
            name=labels[hop_type],
            showlegend=True,
        ))

    _apply_layout(fig, title)
    return fig


# ── Internal helpers ──────────────────────────────────────────────────

def _add_arrow_edge(
    fig: go.Figure,
    x0: float, y0: float,
    x1: float, y1: float,
    color: str,
    hop_type: str,
    label: str,
):
    """Draw a directed edge as a line with an arrowhead annotation."""
    dash = "dash" if hop_type == "intercell" else "solid"
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2

    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1],
        mode="lines",
        line=dict(color=color, width=1.8, dash=dash),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Arrowhead via annotation
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length > 0:
        # Place arrow slightly before the target node
        offset = 0.22
        ax = x1 - (dx / length) * offset
        ay = y1 - (dy / length) * offset
        fig.add_annotation(
            x=x1, y=y1,
            ax=ax, ay=ay,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=1.8,
            arrowcolor=color,
        )

    # Hover label at midpoint
    fig.add_trace(go.Scatter(
        x=[mid_x], y=[mid_y],
        mode="markers",
        marker=dict(size=8, color=color, opacity=0.0),
        hovertext=label,
        hoverinfo="text",
        showlegend=False,
    ))


def _add_self_loop(
    fig: go.Figure,
    cx: float, cy: float,
    color: str,
    label: str,
    r: float = 0.3,
):
    """Draw a small circle above the node to represent an on-site term."""
    n = 40
    thetas = [2 * math.pi * i / n for i in range(n + 1)]
    lx = [cx + r * math.cos(t) for t in thetas]
    ly = [cy + r + r * math.sin(t) for t in thetas]

    fig.add_trace(go.Scatter(
        x=lx, y=ly,
        mode="lines",
        line=dict(color=color, width=2, dash="dot"),
        hovertext=f"On-site: {label}",
        hoverinfo="text",
        showlegend=False,
    ))


def _apply_layout(fig: go.Figure, title: str):
    """Apply consistent dark-theme layout to the TBM diagram."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        paper_bgcolor="rgba(14,17,23,0)",
        plot_bgcolor="rgba(14,17,23,0.6)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-3.5, 3.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-3.5, 3.5], scaleanchor="x"),
        legend=dict(
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor="rgba(99,102,241,0.3)",
            borderwidth=1,
            font=dict(color="#94a3b8", size=11),
            orientation="h",
            yanchor="bottom", y=1.01, xanchor="right", x=1,
        ),
        margin=dict(l=10, r=10, t=45, b=10),
        height=450,
    )
