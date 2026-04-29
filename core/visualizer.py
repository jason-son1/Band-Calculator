"""
Plotly-based interactive visualizations for band structures.
"""

import numpy as np
import plotly.graph_objects as go


BAND_COLORS = {
    'lower': '#a855f7',
    'upper': '#06b6d4',
    'gap': 'rgba(250, 204, 21, 0.15)',
}

LAYOUT_DEFAULTS = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(14,17,23,1)',
    font=dict(family='Inter, sans-serif', size=13, color='#e2e8f0'),
    margin=dict(l=60, r=30, t=50, b=50),
)


def plot_1d_bands(k_values, eigenvalues, k_ticks, k_labels, title="1D Band Structure"):
    fig = go.Figure()

    N = eigenvalues.shape[1]
    import plotly.express as px
    # Generate distinct colors for N bands
    colors = px.colors.sample_colorscale("Rainbow", [i/(N-1) if N > 1 else 0.5 for i in range(N)])

    for i in range(N):
        fig.add_trace(go.Scatter(
            x=k_values, y=eigenvalues[:, i], mode='lines', name=f'E_{i}',
            line=dict(color=colors[i], width=2.5),
            hovertemplate='k = %{x:.3f}<br>E = %{y:.4f}<extra>E_' + str(i) + '</extra>',
        ))

    # Add a filled area for the middle gap if N is even
    if N % 2 == 0:
        mid = N // 2
        fig.add_trace(go.Scatter(
            x=np.concatenate([k_values, k_values[::-1]]),
            y=np.concatenate([eigenvalues[:, mid], eigenvalues[::-1, mid-1]]),
            fill='toself', fillcolor=BAND_COLORS['gap'],
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ))

    for tick in k_ticks:
        fig.add_vline(x=tick, line_dash="dot", line_color="rgba(148,163,184,0.4)", line_width=1)
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,0.3)", line_width=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="k-path", tickmode='array', tickvals=k_ticks, ticktext=k_labels, showgrid=False, zeroline=False),
        yaxis=dict(title="Energy E(k)", showgrid=True, gridcolor='rgba(148,163,184,0.1)', zeroline=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
        height=420,
    )
    return fig


def plot_2d_surface(kx_mesh, ky_mesh, eigenvalues_mesh, title="2D Energy Surface"):
    fig = go.Figure()

    N = len(eigenvalues_mesh)
    import plotly.express as px
    
    for i in range(N):
        # Create a monochromatic colorscale for each band based on Rainbow
        base_color = px.colors.sample_colorscale("Rainbow", i/(N-1) if N > 1 else 0.5)[0]
        # Very simple custom colorscale using the base color and a lighter version
        cs = [[0, 'rgba(14,17,23,0.8)'], [1, base_color]]
        
        fig.add_trace(go.Surface(
            x=kx_mesh, y=ky_mesh, z=eigenvalues_mesh[i], name=f'E_{i}',
            colorscale=cs, opacity=0.92, showscale=False,
            hovertemplate='kₓ=%{x:.2f}<br>kᵧ=%{y:.2f}<br>E=' + str(i) + '=%{z:.4f}<extra></extra>',
        ))

    axis_style = dict(backgroundcolor='rgba(14,17,23,0.9)', gridcolor='rgba(148,163,184,0.15)', showbackground=True)
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis=dict(title='kₓ', **axis_style),
            yaxis=dict(title='kᵧ', **axis_style),
            zaxis=dict(title='Energy E(k)', **axis_style),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        height=550,
    )
    return fig
