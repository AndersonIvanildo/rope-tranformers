"""Gráficos compartilhados pelo app e pelo notebook."""

from html import escape

import numpy as np
import plotly.graph_objects as go

from rope import METHODS, embeddings, projections, rotate, sinusoidal

COLORS = ("#64748B", "#D97706", "#0F766E")


def rotation_chart(token: str, dim: int, pair: int, positions: np.ndarray) -> go.Figure:
    x = embeddings([token], dim)[0]
    wq = projections(dim)[0]
    path = np.linspace(positions[0], positions[-1], max(2, len(positions) * 20))
    fig = go.Figure()
    for method, color in zip(METHODS, COLORS):
        def query_at(p):
            repeated = np.tile(x, (len(p), 1))
            if method == "Fixa senoidal":
                return (repeated + sinusoidal(p, dim)) @ wq
            q = repeated @ wq
            return rotate(q, p) if method == "RoPE" else q

        curve, points = query_at(path), query_at(positions)
        a, b = 2 * pair, 2 * pair + 1
        fig.add_trace(go.Scatter3d(
            x=curve[:, a], y=curve[:, b], z=path, mode="lines",
            line={"color": color, "width": 5}, name=method,
            hoverinfo="skip", legendgroup=method,
        ))
        fig.add_trace(go.Scatter3d(
            x=points[:, a], y=points[:, b], z=positions, mode="markers",
            marker={"color": color, "size": 4}, name=method,
            showlegend=False, legendgroup=method,
            hovertemplate=f"{method}<br>posição %{{z:.0f}}<br>q{a}=%{{x:.4f}}<br>q{b}=%{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        height=480, margin={"l": 0, "r": 0, "b": 0, "t": 20},
        legend={"orientation": "h"},
        scene={"xaxis_title": f"q[{2 * pair}]", "yaxis_title": f"q[{2 * pair + 1}]",
               "zaxis_title": "Posição", "aspectmode": "cube"},
        uirevision=f"{dim}-{pair}",
    )
    return fig


def attention_chart(weights: np.ndarray, labels: list[str]) -> go.Figure:
    safe_labels = [escape(label) for label in labels]
    fig = go.Figure(go.Heatmap(
        z=weights, x=safe_labels, y=safe_labels,
        zmin=0, zmax=1, colorscale="Teal", showscale=False,
        hovertemplate="Consulta: %{y}<br>Chave: %{x}<br>Atenção: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=370, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis={"title": "Chave (K)", "tickangle": -45, "nticks": 12},
        yaxis={"title": "Consulta (Q)", "autorange": "reversed", "nticks": 12},
    )
    return fig
