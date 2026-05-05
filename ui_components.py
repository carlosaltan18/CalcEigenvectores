import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import plotly.graph_objects as go


def render_matrix_input(n: int) -> Optional[np.ndarray]:
    """Render an n×n matrix input grid and return the matrix."""
    example = st.session_state.get('example_matrix', None)

    matrix = []
    cols_header = st.columns(n)
    for j, col in enumerate(cols_header):
        col.markdown(f"<center><b>col {j+1}</b></center>", unsafe_allow_html=True)

    for i in range(n):
        cols = st.columns(n)
        row = []
        for j, col in enumerate(cols):
            default_val = 0.0
            if example and i < len(example) and j < len(example[i]):
                default_val = float(example[i][j])
            val = col.number_input(
                f"a[{i+1},{j+1}]",
                value=default_val,
                step=1.0,
                format="%.2f",
                label_visibility="collapsed",
                key=f"cell_{n}_{i}_{j}"
            )
            row.append(val)
        matrix.append(row)

    return [[0 if v is None else v for v in row] for row in matrix]


def render_results(matrix, eigenvalues: np.ndarray, eigenvectors: np.ndarray, n: int):
    """Render eigenvalue/eigenvector results with verification."""
    from eigen_solver import EigenSolver

    A = np.array(matrix, dtype=float)
    solver = EigenSolver(matrix)

    st.markdown("### 📊 Resultados")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Matriz A:**")
        df_A = pd.DataFrame(A, columns=[f"c{j+1}" for j in range(n)],
                            index=[f"f{i+1}" for i in range(n)])
        st.dataframe(df_A.style.format("{:.2f}"), use_container_width=True)

    with col2:
        st.markdown("**Eigenvalores encontrados:**")
        for i, ev in enumerate(eigenvalues):
            st.markdown(f'<div class="result-card"><span class="eigen-value">λ_{i+1} = {ev:.6f}</span></div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧭 Eigenvectores")

    if len(eigenvectors.shape) == 2:
        evec_cols = st.columns(len(eigenvalues))
        for i, (ev, col) in enumerate(zip(eigenvalues, evec_cols)):
            with col:
                st.markdown(f"**λ_{i+1} = {ev:.4f}**")
                vec = eigenvectors[:, i]
                df_vec = pd.DataFrame(vec, columns=["v"], index=[f"v{k+1}" for k in range(len(vec))])
                st.dataframe(df_vec.style.format("{:.6f}"), use_container_width=True)
    else:
        st.warning("No se pudieron calcular eigenvectores.")

    st.markdown("---")
    st.markdown("### ✅ Verificación: A·v = λ·v")

    verifications = solver.verify(eigenvalues, eigenvectors)
    for i, ver in enumerate(verifications):
        with st.expander(f"λ_{i+1} = {ver['eigenvalue']:.6f} {'✅' if ver['verified'] else '⚠️'}"):
            vcol1, vcol2, vcol3 = st.columns(3)
            with vcol1:
                st.markdown("**A · v:**")
                for k, val in enumerate(ver['Av']):
                    st.write(f"  {val:.6f}")
            with vcol2:
                st.markdown("**λ · v:**")
                for k, val in enumerate(ver['lambda_v']):
                    st.write(f"  {val:.6f}")
            with vcol3:
                st.markdown("**Residuo:**")
                st.metric("||Av - λv||", f"{ver['residual']:.2e}")
                if ver['verified']:
                    st.success("Verificado ✓")
                else:
                    st.warning("Residuo alto")

    # Visualization for 2x2 matrices
    if n == 2 and len(eigenvalues) >= 1:
        st.markdown("---")
        st.markdown("### 📈 Visualización (2D)")
        _plot_eigenvectors_2d(eigenvalues, eigenvectors)


def _plot_eigenvectors_2d(eigenvalues, eigenvectors):
    """Plot eigenvectors in 2D space."""
    colors = ['#4a6cf7', '#f74a6c', '#4af7a0', '#f7c44a']
    fig = go.Figure()

    # Origin
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
                             marker=dict(size=10, color='black'), name='Origen'))

    if len(eigenvectors.shape) == 2:
        for i in range(min(len(eigenvalues), eigenvectors.shape[1])):
            vec = eigenvectors[:, i]
            scale = 2.0
            color = colors[i % len(colors)]
            lam = eigenvalues[i]

            fig.add_annotation(
                x=vec[0] * scale, y=vec[1] * scale,
                ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.5,
                arrowwidth=3,
                arrowcolor=color
            )
            fig.add_trace(go.Scatter(
                x=[vec[0] * scale], y=[vec[1] * scale],
                mode='markers+text',
                text=[f"v_{i+1} (λ={lam:.2f})"],
                textposition="top center",
                marker=dict(size=8, color=color),
                name=f"v_{i+1}",
                showlegend=True
            ))

    fig.update_layout(
        title="Eigenvectores en el plano 2D",
        xaxis_title="x₁", yaxis_title="x₂",
        xaxis=dict(zeroline=True, range=[-3, 3]),
        yaxis=dict(zeroline=True, range=[-3, 3], scaleanchor="x", scaleratio=1),
        height=400,
        plot_bgcolor='#f8f9fa'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_steps(steps: List[Dict]):
    """Render algorithm steps."""
    st.markdown("### 🔍 Pasos del Algoritmo")

    for step in steps:
        step_num = step.get("step", "?")
        title = step.get("title", "")
        description = step.get("description", "")
        formula = step.get("formula", "")
        latex_str = step.get("latex", "")
        found = step.get("found", None)

        icon = "✅" if found is True else ("⚠️" if found is False else "🔹")
        with st.expander(f"{icon} Paso {step_num}: {title}", expanded=True):
            if description:
                st.write(description)
            if latex_str:
                st.latex(latex_str)
            elif formula:
                st.code(formula, language=None)
            if step.get("matrix_display") and step.get("matrix"):
                try:
                    import sympy
                    st.latex(sympy.latex(step["matrix"]))
                except Exception:
                    st.code(str(step["matrix"]))