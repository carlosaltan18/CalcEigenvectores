"""Componentes de interfaz para el laboratorio de álgebra lineal."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def format_number(value: complex | float, precision: int = 4) -> str:
    """Formatea escalares reales o complejos de forma legible."""
    number = complex(value)
    real = 0.0 if abs(number.real) < 1e-10 else number.real
    imaginary = 0.0 if abs(number.imag) < 1e-10 else number.imag
    if imaginary == 0:
        return f"{real:.{precision}f}"
    sign = "+" if imaginary >= 0 else "−"
    return f"{real:.{precision}f} {sign} {abs(imaginary):.{precision}f}i"


def matrix_cell_key(size: int, row: int, column: int) -> str:
    return f"matrix_{size}_{row}_{column}"


def load_matrix(matrix: Iterable[Iterable[float]]) -> None:
    """Carga un ejemplo en el estado de Streamlit antes de renderizar los inputs."""
    values = np.asarray(matrix, dtype=float)
    size = values.shape[0]
    for row in range(size):
        for column in range(size):
            st.session_state[matrix_cell_key(size, row, column)] = float(values[row, column])


def render_matrix_input(size: int) -> List[List[float]]:
    """Renderiza una matriz cuadrada editable con etiquetas de filas y columnas."""
    header = st.columns(size + 1)
    header[0].markdown("&nbsp;", unsafe_allow_html=True)
    for column in range(size):
        header[column + 1].markdown(f"<div class='matrix-label'>c<sub>{column + 1}</sub></div>", unsafe_allow_html=True)

    matrix: List[List[float]] = []
    for row in range(size):
        columns = st.columns(size + 1)
        columns[0].markdown(f"<div class='matrix-label'>f<sub>{row + 1}</sub></div>", unsafe_allow_html=True)
        matrix_row = []
        for column in range(size):
            key = matrix_cell_key(size, row, column)
            if key not in st.session_state:
                st.session_state[key] = 0.0
            matrix_row.append(
                columns[column + 1].number_input(
                    f"a[{row + 1},{column + 1}]",
                    step=1.0,
                    format="%.2f",
                    key=key,
                    label_visibility="collapsed",
                )
            )
        matrix.append(matrix_row)
    return matrix


def _dataframe_from_matrix(matrix: np.ndarray, row_prefix: str = "f", col_prefix: str = "c") -> pd.DataFrame:
    return pd.DataFrame(
        [[format_number(value, 5) for value in row] for row in matrix],
        columns=[f"{col_prefix}{index + 1}" for index in range(matrix.shape[1])],
        index=[f"{row_prefix}{index + 1}" for index in range(matrix.shape[0])],
    )


def render_results(matrix: List[List[float]], eigenvalues: np.ndarray, eigenvectors: np.ndarray, solver: Any) -> None:
    """Muestra resultados, diagnóstico espectral y verificación numérica."""
    analysis = solver.analysis()
    matrix_array = np.asarray(matrix, dtype=float)

    st.markdown("<div class='section-kicker'>DIAGNÓSTICO ESPECTRAL</div>", unsafe_allow_html=True)
    st.subheader("Lectura rápida de la matriz")
    metrics = st.columns(4)
    metrics[0].metric("Determinante", str(analysis["determinant"]))
    metrics[1].metric("Traza", str(analysis["trace"]))
    metrics[2].metric("Rango", str(analysis["rank"]))
    metrics[3].metric("Pares base", f"{len(eigenvalues)} / {len(matrix)}")

    badges = [
        "✓ Invertible" if analysis["is_invertible"] else "× Singular",
        "✓ Simétrica" if analysis["is_symmetric"] else "• No simétrica",
        "✓ Diagonalizable" if analysis["is_diagonalizable"] else "× No diagonalizable",
        "• Eigenvalores complejos" if analysis["has_complex_eigenvalues"] else "✓ Espectro real",
    ]
    st.markdown("".join(f"<span class='status-badge'>{badge}</span>" for badge in badges), unsafe_allow_html=True)

    if not analysis["is_diagonalizable"]:
        st.warning(
            "La matriz no tiene suficientes eigenvectores linealmente independientes para formar una base. "
            "Aun así se muestra la base disponible de cada espacio propio."
        )
    if analysis["has_complex_eigenvalues"]:
        st.info("La matriz tiene eigenvalores complejos; se muestran en la forma a ± bi.")

    matrix_column, values_column = st.columns([1, 1.35])
    with matrix_column:
        st.markdown("#### Matriz A")
        st.dataframe(_dataframe_from_matrix(matrix_array), width="stretch", hide_index=False)
    with values_column:
        st.markdown("#### Eigenvalores y multiplicidades")
        multiplicity_rows = []
        for eigenvalue, algebraic in analysis["algebraic_multiplicities"].items():
            multiplicity_rows.append(
                {
                    "Eigenvalor λ": format_number(complex(eigenvalue.evalf())),
                    "Algebraica": algebraic,
                    "Geométrica": analysis["geometric_multiplicities"][eigenvalue],
                }
            )
        st.dataframe(pd.DataFrame(multiplicity_rows), width="stretch", hide_index=True)

    st.markdown("#### Bases de los espacios propios")
    if eigenvectors.shape[1] == 0:
        st.error("No se pudo construir una base de vectores propios para esta matriz.")
    else:
        vector_columns = st.columns(min(3, eigenvectors.shape[1]))
        for index, eigenvalue in enumerate(eigenvalues):
            with vector_columns[index % len(vector_columns)]:
                vector = eigenvectors[:, index]
                st.markdown(f"<div class='eigen-card'><b>λ = {format_number(eigenvalue)}</b><br><span>Vector propio {index + 1}</span></div>", unsafe_allow_html=True)
                st.dataframe(
                    pd.DataFrame({"v": [format_number(item, 6) for item in vector]}, index=[f"v{row + 1}" for row in range(len(vector))]),
                    width="stretch",
                )

    st.markdown("#### Verificación numérica")
    verification_rows = []
    for index, verification in enumerate(solver.verify(eigenvalues, eigenvectors), start=1):
        verification_rows.append(
            {
                "Vector": f"v{index}",
                "λ": format_number(verification["eigenvalue"]),
                "‖Av − λv‖": f"{verification['residual']:.2e}",
                "Estado": "✓ Verificado" if verification["verified"] else "⚠ Revisar",
            }
        )
    st.dataframe(pd.DataFrame(verification_rows), width="stretch", hide_index=True)

    if matrix_array.shape == (2, 2) and not analysis["has_complex_eigenvalues"] and eigenvectors.shape[1]:
        st.markdown("#### Transformación en el plano")
        _plot_eigenvectors_2d(eigenvalues, eigenvectors, matrix_array)


def _plot_eigenvectors_2d(eigenvalues: np.ndarray, eigenvectors: np.ndarray, matrix: np.ndarray) -> None:
    """Visualiza v y Av con límites que se ajustan a los datos."""
    figure = go.Figure()
    colors = ["#7c3aed", "#06b6d4", "#f59e0b", "#ef4444"]
    endpoints: List[float] = [1.0]

    for index in range(eigenvectors.shape[1]):
        vector = np.real_if_close(eigenvectors[:, index]).real
        transformed = matrix @ vector
        endpoints.extend(vector.tolist() + transformed.tolist())
        color = colors[index % len(colors)]
        figure.add_trace(
            go.Scatter(
                x=[0, vector[0]], y=[0, vector[1]], mode="lines+markers",
                line={"width": 4, "color": color}, name=f"v{index + 1}",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[0, transformed[0]], y=[0, transformed[1]], mode="lines+markers",
                line={"width": 3, "dash": "dash", "color": color}, name=f"A·v{index + 1}",
            )
        )

    limit = max(1.5, max(abs(value) for value in endpoints) * 1.25)
    figure.update_layout(
        height=390,
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        legend={"orientation": "h", "y": 1.08},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        xaxis={"title": "x₁", "range": [-limit, limit], "zeroline": True, "gridcolor": "#e2e8f0"},
        yaxis={"title": "x₂", "range": [-limit, limit], "zeroline": True, "scaleanchor": "x", "scaleratio": 1, "gridcolor": "#e2e8f0"},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def render_steps(steps: List[Dict[str, Any]]) -> None:
    """Renderiza el desarrollo simbólico sin saturar la pantalla."""
    st.markdown("<div class='section-kicker'>DESARROLLO SIMBÓLICO</div>", unsafe_allow_html=True)
    st.subheader("Cómo se obtuvo el resultado")
    for step in steps:
        status = step.get("found")
        icon = "✓" if status is True else "!" if status is False else "→"
        with st.expander(f"{icon}  Paso {step.get('step', '?')} · {step.get('title', '')}", expanded=step.get("step") in (1, 2, 3)):
            if step.get("description"):
                st.write(step["description"])
            if step.get("latex"):
                st.latex(step["latex"])
            if step.get("matrix_display") and step.get("matrix") is not None:
                from sympy import latex as sympy_latex

                st.latex(sympy_latex(step["matrix"]))
