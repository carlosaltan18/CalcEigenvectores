import streamlit as st # type: ignore
import numpy as np
from eigen_solver import EigenSolver
from ui_components import render_matrix_input, render_results, render_steps

st.set_page_config(
    page_title="Calculadora de Eigenvalores",
    page_icon="🔢",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #FFFFFF;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .step-box {
        background: #f0f4ff;
        border-left: 4px solid #4a6cf7;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-family: monospace;
    }
    .result-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .eigen-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4a6cf7;
    }
    .algo-box {
        background: #1a1a2e;
        color: #a8d8a8;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-family: monospace;
        font-size: 0.85rem;
        line-height: 1.8;
    }    
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔢 Calculadora de Eigenvalores y Eigenvectores</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Basada en el algoritmo det(A − λI) = 0 con resolución paso a paso</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("⚙️ Configuración")

    n = st.selectbox("Tamaño de la matriz (n × n)", options=[2, 3, 4], index=0)

    with st.expander("📋 Ver Algoritmo", expanded=False):
        st.markdown("""
<div class="algo-box">
Algorithm: Eigenvalues_and_Eigenvectors<br>
Input: Matriz A (n × n)<br>
Output: Eigenvalores λ, Eigenvectores v<br><br>
1: Construir I (identidad n × n)<br>
2: Calcular B ← A − λI<br>
3: Calcular det(B)<br>
4: Resolver det(A − λI) = 0<br>
5: Obtener eigenvalores λ<br>
6: for each λ do<br>
7: &nbsp;&nbsp;&nbsp;C ← A − λI<br>
8: &nbsp;&nbsp;&nbsp;Resolver C · v = 0<br>
9: &nbsp;&nbsp;&nbsp;if solución no trivial:<br>
10:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Obtener eigenvector v<br>
11:&nbsp;&nbsp;&nbsp;end if<br>
12: end for<br>
13: Retornar λ y v
</div>
""", unsafe_allow_html=True)

    st.subheader(f"📝 Ingresa la Matriz {n}×{n}")
    matrix = render_matrix_input(n)

    use_example = st.button("📌 Cargar ejemplo", use_container_width=True)
    if use_example:
        if n == 2:
            st.session_state['example_matrix'] = [[4, 1], [2, 3]]
        elif n == 3:
            st.session_state['example_matrix'] = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
        else:
            st.session_state['example_matrix'] = [[1,2,0,0],[2,1,0,0],[0,0,3,1],[0,0,1,3]]
        st.rerun()

    calculate = st.button("🚀 Calcular Eigenvalores", type="primary", use_container_width=True)

with col_right:
    if calculate and matrix is not None:
        solver = EigenSolver(matrix)
        eigenvalues, eigenvectors, steps = solver.solve()

        tab1, tab2 = st.tabs(["📊 Resultados", "🔍 Pasos del Algoritmo"])

        with tab1:
            render_results(matrix, eigenvalues, eigenvectors, n)

        with tab2:
            render_steps(steps)
    else:
        st.info("👈 Ingresa una matriz y presiona **Calcular Eigenvalores** para comenzar.")
        st.markdown("### ¿Qué son los Eigenvalores?")
        st.markdown("""
        Un **eigenvalor** λ de una matriz A es un escalar tal que existe un vector no nulo **v** donde:

        > **A · v = λ · v**

        El vector **v** se llama **eigenvector** asociado a λ.

        **Aplicaciones:** análisis de componentes principales (PCA), vibraciones estructurales,
        Google PageRank, mecánica cuántica, redes neuronales y más.
        """)