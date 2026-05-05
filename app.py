import streamlit as st
import numpy as np
from PIL import Image
from eigen_solver import EigenSolver
from ui_components import render_matrix_input, render_results, render_steps

st.set_page_config(
    page_title="Calculadora y PCA",
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

st.sidebar.title("Navegación")
st.sidebar.markdown("Elige una herramienta:")
modo = st.sidebar.radio("", ["🧮 Calculadora de Eigenvalores", "🖼️ Compresor de Imágenes (PCA)"])
st.sidebar.markdown("---")
st.sidebar.info("Proyecto que demuestra la teoría de Eigenvalores y su aplicación práctica.")

# ==========================================
# MODO 1: CALCULADORA TEÓRICA 
# ==========================================
if modo == "🧮 Calculadora de Eigenvalores":
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
        calculate = st.button("🚀 Calcular Eigenvalores", type="primary", use_container_width=True)

    with col_right:
        if calculate and matrix is not None:
            solver = EigenSolver(matrix)
            eigenvalues, eigenvectors, steps = solver.solve()

            tab1, tab2 = st.tabs(["📊 Resultados", "🔍 Procedimiento"])

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
            """)

# ==========================================
# MODO 2: COMPRESOR PCA
# ==========================================
elif modo == "🖼️ Compresor de Imágenes (PCA)":
    st.markdown('<div class="main-title">🖼️ Compresión usando Eigenvectores (PCA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Reduce el peso de una imagen conservando los componentes principales</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Sube una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L')
        img_array = np.array(img)
        h, w = img_array.shape

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(img, caption=f"Imagen Original (Escala de grises) ({w}x{h} px)", use_container_width=True)
            max_components = min(w, h)
            
            st.markdown("### ⚙️ Ajuste PCA")
            k = st.slider(
                "Cantidad de Eigenvectores a usar:", 
                min_value=1, 
                max_value=max_components, 
                value=int(max_components * 0.1)
            )

        with col2:
            with st.spinner('Calculando Matriz de Covarianza y Eigenvectores...'):
                mean = np.mean(img_array, axis=0)
                X_centered = img_array - mean
                cov_matrix = np.cov(X_centered, rowvar=False)

                # Cálculo de eigenvalores y eigenvectores
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

                sorted_idx = np.argsort(eigenvalues)[::-1]
                sorted_eigenvalues = eigenvalues[sorted_idx]
                sorted_eigenvectors = eigenvectors[:, sorted_idx]

                eigenvectors_subset = sorted_eigenvectors[:, :k]

                # Proyección y reconstrucción
                X_reduced = np.dot(X_centered, eigenvectors_subset)
                X_reconstructed = np.dot(X_reduced, eigenvectors_subset.T) + mean
                
                img_reconstructed = np.clip(X_reconstructed, 0, 255).astype(np.uint8)

                total_variance = np.sum(sorted_eigenvalues)
                explained_variance = np.sum(sorted_eigenvalues[:k]) / total_variance * 100
                compression_ratio = (h * k + k * w) / (h * w) * 100 

            st.image(Image.fromarray(img_reconstructed), caption=f"Imagen Reconstruida usando {k} componentes", use_container_width=True)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Componentes (k)", f"{k} de {max_components}")
            m2.metric("Varianza Retenida", f"{explained_variance:.2f}%")
            m3.metric("Tamaño Relativo", f"{compression_ratio:.1f}%")