"""Laboratorio interactivo de álgebra lineal y PCA."""

from __future__ import annotations

from io import BytesIO, StringIO
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from eigen_solver import EigenSolver
from pca_tools import color_image_pca, color_image_svd, fit_pca
from ui_components import format_number, load_matrix, render_matrix_input, render_results, render_steps


st.set_page_config(page_title="Álgebra Visual", page_icon="◈", layout="wide", initial_sidebar_state="expanded")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --ink: #0f172a; --muted: #475569; --violet: #6d28d9; --cyan: #0e7490; }
        .stApp, [data-testid="stAppViewContainer"] { background: radial-gradient(circle at 10% -10%, #e9ddff 0, transparent 32rem), #f8fafc; color: var(--ink); color-scheme: light; }
        [data-testid="stMain"] { color: var(--ink); }
        section[data-testid="stSidebar"] { background: #111b34; }
        section[data-testid="stSidebar"] * { color: #e8efff; }
        section[data-testid="stSidebar"] .stRadio label { padding: .35rem 0; }
        .block-container { max-width: 1360px; padding-top: 2rem; padding-bottom: 3rem; }
        .hero { background: linear-gradient(115deg, #172554 0%, #4c1d95 55%, #0e7490 100%); border-radius: 24px; padding: 2.15rem 2.35rem; color: white; box-shadow: 0 18px 38px rgba(49, 46, 129, .22); margin-bottom: 1.5rem; }
        .hero-kicker, .section-kicker { font-size: .74rem; font-weight: 800; letter-spacing: .12em; }
        .hero-kicker { color: #c4b5fd; margin-bottom: .5rem; }
        .section-kicker { color: #7c3aed; margin-top: .8rem; }
        .hero h1 { color: white; font-size: clamp(2rem, 4vw, 3.25rem); line-height: 1.08; margin: 0 0 .55rem; }
        .hero p { max-width: 760px; color: #e0e7ff; font-size: 1.06rem; margin: 0; }
        .matrix-label { color: #64748b; font-size: .84rem; font-weight: 800; text-align: center; padding-top: .6rem; }
        .status-badge { display: inline-block; padding: .35rem .65rem; background: #eef2ff; border: 1px solid #dbeafe; border-radius: 999px; color: #3730a3; font-size: .82rem; font-weight: 650; margin: 0 .35rem .65rem 0; }
        .eigen-card { background: linear-gradient(135deg, #f5f3ff, #ecfeff); border: 1px solid #ddd6fe; border-radius: 12px; padding: .75rem .8rem; color: #312e81; }
        .eigen-card span { color: #64748b; font-size: .8rem; }
        div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #dbe3ef; border-radius: 14px; padding: .6rem .8rem; box-shadow: 0 3px 12px rgba(15, 23, 42, .04); }
        [data-testid="stMain"] [data-testid="stMetricLabel"], [data-testid="stMain"] [data-testid="stMetricLabel"] *, [data-testid="stMain"] [data-testid="stMetricDelta"] { color: #475569 !important; }
        [data-testid="stMain"] [data-testid="stMetricValue"], [data-testid="stMain"] [data-testid="stMetricValue"] * { color: #0f172a !important; }
        [data-testid="stMain"] label, [data-testid="stMain"] label p, [data-testid="stMain"] [data-testid="stWidgetLabel"], [data-testid="stMain"] [data-testid="stWidgetLabel"] p, [data-testid="stMain"] .stCaption, [data-testid="stMain"] .stCaption p { color: #334155 !important; font-weight: 600; }
        [data-testid="stMain"] input, [data-testid="stMain"] textarea { color: #0f172a !important; caret-color: #0f172a !important; }
        [data-testid="stMain"] [data-baseweb="input"], [data-testid="stMain"] [data-baseweb="textarea"] { background: #ffffff !important; border-color: #cbd5e1 !important; }
        [data-testid="stMain"] [data-baseweb="textarea"] textarea { background: transparent !important; color: #0f172a !important; }
        [data-testid="stMain"] [data-baseweb="slider"] div { color: #334155; }
        [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] { background: #ffffff !important; border: 1px dashed #94a3b8 !important; }
        [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] * { color: #334155 !important; }
        [data-testid="stMain"] [data-testid="stFileUploader"] small { color: #64748b !important; }
        .tip-box { background: #eff6ff; border-left: 4px solid #0ea5e9; padding: .8rem 1rem; border-radius: 0 10px 10px 0; color: #1e3a5f; }
        .footer-note { color: #64748b; font-size: .85rem; text-align: center; padding-top: 1.5rem; }
        div[data-testid="stExpander"] { background: white; border: 1px solid #e2e8f0; border-radius: 12px; }
        .stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 650; }
        [data-testid="stMain"] [role="tab"] { color: #334155 !important; font-weight: 650; }
        [data-testid="stMain"] [role="tab"][aria-selected="true"] { color: #5b21b6 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""<div class='hero'><div class='hero-kicker'>{kicker}</div><h1>{title}</h1><p>{description}</p></div>""",
        unsafe_allow_html=True,
    )


def matrix_examples(size: int) -> dict[str, np.ndarray]:
    examples = {
        2: {
            "Diagonal (simple)": np.array([[4, 0], [0, 2]]),
            "Simétrica": np.array([[4, 2], [2, 1]]),
            "Rotación (complejos)": np.array([[0, -1], [1, 0]]),
            "No diagonalizable": np.array([[2, 1], [0, 2]]),
        },
        3: {
            "Diagonal": np.array([[5, 0, 0], [0, 3, 0], [0, 0, 1]]),
            "Simétrica": np.array([[4, 1, 1], [1, 3, 0], [1, 0, 2]]),
            "Triangular": np.array([[4, 1, 0], [0, 2, 1], [0, 0, 1]]),
        },
        4: {
            "Diagonal": np.diag([7, 4, 2, 1]),
            "Simétrica": np.array([[5, 1, 0, 0], [1, 4, 1, 0], [0, 1, 3, 1], [0, 0, 1, 2]]),
            "Bloques": np.array([[3, 1, 0, 0], [1, 3, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]),
        },
    }
    return examples[size]


def render_spectral_calculator() -> None:
    render_hero(
        "LABORATORIO ESPECTRAL",
        "Entiende la matriz, no solo el resultado.",
        "Calcula eigenvalores y bases de espacios propios, verifica Av = λv y explora qué propiedades cambian la geometría de una transformación lineal.",
    )
    left, right = st.columns([0.78, 1.55], gap="large")
    with left:
        st.markdown("<div class='section-kicker'>CONFIGURACIÓN</div>", unsafe_allow_html=True)
        st.subheader("Construye tu matriz")
        size = st.select_slider("Dimensión", options=[2, 3, 4], value=2, format_func=lambda value: f"{value} × {value}")
        examples = matrix_examples(size)
        example_name = st.selectbox("Ejemplo guiado", ["Personalizada", *examples.keys()])
        action_left, action_right = st.columns(2)
        with action_left:
            if st.button("Cargar ejemplo", width="stretch", disabled=example_name == "Personalizada"):
                load_matrix(examples[example_name])
        with action_right:
            if st.button("Limpiar", width="stretch"):
                load_matrix(np.zeros((size, size)))

        st.markdown("#### Matriz A")
        matrix = render_matrix_input(size)
        calculate = st.button("Analizar matriz", type="primary", width="stretch")
        st.markdown(
            "<div class='tip-box'><b>Pista:</b> prueba la matriz de rotación para ver pares complejos o la matriz no diagonalizable para comparar multiplicidades.</div>",
            unsafe_allow_html=True,
        )

    with right:
        if calculate:
            try:
                solver = EigenSolver(matrix)
                eigenvalues, eigenvectors, steps = solver.solve()
                tab_results, tab_steps, tab_concepts = st.tabs(["Resultados", "Procedimiento", "Guía rápida"])
                with tab_results:
                    render_results(matrix, eigenvalues, eigenvectors, solver)
                with tab_steps:
                    render_steps(steps)
                with tab_concepts:
                    st.markdown("#### Ideas clave")
                    st.markdown(
                        """
                        - Un **eigenvector** conserva su dirección tras aplicar la transformación A; solo cambia su escala por λ.
                        - La **multiplicidad algebraica** indica cuántas veces aparece una raíz en el polinomio característico.
                        - La **multiplicidad geométrica** es la dimensión del espacio propio. Si las bases propias suman n, A es diagonalizable.
                        - Las matrices reales también pueden tener eigenvalores complejos, normalmente en pares conjugados.
                        """
                    )
            except Exception as error:
                st.error(f"No fue posible analizar la matriz: {error}")
        else:
            st.markdown("<div class='section-kicker'>EMPIEZA AQUÍ</div>", unsafe_allow_html=True)
            st.subheader("Una calculadora con contexto")
            st.write(
                "Elige un ejemplo o ingresa tus valores. El resultado incluye propiedades matriciales, espacios propios, comprobación numérica y, para matrices 2 × 2 reales, una visualización de la transformación."
            )
            st.info("Los números de entrada son reales; la aplicación también resuelve eigenvalores complejos que resulten de esas matrices.")


def demo_dataset(name: str) -> Tuple[np.ndarray, List[str]]:
    generator = np.random.default_rng(2026)
    if name == "Nube correlacionada (2 variables)":
        base = generator.normal(0, 1, 90)
        data = np.column_stack([base * 2.1 + generator.normal(0, 0.25, 90), base * 0.75 + generator.normal(0, 0.2, 90)])
        return data, ["x₁", "x₂"]
    if name == "Mediciones de estudiantes (4 variables)":
        base = generator.normal(0, 1, 75)
        data = np.column_stack(
            [170 + 8 * base + generator.normal(0, 2, 75), 63 + 9 * base + generator.normal(0, 3, 75), 71 + 6 * base + generator.normal(0, 3, 75), 8 + 1.5 * base + generator.normal(0, 1, 75)]
        )
        return data, ["estatura", "peso", "puntaje", "horas"]
    angles = generator.uniform(0, 2 * np.pi, 120)
    radii = generator.normal(1.7, 0.18, 120)
    data = np.column_stack([radii * np.cos(angles), radii * np.sin(angles), 0.65 * radii + generator.normal(0, 0.12, 120)])
    return data, ["x", "y", "z"]


def parse_csv_data(raw_data: str) -> np.ndarray:
    data = np.genfromtxt(StringIO(raw_data.strip()), delimiter=",", dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if not np.isfinite(data).all():
        raise ValueError("Usa únicamente números separados por comas; cada fila debe ser una observación.")
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("Escribe como mínimo dos filas y dos columnas.")
    return data


def load_uploaded_csv(uploaded_file) -> Tuple[np.ndarray, List[str], int]:
    """Carga un CSV con encabezados y conserva únicamente columnas numéricas útiles."""
    frame = pd.read_csv(uploaded_file)
    numeric = frame.select_dtypes(include=[np.number]).dropna(axis=0, how="any")
    if numeric.shape[1] < 2:
        raise ValueError("El archivo necesita al menos dos columnas numéricas con encabezados.")
    if numeric.shape[0] < 2:
        raise ValueError("El archivo necesita al menos dos filas numéricas completas.")
    return numeric.to_numpy(dtype=float), numeric.columns.astype(str).tolist(), len(frame) - len(numeric)


def render_scree_plot(explained_ratio: np.ndarray) -> None:
    positions = list(range(1, len(explained_ratio) + 1))
    figure = go.Figure()
    figure.add_trace(go.Bar(x=positions, y=explained_ratio * 100, marker_color="#7c3aed", name="Individual"))
    figure.add_trace(go.Scatter(x=positions, y=np.cumsum(explained_ratio) * 100, mode="lines+markers", line={"color": "#0891b2", "width": 3}, name="Acumulada"))
    figure.update_layout(
        height=330, margin={"l": 10, "r": 10, "t": 25, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        template="plotly_white", font={"color": "#0f172a", "family": "Arial, sans-serif"},
        xaxis_title="Componente principal", yaxis_title="Varianza explicada (%)", yaxis={"range": [0, 105], "gridcolor": "#e2e8f0"}, legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def render_pca_lab() -> None:
    render_hero(
        "PCA INTERACTIVO",
        "Reduce dimensiones y conserva la señal.",
        "Explora cómo la matriz de covarianza y sus eigenvectores encuentran las direcciones con mayor variación en un conjunto de datos.",
    )
    control, workspace = st.columns([0.78, 1.55], gap="large")
    with control:
        st.markdown("<div class='section-kicker'>DATOS</div>", unsafe_allow_html=True)
        source = st.radio("Fuente", ["Dataset de ejemplo", "Pegar datos", "Subir CSV"], label_visibility="collapsed")
        if source == "Dataset de ejemplo":
            dataset_name = st.selectbox("Dataset", ["Nube correlacionada (2 variables)", "Mediciones de estudiantes (4 variables)", "Anillo tridimensional"])
            data, labels = demo_dataset(dataset_name)
            st.caption(f"{data.shape[0]} observaciones · {data.shape[1]} variables")
        elif source == "Pegar datos":
            raw_data = st.text_area(
                "Observaciones por fila, variables separadas por comas",
                value="1.0, 2.1, 1.4\n2.0, 4.1, 2.6\n3.1, 6.2, 3.8\n4.0, 8.0, 5.3\n5.2, 10.4, 6.4",
                height=170,
            )
            try:
                data = parse_csv_data(raw_data)
                labels = [f"variable {index + 1}" for index in range(data.shape[1])]
                st.caption(f"{data.shape[0]} observaciones · {data.shape[1]} variables")
            except ValueError as error:
                st.error(str(error))
                return
        else:
            uploaded_csv = st.file_uploader("Archivo CSV con encabezados", type=["csv"], help="La aplicación usará las columnas numéricas y omitirá filas incompletas.")
            if uploaded_csv is None:
                st.info("Sube un CSV con encabezados, por ejemplo: edad,ingreso,puntaje.")
                return
            try:
                data, labels, discarded_rows = load_uploaded_csv(uploaded_csv)
                st.caption(f"{data.shape[0]} observaciones · {data.shape[1]} variables numéricas")
                if discarded_rows:
                    st.caption(f"Se omitieron {discarded_rows} fila(s) con datos faltantes.")
            except (ValueError, pd.errors.ParserError) as error:
                st.error(f"No se pudo leer el CSV: {error}")
                return

        max_components = min(data.shape)
        components = st.slider("Componentes a conservar", 1, max_components, min(2, max_components))
        standardize = st.checkbox("Estandarizar variables", value=True, help="Convierte cada variable a una escala comparable antes de PCA.")
        available_views = ["2D"] + (["3D"] if data.shape[1] >= 3 else [])
        projection_view = st.selectbox("Visualizar reducción", available_views, help="La vista 3D utiliza PC1, PC2 y PC3.")
        st.markdown(
            "<div class='tip-box'><b>Decisión importante:</b> estandariza cuando las variables usan unidades o escalas diferentes.</div>",
            unsafe_allow_html=True,
        )

    with workspace:
        try:
            result = fit_pca(data, components, standardize=standardize)
        except ValueError as error:
            st.error(str(error))
            return

        retained = result["cumulative_ratio"][components - 1] * 100
        metrics = st.columns(3)
        metrics[0].metric("Varianza retenida", f"{retained:.2f}%")
        metrics[1].metric("Componentes", f"{components} de {data.shape[1]}")
        metrics[2].metric("Error MSE", f"{result['reconstruction_error']:.4f}")
        if result["total_variance"] < 1e-12:
            st.warning("Las variables no presentan variación. PCA no puede identificar una dirección principal útil.")

        chart_column, scatter_column = st.columns(2)
        with chart_column:
            st.markdown("#### Varianza explicada")
            render_scree_plot(result["explained_ratio"])
        with scatter_column:
            st.markdown("#### Datos proyectados")
            scores = result["all_scores"]
            if projection_view == "3D":
                figure = go.Figure(
                    go.Scatter3d(
                        x=scores[:, 0], y=scores[:, 1], z=scores[:, 2], mode="markers",
                        marker={"color": "#7c3aed", "size": 5, "opacity": .78},
                        text=[f"Observación {index + 1}" for index in range(len(scores))],
                    )
                )
                figure.update_layout(
                    height=330, margin={"l": 0, "r": 0, "t": 25, "b": 0}, paper_bgcolor="rgba(0,0,0,0)",
                    template="plotly_white", font={"color": "#0f172a", "family": "Arial, sans-serif"},
                    scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"},
                )
            elif scores.shape[1] >= 2:
                figure = go.Figure(go.Scatter(x=scores[:, 0], y=scores[:, 1], mode="markers", marker={"color": "#7c3aed", "size": 9, "opacity": .78}, text=[f"Observación {index + 1}" for index in range(len(scores))]))
                x_title, y_title = "PC1", "PC2"
            else:
                figure = go.Figure(go.Scatter(x=np.arange(1, len(scores) + 1), y=scores[:, 0], mode="markers", marker={"color": "#7c3aed", "size": 9, "opacity": .78}))
                x_title, y_title = "Observación", "PC1"
            if projection_view != "3D":
                figure.update_layout(height=330, margin={"l": 10, "r": 10, "t": 25, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff", template="plotly_white", font={"color": "#0f172a", "family": "Arial, sans-serif"}, xaxis_title=x_title, yaxis_title=y_title)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

        data_tab, covariance_tab, components_tab = st.tabs(["Datos", "Covarianza", "Componentes"])
        with data_tab:
            st.dataframe(pd.DataFrame(data, columns=labels).round(3), width="stretch", hide_index=True, height=250)
        with covariance_tab:
            covariance = result["covariance"]
            heatmap = go.Figure(go.Heatmap(z=covariance, x=labels, y=labels, colorscale="PuBu", zmid=0, colorbar={"title": "Cov."}))
            heatmap.update_layout(height=360, margin={"l": 10, "r": 10, "t": 15, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", template="plotly_white", font={"color": "#0f172a", "family": "Arial, sans-serif"})
            st.plotly_chart(heatmap, width="stretch", config={"displayModeBar": False})
        with components_tab:
            loading_data = {f"PC{index + 1}": result["components"][:, index] for index in range(components)}
            loading_frame = pd.DataFrame(loading_data, index=labels).round(4)
            st.caption("Cada columna es un eigenvector de la matriz de covarianza; sus valores muestran la contribución de cada variable.")
            st.dataframe(loading_frame, width="stretch")


def image_to_bytes(image_array: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def compress_rgb_with_pca(image_array: np.ndarray, components: int):
    return color_image_pca(image_array, components)


@st.cache_data(show_spinner=False)
def compress_rgb_with_svd(image_array: np.ndarray, components: int):
    return color_image_svd(image_array, components)


def render_color_result(original: np.ndarray, result: dict, method: str, components: int) -> None:
    """Muestra la comparación visual que hace comprensible la compresión."""
    metrics = st.columns(4)
    metrics[0].metric("Energía retenida", f"{result['retained_variance'] * 100:.2f}%")
    metrics[1].metric("Error MSE", f"{result['mse']:.2f}")
    metrics[2].metric("PSNR", "∞" if np.isinf(result["psnr"]) else f"{result['psnr']:.2f} dB")
    metrics[3].metric("Datos del modelo", f"{result['storage_ratio']:.1f}%")

    original_column, reconstructed_column, difference_column = st.columns(3)
    with original_column:
        st.image(original, caption="Original RGB", width="stretch")
    with reconstructed_column:
        st.image(result["reconstructed_image"], caption=f"Reconstrucción {method} · k={components}", width="stretch")
    with difference_column:
        st.image(result["difference_image"], caption="Diferencia absoluta × 4", width="stretch")

    st.download_button(
        f"Descargar reconstrucción {method}",
        data=image_to_bytes(result["reconstructed_image"]),
        file_name=f"{method.lower()}_rgb_k{components}.png",
        mime="image/png",
        width="stretch",
    )


def render_image_compressor() -> None:
    render_hero(
        "PCA, SVD Y VISIÓN POR COMPUTADORA",
        "Comprime imágenes RGB sin perder el color.",
        "Aplica PCA y SVD por separado a los canales rojo, verde y azul. Compara reconstrucción, diferencia visual, error y energía conservada.",
    )
    uploaded_file = st.file_uploader("Sube una imagen PNG o JPG", type=["png", "jpg", "jpeg"], help="La imagen se procesa localmente dentro de la sesión.")
    if uploaded_file is None:
        st.info("Sube una imagen para iniciar la comparación PCA vs. SVD. Las imágenes grandes se ajustan a un máximo de 360 px por lado para que la demostración sea ágil.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
        image.thumbnail((360, 360), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
    except Exception as error:
        st.error(f"No se pudo abrir la imagen: {error}")
        return

    height, width, _ = image_array.shape
    controls, output = st.columns([0.78, 1.55], gap="large")
    with controls:
        st.markdown("<div class='section-kicker'>AJUSTE RGB</div>", unsafe_allow_html=True)
        st.subheader("Elige el detalle")
        maximum = min(height, width)
        default = max(1, min(maximum, round(maximum * 0.12)))
        components = st.slider("Componentes principales (k)", 1, maximum, default)
        st.image(image, caption=f"Original · {width} × {height} px", width="stretch")
        st.caption("A menor k, mayor compresión y menor fidelidad. PCA centra cada canal; SVD calcula directamente una aproximación de rango k.")

    with output:
        with st.spinner("Calculando PCA y SVD en los tres canales de color…"):
            pca_result = compress_rgb_with_pca(image_array, components)
            svd_result = compress_rgb_with_svd(image_array, components)

        pca_tab, svd_tab, comparison_tab = st.tabs(["PCA por canal RGB", "SVD de rango k", "PCA vs. SVD"])
        with pca_tab:
            render_color_result(image_array, pca_result, "PCA", components)
            st.caption("PCA encuentra las direcciones de máxima varianza después de centrar cada canal RGB. “Energía retenida” corresponde a la varianza explicada ponderada de los tres canales.")
        with svd_tab:
            render_color_result(image_array, svd_result, "SVD", components)
            st.caption("SVD aproxima directamente cada matriz de color con k valores singulares. Es la aproximación óptima de rango k respecto al error cuadrático para cada canal.")
        with comparison_tab:
            comparison = pd.DataFrame(
                [
                    {"Método": "PCA por canal RGB", "Energía retenida": f"{pca_result['retained_variance'] * 100:.2f}%", "MSE": f"{pca_result['mse']:.2f}", "PSNR": "∞" if np.isinf(pca_result["psnr"]) else f"{pca_result['psnr']:.2f} dB", "Datos del modelo": f"{pca_result['storage_ratio']:.1f}%"},
                    {"Método": "SVD de rango k", "Energía retenida": f"{svd_result['retained_variance'] * 100:.2f}%", "MSE": f"{svd_result['mse']:.2f}", "PSNR": "∞" if np.isinf(svd_result["psnr"]) else f"{svd_result['psnr']:.2f} dB", "Datos del modelo": f"{svd_result['storage_ratio']:.1f}%"},
                ]
            )
            st.dataframe(comparison, width="stretch", hide_index=True)
            st.markdown("#### Cómo explicarlo")
            st.write("PCA descompone la varianza de datos centrados; SVD factoriza la matriz original. Ambas reducen dimensionalidad y usan componentes ordenados por importancia. En imágenes, SVD suele minimizar el error de una aproximación de rango fijo, mientras PCA conecta directamente con la covarianza y los eigenvectores.")
        if pca_result["total_variance"] < 1e-12:
            st.info("La imagen no tiene variación tonal; cualquier número de componentes produce la misma reconstrucción.")


inject_styles()
st.sidebar.markdown("## ◈ Álgebra Visual")
st.sidebar.caption("Un laboratorio para explorar matrices, espacios propios y PCA.")
mode = st.sidebar.radio("Navegación", ["Calculadora espectral", "Laboratorio PCA", "Compresor de imágenes"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Ruta de aprendizaje**")
st.sidebar.caption("1. Analiza una matriz\n\n2. Relaciona covarianza y eigenvectores\n\n3. Aplica PCA a una imagen")

if mode == "Calculadora espectral":
    render_spectral_calculator()
elif mode == "Laboratorio PCA":
    render_pca_lab()
else:
    render_image_compressor()

st.markdown("<div class='footer-note'>Álgebra Visual · Eigenvalores, espacios propios, covarianza y reducción dimensional</div>", unsafe_allow_html=True)
