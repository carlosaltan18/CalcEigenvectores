"""Laboratorio interactivo de álgebra lineal y PCA."""

from __future__ import annotations

from io import BytesIO, StringIO
from hashlib import sha256
from base64 import b64encode
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

from image_comparison import comparison_html, render_comparison_html, png_url
from image_compression import ImageWorkspace, preview_image

from eigen_solver import EigenSolver
from pca_tools import fit_pca
from ui_components import format_number, load_matrix, render_matrix_input, render_results, render_steps


st.set_page_config(page_title="Álgebra Visual", page_icon="◈", layout="wide", initial_sidebar_state="expanded")


def inject_styles() -> None:
    hero_asset = Path(__file__).with_name("assets") / "algebra-hero.png"
    hero_image = ""
    if hero_asset.exists():
        hero_image = "url('data:image/png;base64," + b64encode(hero_asset.read_bytes()).decode("ascii") + "')"
    styles = """
        <style>
        :root { --void:#080a0c; --carbon:#111518; --panel:#151a1e; --panel-2:#1a2025; --line:#343b42; --ink:#f4f0e9; --muted:#a8afb5; --ember:#ff8a1f; --signal:#f04b2b; --cyan:#75bbc7; }
        .stApp, [data-testid="stAppViewContainer"] { background: radial-gradient(circle at 78% -12%, rgba(190,65,20,.2), transparent 34rem), linear-gradient(115deg, rgba(255,138,31,.045), transparent 38%), var(--void); color:var(--ink); color-scheme:dark; }
        [data-testid="stMain"] { color:var(--ink); }
        [data-testid="stMainBlockContainer"], .block-container { max-width:1440px; padding-top:2.4rem; padding-bottom:3.5rem; }
        section[data-testid="stSidebar"] { background:linear-gradient(180deg,#171b1e,#0a0c0e 70%); border-right:1px solid var(--line); }
        section[data-testid="stSidebar"] * { color:var(--ink); }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top:1.15rem; }
        section[data-testid="stSidebar"] .stRadio label { padding:.48rem .7rem; border-left:2px solid transparent; transition:.18s ease; }
        section[data-testid="stSidebar"] .stRadio label:hover { background:rgba(255,138,31,.09); border-left-color:var(--ember); }
        .hero { position:relative; overflow:hidden; min-height:220px; padding:2.45rem 2.6rem; border:1px solid rgba(255,179,76,.32); border-radius:4px; background-image:linear-gradient(90deg,rgba(6,8,10,.97) 0%,rgba(6,8,10,.91) 39%,rgba(6,8,10,.23) 100%),_HERO_IMAGE_; background-position:center,right center; background-size:cover,cover; box-shadow:0 25px 70px rgba(0,0,0,.35); margin-bottom:1.7rem; }
        .hero::after { content:""; position:absolute; inset:0; pointer-events:none; background:repeating-linear-gradient(0deg,transparent 0 3px,rgba(255,255,255,.022) 3px 4px); mix-blend-mode:screen; }
        .hero > * { position:relative; z-index:1; }
        .hero-kicker, .section-kicker { font-family:"Courier New",monospace; font-size:.72rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; }
        .hero-kicker { color:var(--ember); margin-bottom:.7rem; }
        .section-kicker { color:var(--ember); margin-top:.8rem; }
        .hero h1 { max-width:670px; color:var(--ink); font-size:clamp(2.15rem,4vw,3.65rem); font-weight:800; letter-spacing:-.04em; line-height:1; margin:0 0 .7rem; }
        .hero p { max-width:700px; color:#d4d0c8; font-size:1.06rem; margin:0; }
        .hero-signals { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.35rem; }.hero-signals span { padding:.34rem .52rem; border:1px solid rgba(255,180,81,.38); color:#f0c28f; background:rgba(8,10,12,.42); font:700 .67rem "Courier New",monospace; letter-spacing:.09em; }
        .system-bar { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:0 0 .9rem; padding:.62rem .82rem; border:1px solid #394149; background:rgba(21,26,30,.76); }.system-title { display:flex; align-items:center; gap:.55rem; color:#f1ece2; font:700 .72rem "Courier New",monospace; letter-spacing:.13em; }.system-title::before { content:""; width:7px; height:7px; border-radius:50%; background:#ff8a1f; box-shadow:0 0 12px #ff8a1f; }.system-tags { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:.38rem; }.system-tags span { border-left:1px solid #59616a; padding-left:.42rem; color:#9fa8af; font:700 .65rem "Courier New",monospace; letter-spacing:.06em; }
        .brand-lockup { margin:-.25rem 0 1.5rem; padding:0 0 1rem; border-bottom:1px solid #3a4249; }.brand-lockup .brand-mark { color:var(--ember); font:800 .72rem "Courier New",monospace; letter-spacing:.2em; }.brand-lockup h2 { margin:.42rem 0 .35rem; font-size:1.55rem; }.brand-lockup p { margin:0; color:#a8afb5; font-size:.82rem; line-height:1.45; }
        h1,h2,h3 { color:var(--ink) !important; letter-spacing:-.02em; } p,li { color:#d0d4d6; }
        .matrix-label { color:var(--ember); font-family:"Courier New",monospace; font-size:.84rem; font-weight:800; text-align:center; padding-top:.6rem; }
        .status-badge { display:inline-block; padding:.35rem .65rem; border:1px solid #4c535a; border-radius:2px; color:#e9e5dc; font-family:"Courier New",monospace; font-size:.78rem; font-weight:700; margin:0 .35rem .65rem 0; background:#1b2024; }
        .eigen-card { background:linear-gradient(135deg,#1a1f23,#111416); border-left:3px solid var(--ember); border-radius:2px; padding:.85rem .9rem; color:var(--ink); }.eigen-card span { color:var(--muted); font-size:.8rem; }
        div[data-testid="stMetric"] { background:linear-gradient(135deg,#1b2024,#121619); border:1px solid #363e45; border-radius:3px; padding:.7rem .85rem; box-shadow:inset 0 1px 0 rgba(255,255,255,.035),0 10px 24px rgba(0,0,0,.16); }
        [data-testid="stMain"] [data-testid="stMetricLabel"], [data-testid="stMain"] [data-testid="stMetricLabel"] *, [data-testid="stMain"] [data-testid="stMetricDelta"] { color:var(--muted) !important; font-family:"Courier New",monospace; font-size:.74rem !important; letter-spacing:.055em; text-transform:uppercase; }
        [data-testid="stMain"] [data-testid="stMetricValue"], [data-testid="stMain"] [data-testid="stMetricValue"] * { color:var(--ember) !important; }
        [data-testid="stMain"] label, [data-testid="stMain"] label p, [data-testid="stMain"] [data-testid="stWidgetLabel"], [data-testid="stMain"] [data-testid="stWidgetLabel"] p, [data-testid="stMain"] .stCaption, [data-testid="stMain"] .stCaption p { color:#d7dbdc !important; font-weight:650; }
        [data-testid="stMain"] input, [data-testid="stMain"] textarea { color:var(--ink) !important; caret-color:var(--ember) !important; }
        [data-testid="stMain"] [data-baseweb="input"], [data-testid="stMain"] [data-baseweb="textarea"], [data-testid="stMain"] [data-baseweb="select"] > div { background:#111518 !important; border-color:#454d54 !important; border-radius:2px !important; }
        [data-testid="stMain"] [data-baseweb="textarea"] textarea { background:transparent !important; color:var(--ink) !important; }
        [data-testid="stMain"] [data-baseweb="slider"] div { color:var(--muted); }
        [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] { background:#111518 !important; border:1px dashed #757e84 !important; border-radius:2px !important; }
        [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] * { color:#d7dbdc !important; }
        [data-testid="stMain"] [data-testid="stFileUploader"] small { color:var(--muted) !important; }
        .tip-box { background:rgba(255,138,31,.08); border-left:3px solid var(--ember); padding:.9rem 1rem; border-radius:0; color:#e7e1d7; }
        .footer-note { color:#868f96; font-family:"Courier New",monospace; font-size:.75rem; letter-spacing:.08em; text-align:center; padding-top:1.8rem; }
        div[data-testid="stExpander"] { background:#13181b; border:1px solid #3a4249; border-radius:2px; } [data-testid="stExpander"] summary { color:var(--ink); }
        .stButton > button, .stDownloadButton > button { border:1px solid var(--ember); border-radius:2px; color:#f9f5ed; background:transparent; font-family:"Courier New",monospace; font-weight:700; letter-spacing:.04em; transition:.18s ease; }.stButton > button:hover, .stDownloadButton > button:hover { color:#111; background:var(--ember); border-color:var(--ember); }
        .stButton > button[kind="primary"] { background:var(--signal); border-color:var(--signal); }.stButton > button[kind="primary"]:hover { background:#ff6a41; border-color:#ff6a41; }
        [data-testid="stMain"] [role="tab"] { color:#aeb5ba !important; font-family:"Courier New",monospace; font-weight:700; letter-spacing:.04em; } [data-testid="stMain"] [role="tab"][aria-selected="true"] { color:var(--ember) !important; }
        [data-testid="stMain"] [data-testid="stDataFrame"], [data-testid="stMain"] [data-testid="stTable"] { border:1px solid #394149; }
        [data-testid="stMain"] .stAlert { background:#1a2024; border:1px solid #48515a; border-radius:2px; color:var(--ink); }
        @media (max-width:700px) { .hero { min-height:190px; padding:1.7rem; background-position:58% center,right center; }.hero h1 { font-size:2.2rem; }.block-container { padding-top:1.4rem; }.system-bar { align-items:flex-start; flex-direction:column; }.system-tags { justify-content:flex-start; } }
        </style>
        """
    st.markdown(styles.replace("_HERO_IMAGE_", hero_image or "none"), unsafe_allow_html=True)


def render_hero(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""<div class='hero'><div class='hero-kicker'>{kicker}</div><h1>{title}</h1><p>{description}</p><div class='hero-signals'><span>● SISTEMA ACTIVO</span><span>ÁLGEBRA LINEAL</span><span>ANÁLISIS VISUAL</span></div></div>""",
        unsafe_allow_html=True,
    )


def render_system_bar(module: str, *tags: str) -> None:
    """Renderiza una barra de estado visual reutilizable para cada laboratorio."""
    tag_html = "".join(f"<span>{tag}</span>" for tag in tags)
    st.markdown(
        f"<div class='system-bar'><div class='system-title'>{module}</div><div class='system-tags'>{tag_html}</div></div>",
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
    render_system_bar("MÓDULO 01 · ESPECTRAL", "MATRICES", "ESPACIOS PROPIOS", "DIAGNÓSTICO")
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
    figure.add_trace(go.Bar(x=positions, y=explained_ratio * 100, marker_color="#f04b2b", name="Individual"))
    figure.add_trace(go.Scatter(x=positions, y=np.cumsum(explained_ratio) * 100, mode="lines+markers", line={"color": "#ff9c35", "width": 3}, name="Acumulada"))
    figure.update_layout(
        height=330, margin={"l": 10, "r": 10, "t": 25, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#14191d",
        template="plotly_dark", font={"color": "#e8e5dd", "family": "Arial, sans-serif"},
        xaxis_title="Componente principal", yaxis_title="Varianza explicada (%)", yaxis={"range": [0, 105], "gridcolor": "#394149"}, legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def render_pca_lab() -> None:
    render_system_bar("MÓDULO 02 · PCA", "DATOS", "COVARIANZA", "PROYECCIÓN")
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
                        marker={"color": "#ff8a1f", "size": 5, "opacity": .78},
                        text=[f"Observación {index + 1}" for index in range(len(scores))],
                    )
                )
                figure.update_layout(
                    height=330, margin={"l": 0, "r": 0, "t": 25, "b": 0}, paper_bgcolor="rgba(0,0,0,0)",
                    template="plotly_dark", font={"color": "#e8e5dd", "family": "Arial, sans-serif"},
                    scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"},
                )
            elif scores.shape[1] >= 2:
                figure = go.Figure(go.Scatter(x=scores[:, 0], y=scores[:, 1], mode="markers", marker={"color": "#ff8a1f", "size": 9, "opacity": .78}, text=[f"Observación {index + 1}" for index in range(len(scores))]))
                x_title, y_title = "PC1", "PC2"
            else:
                figure = go.Figure(go.Scatter(x=np.arange(1, len(scores) + 1), y=scores[:, 0], mode="markers", marker={"color": "#ff8a1f", "size": 9, "opacity": .78}))
                x_title, y_title = "Observación", "PC1"
            if projection_view != "3D":
                figure.update_layout(height=330, margin={"l": 10, "r": 10, "t": 25, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#14191d", template="plotly_dark", font={"color": "#e8e5dd", "family": "Arial, sans-serif"}, xaxis_title=x_title, yaxis_title=y_title)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

        data_tab, covariance_tab, components_tab = st.tabs(["Datos", "Covarianza", "Componentes"])
        with data_tab:
            st.dataframe(pd.DataFrame(data, columns=labels).round(3), width="stretch", hide_index=True, height=250)
        with covariance_tab:
            covariance = result["covariance"]
            heatmap = go.Figure(go.Heatmap(z=covariance, x=labels, y=labels, colorscale=[[0, "#1b2730"], [.5, "#d2c3a1"], [1, "#f04b2b"]], zmid=0, colorbar={"title": "Cov."}))
            heatmap.update_layout(height=360, margin={"l": 10, "r": 10, "t": 15, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", template="plotly_dark", font={"color": "#e8e5dd", "family": "Arial, sans-serif"})
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


def render_image_metrics(result: dict, method: str) -> None:
    metrics = st.columns(4)
    metrics[0].metric("Varianza retenida" if method == "PCA" else "Energía retenida", f"{result['retained_variance'] * 100:.2f}%")
    metrics[1].metric("Error MSE", f"{result['mse']:.2f}")
    metrics[2].metric("PSNR", "∞" if np.isinf(result["psnr"]) else f"{result['psnr']:.2f} dB")
    metrics[3].metric("Datos del modelo", f"{result['storage_ratio']:.1f}%")


def result_png(result: dict) -> bytes:
    """Reuse the current reconstruction encoding for display and download."""
    if "png" not in result:
        result["png"] = image_to_bytes(result["reconstructed_image"])
    return result["png"]


def render_image_compressor() -> None:
    render_system_bar("MÓDULO 03 · IMAGEN", "RGB", "RANGO k", "PCA / SVD")
    render_hero("PCA · SVD", "Reducción y reconstrucción RGB", "Explora el detalle que conserva cada método.")
    uploaded_file = st.file_uploader("Sube una imagen PNG o JPG", type=["png", "jpg", "jpeg"], help="Se reconstruyen los tres canales RGB; los colores y detalles pueden variar.")
    if uploaded_file is None:
        st.session_state.pop("image_workspace", None)
        st.info("Sube una imagen para comparar el original y su reconstrucción.")
        return
    resolution = st.selectbox("Resolución de procesamiento", ["Vista previa · 720 px", "Vista previa · 360 px", "Resolución original"], help="La vista previa reduce ambos lados por igual, sin recortar. La resolución original requiere más tiempo y memoria.")
    limit = {"Vista previa · 720 px": 720, "Vista previa · 360 px": 360, "Resolución original": None}[resolution]
    raw = uploaded_file.getvalue()
    identity = (sha256(raw).hexdigest(), limit)
    entry = st.session_state.get("image_workspace")
    if entry is None or entry["identity"] != identity:
        # Release the previous image's factors before allocating a new workspace.
        st.session_state.pop("image_workspace", None)
        entry = None
        try:
            image = ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert("RGB")
            source_size = image.size
            workspace = ImageWorkspace(preview_image(image, limit))
        except (ValueError, OSError, Image.DecompressionBombError) as error:
            st.error(f"No se pudo abrir la imagen: {error}")
            return
        entry = {"identity": identity, "workspace": workspace, "source_size": source_size,
                 "original_url": png_url(workspace.image), "html_key": None, "html": None}
        st.session_state.image_workspace = entry
    workspace = entry["workspace"]
    image_array = workspace.image
    height, width = image_array.shape[:2]
    source_width, source_height = entry["source_size"]
    st.caption(f"Archivo: {source_width} × {source_height} px · Comparación y descarga: {width} × {height} px · RGB con pérdida · Encuadre completo")
    # A radio selection gates Python execution; inactive views do no work.
    view = st.radio("Método", ["PCA RGB", "SVD rango k", "PCA vs. SVD"], horizontal=True)
    maximum = min(height, width)
    rank_key = "image_rank"
    if rank_key not in st.session_state:
        st.session_state[rank_key] = max(1, round(maximum * 0.12))
    st.session_state[rank_key] = min(maximum, max(1, st.session_state[rank_key]))
    components = st.slider("Componentes / rango (k)", 1, maximum, key=rank_key, help="Menor k reduce los datos del modelo y el detalle. La descomposición se reutiliza al cambiar k.")
    methods = ["PCA", "SVD"] if view == "PCA vs. SVD" else ["PCA" if view == "PCA RGB" else "SVD"]
    with st.spinner("Preparando reconstrucción RGB…"):
        results = {method: workspace.reconstruct(method, components) for method in methods}
    if len(methods) == 1:
        method = methods[0]
        render_image_metrics(results[method], method)
        reference = f"Original vs. {method}"
    else:
        st.dataframe(pd.DataFrame([
            {"Método": method, "MSE": result["mse"], "PSNR (dB)": result["psnr"], "Datos del modelo (%)": result["storage_ratio"]}
            for method, result in results.items()
        ]).round(2), width="stretch", hide_index=True)
        reference = st.selectbox("Comparar", ["Original vs. PCA", "Original vs. SVD", "PCA vs. SVD"])
    right_method = "PCA" if reference == "Original vs. PCA" else "SVD"
    left = results["PCA"]["reconstructed_image"] if reference == "PCA vs. SVD" else image_array
    html_key = (reference, components)
    if entry["html_key"] != html_key:
        # Only one inspector payload survives a rerun; never cache each visited k.
        entry["html"] = None
        left_url = ("data:image/png;base64," + b64encode(result_png(results["PCA"])).decode("ascii")) if reference == "PCA vs. SVD" else entry["original_url"]
        right_url = "data:image/png;base64," + b64encode(result_png(results[right_method])).decode("ascii")
        entry["html"] = comparison_html(left, results[right_method]["reconstructed_image"], reference, components,
                                        "rgb-inspector", "Reconstrucción PCA" if reference == "PCA vs. SVD" else "Original",
                                        original_url=left_url, reconstructed_url=right_url)
        entry["html_key"] = html_key
    render_comparison_html(entry["html"])
    for method, result in results.items():
        st.download_button(f"Descargar reconstrucción {method}", data=result_png(result),
                           file_name=f"{method.lower()}_rgb_{width}x{height}_k{components}.png", mime="image/png")
    with st.expander("Cómo leer la comparación"):
        st.markdown("**PCA** centra cada canal; su retención mide varianza. **SVD** aproxima cada canal con rango k; su retención mide energía. Estas medidas no son equivalentes.")
        st.markdown("**Diferencia** muestra el error absoluto por canal; **×4** lo amplifica y limita a 255. Negro indica coincidencia. En PCA vs. SVD, el mapa compara ambas reconstrucciones.")
        st.caption("Datos del modelo cuenta valores numéricos respecto a los píxeles RGB; no representa el tamaño del PNG. La vista previa y su descarga usan la resolución indicada. Selecciona Resolución original para procesar todos los píxeles.")


inject_styles()
st.sidebar.markdown("<div class='brand-lockup'><div class='brand-mark'>◈ SISTEMA DE ANÁLISIS</div><h2>Álgebra Visual</h2><p>Un laboratorio para explorar matrices, espacios propios y PCA.</p></div>", unsafe_allow_html=True)
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
