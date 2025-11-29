"""
Simulación didáctica en tiempo discreto de un lazo PD para autoescalado de pods GPU.

Se ejecuta con:
    streamlit run main.py
o simplemente:
    python main.py

El código está escrito únicamente con funciones (sin clases) y mantiene el estado en
`st.session_state` para poder correr la simulación de manera continua, ajustar la carga
en vivo y ver los gráficos actualizarse en tiempo real.
"""

import time
import base64
from pathlib import Path
from typing import Dict, List

import numpy as np
import streamlit as st
from PIL import Image

# Algunos entornos con Python 3.13 tienen problemas con pandas; creamos un stub mínimo
# antes de importar Plotly para evitar errores de importación circular.
import sys
import types


INITIAL = 1
def _ensure_pandas_stub():
    try:
        import pandas as pd  # type: ignore

        if hasattr(pd, "Series") and hasattr(pd, "Index"):
            return pd
    except Exception:
        pass

    pd_stub = types.ModuleType("pandas")
    pd_stub.Series = tuple  # type: ignore[attr-defined]
    pd_stub.Index = tuple  # type: ignore[attr-defined]
    pd_stub.DataFrame = dict  # type: ignore[attr-defined]
    sys.modules["pandas"] = pd_stub
    return pd_stub


_ensure_pandas_stub()

import plotly.express as px
import plotly.graph_objects as go

# Capacidad por pod: 1000 rpm equivalen al 60 % de GPU -> al 100 % son ~1666.7 rpm.
CAPACITY_PER_POD = 1000.0 / 0.6
# Carga que corresponde al 60 % de GPU por pod (valor de referencia)
LOAD_AT_SP_PER_POD = 1000.0
LOGO_PATH = Path("resources/utn cuadrado.jpg")


# ---------------------------------------------------------------------------
# Estilo y layout
# ---------------------------------------------------------------------------


def set_modern_style() -> None:
    """Inyecta estilo para una UI moderna, oscura y contrastada."""
    logo_icon = Image.open(LOGO_PATH) if LOGO_PATH.exists() else "🚀"
    st.set_page_config(
        page_title="Simulación PD - Autoescalado GPU",
        page_icon=logo_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --bg1: #f5eee7;   /* crema */
            --bg2: #e8ddd5;   /* crema suave */
            --fg: #1b0f0a;    /* negro cálido */
            --accent: #c30032;/* rojo UTN */
            --muted: #8b6a60; /* marrón suave */
            --panel: #0e0c0c; /* panel negro */
            --panel-border: #2b2b2b;
        }
        body { background: linear-gradient(135deg, var(--bg1), var(--bg2)); color: var(--fg); }
        .block-container { padding-top: 1.2rem; padding-bottom: 0.2rem; }
        .stMetric { background: rgba(0,0,0,0.04); border-radius: 10px; padding: 6px; border: 1px solid rgba(0,0,0,0.05); color: var(--fg); }
        .stPlotlyChart { padding: 0; background: #ffffff; border-radius: 10px; }
        .element-container { margin-bottom: 0.35rem; }
        h1, h2, h3, h4, h5, h6 { color: var(--fg); }
        .header-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 0.2rem; }
        .header-left { display: flex; flex-direction: column; }
        .logo-title { font-weight: 800; letter-spacing: 0.04em; color: var(--accent); font-size: 2.8rem; line-height: 1.1; text-align: left; margin-top: 12px; margin-bottom: 2px; }
        .logo-sub { color: #8b6a60; font-size: 1rem; }
        .logo-img { height: 110px; margin-right: 24px; margin-top: 6px; }
        /* Sidebar en negro */
        section[data-testid="stSidebar"] { background: var(--panel); color: #f1e8e3; overflow: hidden !important; }
        section[data-testid="stSidebar"] .stSlider label, 
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stNumberInput label {
            color: #f1e8e3;
            font-size: 1.4rem;
            font-weight: 700;
        }
        section[data-testid="stSidebar"] .css-1aumxhk, /* slider label */
        section[data-testid="stSidebar"] .st-bx { color: #f1e8e3; }
        section[data-testid="stSidebar"] .st-c8 { color: #f1e8e3; }
        section[data-testid="stSidebar"] .stSlider > div > div > div { background: rgba(255,255,255,0.1); }
        section[data-testid="stSidebar"] .stSlider > div > div > div span { background: var(--accent); }
        section[data-testid="stSidebar"] .stSlider > div > div > div[data-baseweb="slider"] > div { background: rgba(255,255,255,0.2); }
        section[data-testid="stSidebar"] .stButton>button { background: var(--accent); color: #fff; border: 1px solid var(--accent); }
        section[data-testid="stSidebar"] .stButton>button:hover { background: #a00029; border-color: #a00029; }
        section[data-testid="stSidebar"] .block-container { padding: 0.8rem 0.9rem 1.2rem 0.9rem; overflow: visible !important; }
        section[data-testid="stSidebar"] > div:first-child { height: 100vh; overflow: hidden !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def logo_base64() -> str:
    """Codifica el logo de /resources en base64 para usar inline."""
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Barra lateral y controles
# ---------------------------------------------------------------------------


def sidebar_controls() -> Dict:
    """Dibuja los controles y devuelve sus valores."""
    st.sidebar.subheader("Controlador PD")
    kp = st.sidebar.slider("Kp (proporcional)", 0.0, 3.0, 1.0, 0.1)
    kd = st.sidebar.slider("Kd (derivativo)", 0.0, 1.0, 0.0, 0.01)
    sp = st.sidebar.slider("Setpoint de GPU [%]", 30.0, 90.0, 60.0, 1.0)
    st.sidebar.subheader("Modelo y simulación")
    dt = 30.0  # fijo en 30 s
    alpha = 0.25  # fijo, no editable
    initial_pods = 1  # fijo en 1

    st.sidebar.subheader("Carga en vivo")
    manual_load = st.sidebar.slider("Carga actual [requests/min]", 0, 10000, 1000, 10)

    return {
        "kp": kp,
        "kd": kd,
        "sp": sp,
        "dt": dt,
        "alpha": alpha,
        "initial_pods": int(initial_pods),
        "manual_load": float(manual_load),
    }


# ---------------------------------------------------------------------------
# Estado de la simulación (todo via session_state)
# ---------------------------------------------------------------------------


def init_state(params: Dict) -> None:
    """Inicializa el estado de la simulación."""
    st.session_state.sim = {
        "time_min": [0.0],
        "gpu": [0.0],
        "gpu_static": [0.0],
        "pods": [1],
        "errors": [0.0],
        "controls": [0.0],
        "derivatives": [0.0],
        "delta_controls": [0.0],
        "loads": [params["manual_load"]],
        "last_error": 0.0,
        "current_pods": 1,
        "running": False,
    }


def reset_simulation(params: Dict) -> None:
    """Reinicia los vectores y estado interno."""
    init_state(params)
    st.toast("Simulación reiniciada", icon="🔄")


def ensure_state(params: Dict) -> None:
    """Crea el estado inicial si aún no existe."""
    if "sim" not in st.session_state:
        init_state(params)


def find_settling_index(time_arr: np.ndarray, gpu_arr: np.ndarray, sp: float) -> int | None:
    """
    Detecta de forma simple el fin del transitorio: primer instante donde
    el error se mantiene dentro de ±2 % y la variación es pequeña por una ventana.
    """
    tol = 2.0
    window = 5  # muestras consecutivas
    slope_thresh = 0.5
    for i in range(len(gpu_arr) - window):
        segment = gpu_arr[i : i + window]
        if np.all(np.abs(segment - sp) <= tol):
            # variación pequeña dentro de la ventana
            if np.max(np.abs(np.diff(segment))) <= slope_thresh:
                return i
    return None


# ---------------------------------------------------------------------------
# Umbrales de control (delta de pods según magnitud del error)
# ---------------------------------------------------------------------------

if "INITIAL" not in st.session_state:
    st.session_state.INITIAL = 1


def umbrales(error):
    abs_error = abs(error)
    print("ABS ERROR: ",abs_error)
    print("initial: ",st.session_state.INITIAL)
    if st.session_state.INITIAL == 1: 
        st.session_state.INITIAL = 0
        return 0 
    if abs_error < 15:             # 15% GPU
        return 0
    elif 15 <= abs_error < 20:     # 15 - 20% GPU
        return -1 if error < 0 else 1
    elif 20 <= abs_error < 25:     # 10 - 25% GPU
        return -2 if error < 0 else 2
    else:
        return -3 if error < 0 else 3
        
def threshold_delta(value: float) -> int:
    """
    Devuelve un delta entero de pods según la magnitud del control (variable continua).
    Umbrales: pequeños valores no cambian pods, valores medios suman/restan 1,
    altos suman/restan 2 y muy altos permiten hasta ±3 pods.
    """
    abs_val = abs(value)
    if abs_val < 0.2:
        return 0
    elif abs_val < 1.0:
        return 1 if value > 0 else -1
    elif abs_val < 2.0:
        return 2 if value > 0 else -2
    else:
        return 3 if value > 0 else -3


def threshold_delta_derivative(value: float) -> int:
    """
    Ajuste adicional por derivativo: si el cambio es brusco, suma/resta 1 pod.
    """
    abs_val = abs(value)
    if abs_val < 0.5:
        return 0
    return 1 if value > 0 else -1


# ---------------------------------------------------------------------------
# Dinámica del proceso y control (todo en funciones)
# ---------------------------------------------------------------------------


def pd_step(uso_gpu_actual: float, load: float, params: Dict, sim_state: Dict) -> Dict[str, float]:
    """
    Control PD con acción solo dentro de las bandas de error:
    - control = Kp*error + Kd*deriv
    - convierte control a delta entero (threshold)
    - si %GPU entra en banda alta/baja, aplica delta; fuera de banda mantiene pods salvo mínimo por carga
    """
    sp = params["sp"]
    band_low_min = max(sp - 25, 0)
    band_low_max = max(sp - 15, 0)
    band_high_min = min(sp + 15, 100)
    band_high_max = min(sp + 25, 100)
    print("GPU ACT ES: ",uso_gpu_actual)
    print("setpoint es: ",sp)
    error = uso_gpu_actual - sp  # si GPU > SP, el control tiende a agregar pods
    print("ERROR ES: ",error)
    derivative = (error - sim_state["last_error"]) / params["dt"]
    umbral_error = umbrales(error)
    control = params["kp"] * umbral_error + params["kd"] * derivative
    print("KP ES: ", params["kp"])
    print("umbral es: ", umbral_error)
    print("EL CONTROL ES: ",control)

    pods_prev = sim_state["current_pods"]
    new_pods = np.ceil(pods_prev+control)
    limited_new_pods = max(1, min(4, new_pods))
    """ pods = pods_prev
    delta_from_control = 0
    target_pods = int(np.clip(np.ceil(load / LOAD_AT_SP_PER_POD), 1, 4))
    
    delta = threshold_delta(control) + threshold_delta_derivative(derivative)
    
    delta = int(np.clip(delta, -3, 3))  # permite saltos hasta ±3 pods
    delta_abs = abs(delta)

    # Solo actuamos si estamos dentro de las bandas de error superiores/inferiores.
    if band_high_min <= uso_gpu_actual <= band_high_max and control > 0:
        # Sube pods pero sin pasar del objetivo por carga.
        faltante = max(0, target_pods - pods_prev)
        inc = min(delta_abs, faltante) if faltante > 0 else 0
        pods = int(np.clip(pods_prev + inc, 1, 4))
        delta_from_control = pods - pods_prev
    elif band_low_min <= uso_gpu_actual <= band_low_max and control < 0:
        # Baja pods pero no por debajo de lo que pide la carga.
        exceso = max(0, pods_prev - target_pods)
        dec = min(delta_abs, exceso) if exceso > 0 else 0
        pods = int(np.clip(pods_prev - dec, 1, 4))
        delta_from_control = pods - pods_prev
    else:
        delta_from_control = 0 """
    delta_from_control = abs(limited_new_pods-pods_prev)
    sim_state["last_error"] = error
    sim_state["current_pods"] = limited_new_pods
    return {
        "pods": limited_new_pods,
        "error": error,
        "control": control,
        "derivative": derivative,
        "delta_from_control": delta_from_control,
    }


def plant_step(load: float, pods: int, gpu_prev: float, params: Dict) -> Dict[str, float]:
    """Aplica el modelo de primer orden del % de uso de GPU."""
    pods_safe = max(pods, 1)
    gpu_static = min(100.0, (load / (pods_safe * CAPACITY_PER_POD)) * 100.0)
    gpu_next = gpu_prev + params["alpha"] * (gpu_static - gpu_prev)
    return {"gpu": gpu_next, "gpu_static": gpu_static}


def advance_simulation(params: Dict, load_value: float, steps: int = 1) -> None:
    """Avanza la simulación una o varias muestras y guarda en session_state."""
    sim = st.session_state.sim
    for _ in range(steps):
        control_out = pd_step(sim["gpu"][-1], load_value, params, sim)
        plant_out = plant_step(load_value, control_out["pods"], sim["gpu"][-1], params)

        sim["time_min"].append(sim["time_min"][-1] + params["dt"] / 60.0)
        sim["gpu"].append(plant_out["gpu"])
        sim["gpu_static"].append(plant_out["gpu_static"])
        sim["pods"].append(control_out["pods"])
        sim["errors"].append(control_out["error"])
        sim["controls"].append(control_out["control"])
        sim["derivatives"].append(control_out["derivative"])
        sim["delta_controls"].append(control_out["delta_from_control"])
        sim["loads"].append(load_value)


# ---------------------------------------------------------------------------
# Gráficos y KPIs
# ---------------------------------------------------------------------------


def badge_for_kpi(pct_within_band: float) -> str:
    """Placeholder; no se muestra mensaje de semáforo."""
    return ""


def compute_kpis(sim: Dict, sp: float) -> Dict[str, float]:
    """Calcula indicadores rápidos para la UI."""
    gpu_arr = np.array(sim["gpu"])
    pods_arr = np.array(sim["pods"])
    within_band = (gpu_arr >= sp - 10) & (gpu_arr <= sp + 10)
    pct_within = 100.0 * np.mean(within_band)
    return {
        "gpu_final": gpu_arr[-1],
        "pods_current": pods_arr[-1],
        "pods_max": pods_arr.max(),
        "pct_within_band": pct_within,
        "pods_min_allowed": 1,
        "pods_max_allowed": 4,
    }


def plot_gpu(sim: Dict, sp: float) -> go.Figure:
    band_low_min = max(sp - 25, 0)
    band_low_max = max(sp - 15, 0)
    band_high_min = min(sp + 15, 100)
    band_high_max = min(sp + 25, 100)
    settle_idx = find_settling_index(np.array(sim["time_min"]), np.array(sim["gpu"]), sp)

    fig = go.Figure()
    # Banda inferior (relleno)
    fig.add_trace(
        go.Scatter(
            x=sim["time_min"],
            y=[band_low_min] * len(sim["time_min"]),
            mode="lines",
            line=dict(color="rgba(74, 222, 128, 0.0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sim["time_min"],
            y=[band_low_max] * len(sim["time_min"]),
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(74, 222, 128, 0.15)",
            line=dict(color="rgba(74, 222, 128, 0.0)"),
            name="Banda de error (baja)",
            hoverinfo="skip",
        )
    )
    # Banda superior (relleno)
    fig.add_trace(
        go.Scatter(
            x=sim["time_min"],
            y=[band_high_min] * len(sim["time_min"]),
            mode="lines",
            line=dict(color="rgba(249, 115, 22, 0.0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sim["time_min"],
            y=[band_high_max] * len(sim["time_min"]),
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(249, 115, 22, 0.15)",
            line=dict(color="rgba(249, 115, 22, 0.0)"),
            name="Banda de error (alta)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sim["time_min"],
            y=sim["gpu"],
            mode="lines",
            name="%GPU medido",
            line=dict(color="#38bdf8", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sim["time_min"],
            y=[sp] * len(sim["time_min"]),
            mode="lines",
            name="Setpoint",
            line=dict(color="#f97316", width=2, dash="dash"),
        )
    )
    if settle_idx is not None:
        t_settle = sim["time_min"][settle_idx]
        fig.add_vline(
            x=t_settle,
            line=dict(color="#c30032", width=2, dash="dot"),
            annotation_text="Fin transitorio",
            annotation_position="top right",
            annotation_font_color="#c30032",
        )
    fig.update_layout(
        title="% de uso de GPU vs tiempo",
        xaxis_title="Tiempo [min]",
        yaxis_title="% de uso de GPU",
        template=None,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision="gpu-figure",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    fig.update_yaxes(range=[0, 110])
    return fig


def plot_pods(sim: Dict) -> go.Figure:
    fig = px.line(
        x=sim["time_min"],
        y=sim["pods"],
        line_shape="hv",
        labels={"x": "Tiempo [min]", "y": "Pods GPU"},
    )
    fig.update_traces(line=dict(color="#a78bfa", width=3), name="Pods")
    fig.update_layout(
        title="Evolución de pods GPU",
        template=None,
        yaxis=dict(dtick=1, range=[0.8, 4.2]),
        hovermode="x unified",
        uirevision="pods-figure",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    return fig


def plot_load(sim: Dict) -> go.Figure:
    fig = px.area(
        x=sim["time_min"],
        y=sim["loads"],
        labels={"x": "Tiempo [min]", "y": "Requests por minuto"},
    )
    fig.update_traces(line=dict(color="#34d399"), fillcolor="rgba(52,211,153,0.25)", name="Carga")
    fig.update_layout(
        title="Perfil de carga en tiempo real",
        template=None,
        hovermode="x unified",
        uirevision="load-figure",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    return fig


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------


def render_header() -> None:
    encoded = logo_base64()
    img_tag = (
        f'<img src="data:image/jpeg;base64,{encoded}" alt="UTN.BA" class="logo-img">'
        if encoded
        else ""
    )
    st.markdown(
        f"""
        <div class="header-row">
            <div class="header-left">
                <div class="logo-title">Autoescalado de pods GPU</div>
            </div>
            {img_tag}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(sim: Dict, sp: float) -> None:
    kpis = compute_kpis(sim, sp)
    col1, col2, col3 = st.columns(3)
    col1.metric("GPU final [%]", f"{kpis['gpu_final']:.1f}")
    col2.metric("Pods actuales", f"{kpis['pods_current']:.0f}")
    col3.metric("Pods permitidos", f"{kpis['pods_min_allowed']} - {kpis['pods_max_allowed']}")
    # Se omite el mensaje de banda para evitar textos de alerta.


def render_pod_kpis(sim: Dict) -> None:
    """Muestra rpm atendidas por pod en el último instante."""
    pods_now = max(int(sim["pods"][-1]), 1)
    load_now = float(sim["loads"][-1])
    rpm_per_pod = load_now / pods_now
    cols = st.columns(4)
    for idx in range(4):
        value = rpm_per_pod if idx < pods_now else 0.0
        cols[idx].metric(
            f"Pod {idx + 1} rpm",
            f"{value:.0f}",
            help=(
                f"Atiende ~{value:.0f} rpm. Capacidad máx. por pod: "
                f"{CAPACITY_PER_POD:.0f} rpm (~60 % a 1000 rpm)."
            ),
        )


def render_control_kpis(sim: Dict) -> None:
    """Muestra valores instantáneos del lazo: error, derivativo, control y delta aplicado."""
    err = sim["errors"][-1]
    deriv = sim["derivatives"][-1]
    control = sim["controls"][-1]
    delta = sim["delta_controls"][-1]
    # cols = st.columns(3)
    # cols[0].metric("Error [%GPU]", f"{err:.2f}")
    # cols[1].metric("Derivativo", f"{deriv:.2f}")
    # cols[2].metric("Control (PD)", f"{control:.2f}")


def plot_error(sim: Dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sim["time_min"],
            y=sim["errors"],
            mode="lines+markers",
            name="Error",
            line=dict(color="#c30032", width=2),
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        title="Evolución del error (SP - %GPU)",
        xaxis_title="Tiempo [min]",
        yaxis_title="Error [%GPU]",
        template=None,
        hovermode="x unified",
        uirevision="error-figure",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=220,
    )
    return fig


def plot_load(sim: Dict) -> go.Figure:
    fig = px.line(
        x=sim["time_min"],
        y=sim["loads"],
        labels={"x": "Tiempo [min]", "y": "Requests/min"},
    )
    fig.update_traces(line=dict(color="#34d399", width=2), name="Carga rpm")
    fig.update_layout(
        title="Evolución de la carga (rpm)",
        template=None,
        hovermode="x unified",
        uirevision="load-figure",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=220,
    )
    return fig


# ---------------------------------------------------------------------------
# Bucle principal de la app
# ---------------------------------------------------------------------------


def main() -> None:
    set_modern_style()
    render_header()
    params = sidebar_controls()
    ensure_state(params)

    sim = st.session_state.sim

    # Layout compacto en tres filas: KPIs arriba, gráficos en medio, cargas/errores abajo.
    top_left, top_right = st.columns([0.6, 0.4])
    pod_kpi_ph = top_left.container()
    metrics_ph = top_right.container()
    charts_left, charts_right = st.columns(2)
    gpu_ph = charts_left.container()
    pods_ph = charts_right.container()
    bottom_left, bottom_right = st.columns([0.5, 0.5])

    # Controles de ejecución en vivo (intervalo fijo de 0.6 segundos)
    st.sidebar.subheader("Ejecución en vivo")
    step_delay = 0.6
    col_run1, col_run2, col_run3 = st.sidebar.columns(3)
    start_stop = col_run1.button("Pausar" if sim["running"] else "Iniciar")
    step_once = col_run2.button("Paso +1")
    reset = col_run3.button("Reiniciar")

    def render_all() -> None:
        with pod_kpi_ph:
            render_pod_kpis(sim)
        with metrics_ph:
            render_metrics(sim, params["sp"])
        with gpu_ph:
            st.plotly_chart(
                plot_gpu(sim, params["sp"]),
                use_container_width=True,
                key="gpu_chart",
                config={"displayModeBar": False, "staticPlot": False},
            )
        with pods_ph:
            st.plotly_chart(
                plot_pods(sim),
                use_container_width=True,
                key="pods_chart",
                config={"displayModeBar": False, "staticPlot": False},
            )
        with bottom_left:
            st.plotly_chart(
                plot_load(sim),
                use_container_width=True,
                key="load_chart",
                config={"displayModeBar": False, "staticPlot": False},
            )
        with bottom_right:
            st.plotly_chart(
                plot_error(sim),
                use_container_width=True,
                key="error_chart",
                config={"displayModeBar": False, "staticPlot": False},
            )
            render_control_kpis(sim)

    if reset:
        reset_simulation(params)
        sim = st.session_state.sim
        render_all()
        st.stop()

    if start_stop:
        sim["running"] = not sim["running"]

    if step_once:
        advance_simulation(params, params["manual_load"], steps=1)

    # Si está corriendo, avanza un paso por loop.
    if sim["running"]:
        advance_simulation(params, params["manual_load"], steps=1)

    # Render inicial
    render_all()

    # Si sigue corriendo, espera y fuerza un rerun para el siguiente paso.
    if sim["running"]:
        time.sleep(step_delay)
        st.rerun()


if __name__ == "__main__":
    # Permite ejecutar con `python main.py` lanzando automáticamente Streamlit.
    import sys

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        ctx_exists = False
    else:
        ctx_exists = get_script_run_ctx() is not None

    if ctx_exists:
        main()
    else:
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
