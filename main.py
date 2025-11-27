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
from typing import Dict, List

import numpy as np
import streamlit as st

# Algunos entornos con Python 3.13 tienen problemas con pandas; creamos un stub mínimo
# antes de importar Plotly para evitar errores de importación circular.
import sys
import types


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


# ---------------------------------------------------------------------------
# Estilo y layout
# ---------------------------------------------------------------------------


def set_modern_style() -> None:
    """Inyecta estilo para una UI moderna, oscura y contrastada."""
    st.set_page_config(
        page_title="Simulación PD - Autoescalado GPU",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        body { background: radial-gradient(circle at 20% 20%, #0f172a 0, #0b1220 25%, #080d19 60%, #060a12 100%); color: #e2e8f0; }
        .block-container { padding-top: 0.6rem; padding-bottom: 0.6rem; }
        .stMetric { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 8px; }
        .stPlotlyChart { padding: 0; }
        .element-container { margin-bottom: 0.5rem; }
        .badge { padding: 6px 12px; border-radius: 999px; font-weight: 600; display: inline-block; }
        .badge-green { background: #22c55e1a; color: #22c55e; border: 1px solid #22c55e55; }
        .badge-amber { background: #f59e0b1a; color: #fbbf24; border: 1px solid #f59e0b55; }
        .badge-red { background: #ef44441a; color: #f87171; border: 1px solid #ef444455; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Barra lateral y controles
# ---------------------------------------------------------------------------


def sidebar_controls() -> Dict:
    """Dibuja los controles y devuelve sus valores."""
    st.sidebar.subheader("Controlador PD")
    kp = st.sidebar.slider("Kp (proporcional)", 0.0, 5.0, 1.2, 0.1)
    kd = st.sidebar.slider("Kd (derivativo)", 0.0, 5.0, 0.6, 0.1)
    sp = st.sidebar.slider("Setpoint de GPU [%]", 30.0, 90.0, 60.0, 1.0)
    max_delta_pods = st.sidebar.slider("Máx Δpods por paso", 0.2, 2.0, 1.0, 0.1)

    st.sidebar.subheader("Modelo y simulación")
    dt = st.sidebar.slider("Paso de tiempo dt [s]", 2.0, 30.0, 10.0, 1.0)
    alpha = st.sidebar.slider("Constante dinámica α", 0.05, 1.0, 0.3, 0.05)
    initial_pods = st.sidebar.select_slider("Pods iniciales", [1, 2, 3, 4], value=1)

    st.sidebar.subheader("Carga en vivo")
    manual_load = st.sidebar.slider("Carga actual [requests/min]", 0, 10000, 400, 10)
    quick = st.sidebar.radio(
        "Atajos de carga",
        ["Manual", "Baja (150)", "Media (700)", "Alta (1400)", "Pico (1700)"],
        horizontal=False,
        index=0,
    )
    if quick != "Manual":
        manual_load = {"Baja (150)": 150, "Media (700)": 700, "Alta (1400)": 1400, "Pico (1700)": 1700}[quick]

    st.sidebar.subheader("Visualización")
    show_load = st.sidebar.checkbox("Mostrar gráfica de carga", value=False)

    return {
        "kp": kp,
        "kd": kd,
        "sp": sp,
        "dt": dt,
        "alpha": alpha,
        "initial_pods": int(initial_pods),
        "manual_load": float(manual_load),
        "show_load": show_load,
        "max_delta_pods": float(max_delta_pods),
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
        "pods": [params["initial_pods"]],
        "errors": [0.0],
        "controls": [0.0],
        "derivatives": [0.0],
        "delta_controls": [0.0],
        "loads": [params["manual_load"]],
        "last_error": 0.0,
        "current_pods": params["initial_pods"],
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


# ---------------------------------------------------------------------------
# Umbrales de control (delta de pods según magnitud del error)
# ---------------------------------------------------------------------------


def threshold_delta(value: float) -> int:
    """
    Devuelve un delta entero de pods según la magnitud del control (variable continua).
    Umbrales inspirados en el ejemplo aportado: pequeños valores no cambian pods,
    valores medios suman/restan 1, valores grandes (cerca del máximo) suman/restan 2.
    """
    abs_val = abs(value)
    if abs_val < 0.2:
        return 0
    elif abs_val < 0.8:
        return 1 if value > 0 else -1
    else:
        return 2 if value > 0 else -2


# ---------------------------------------------------------------------------
# Dinámica del proceso y control (todo en funciones)
# ---------------------------------------------------------------------------


def pd_step(uso_gpu_actual: float, load: float, params: Dict, sim_state: Dict) -> Dict[str, float]:
    """Calcula la acción PD y ajusta pods solo cuando el %GPU cae en bandas de error."""
    sp = params["sp"]
    band_low_max = max(sp - 15, 0)
    band_low_min = max(sp - 25, 0)
    band_high_min = min(sp + 15, 100)
    band_high_max = min(sp + 25, 100)

    # Control PD clásico (solo para observabilidad; no mueve pods fuera de bandas)
    error = uso_gpu_actual - sp
    derivative = (error - sim_state["last_error"]) / params["dt"]
    control = params["kp"] * error + params["kd"] * derivative
    control = float(np.clip(control, -params["max_delta_pods"], params["max_delta_pods"]))

    pods = sim_state["current_pods"]
    target_pods = int(np.clip(np.ceil(load / LOAD_AT_SP_PER_POD), 1, 4))
    delta_from_control = threshold_delta(control)

    # Ajuste únicamente si el %GPU está dentro de las bandas de error, usando la variable control.
    if band_high_min <= uso_gpu_actual <= band_high_max and control > 0:
        pods = max(pods, target_pods)
        if delta_from_control != 0:
            pods = min(pods + delta_from_control, 4)
    elif band_low_min <= uso_gpu_actual <= band_low_max and control < 0:
        pods = max(pods, target_pods)
        if delta_from_control != 0:
            pods = max(target_pods, pods + delta_from_control, 1)

    sim_state["last_error"] = error
    sim_state["current_pods"] = pods
    # control se devuelve solo como referencia/observabilidad en la UI.
    return {
        "pods": pods,
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
        "pods_avg": pods_arr.mean(),
        "pct_within_band": pct_within,
        "pods_min_allowed": 1,
        "pods_max_allowed": 4,
    }


def plot_gpu(sim: Dict, sp: float) -> go.Figure:
    band_low_min = max(sp - 25, 0)
    band_low_max = max(sp - 15, 0)
    band_high_min = min(sp + 15, 100)
    band_high_max = min(sp + 25, 100)

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
            name="Banda aceptable (baja)",
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
            name="Banda aceptable (alta)",
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
    fig.update_layout(
        title="% de uso de GPU vs tiempo",
        xaxis_title="Tiempo [min]",
        yaxis_title="% de uso de GPU",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision="gpu-figure",
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
        template="plotly_dark",
        yaxis=dict(dtick=1, range=[0.8, 4.2]),
        hovermode="x unified",
        uirevision="pods-figure",
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
        template="plotly_dark",
        hovermode="x unified",
        uirevision="load-figure",
    )
    return fig


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------


def render_header() -> None:
    st.title("Autoescalado de pods GPU")
    st.caption(
        "Microservicio de validación de imágenes sobre GPU. "
        "El sistema de control regula la cantidad de pods para mantener ~60 % de uso de GPU."
    )
    st.markdown("&nbsp;")


def render_metrics(sim: Dict, sp: float) -> None:
    kpis = compute_kpis(sim, sp)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GPU final [%]", f"{kpis['gpu_final']:.1f}")
    col2.metric("Pods actuales", f"{kpis['pods_current']:.0f}")
    col3.metric("Pods promedio", f"{kpis['pods_avg']:.2f}")
    col4.metric("Pods permitidos", f"{kpis['pods_min_allowed']} - {kpis['pods_max_allowed']}")
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
    cols = st.columns(4)
    cols[0].metric("Error [%GPU]", f"{err:.2f}")
    cols[1].metric("Derivativo", f"{deriv:.2f}")
    cols[2].metric("Control (PD)", f"{control:.2f}")
    cols[3].metric("Delta pods", f"{delta:+.0f}")


# ---------------------------------------------------------------------------
# Bucle principal de la app
# ---------------------------------------------------------------------------


def main() -> None:
    set_modern_style()
    render_header()
    params = sidebar_controls()
    ensure_state(params)

    sim = st.session_state.sim

    # Layout compacto en dos filas: KPIs arriba, gráficos debajo.
    top_left, top_right = st.columns([0.55, 0.45])
    pod_kpi_ph = top_left.container()
    ctrl_kpi_ph = top_left.container()
    metrics_ph = top_right.container()
    charts_left, charts_right = st.columns(2)
    gpu_ph = charts_left.container()
    pods_ph = charts_right.container()
    load_ph = st.container()

    # Controles de ejecución con animación suave
    st.sidebar.subheader("Ejecución en vivo")
    step_delay = st.sidebar.slider("Intervalo entre pasos (s)", 0.05, 2.0, 0.4, 0.05)
    col_run1, col_run2, col_run3 = st.sidebar.columns(3)
    start_stop = col_run1.button("Pausar" if sim["running"] else "Iniciar")
    step_once = col_run2.button("Paso +1")
    reset = col_run3.button("Reiniciar")

    def render_all() -> None:
        with pod_kpi_ph:
            render_pod_kpis(sim)
        with ctrl_kpi_ph:
            render_control_kpis(sim)
        with metrics_ph:
            render_metrics(sim, params["sp"])
        with gpu_ph:
            st.plotly_chart(plot_gpu(sim, params["sp"]), use_container_width=True, key="gpu_chart")
        with pods_ph:
            st.plotly_chart(plot_pods(sim), use_container_width=True, key="pods_chart")
        if params["show_load"]:
            with load_ph:
                st.plotly_chart(plot_load(sim), use_container_width=True, key="load_chart")
        else:
            load_ph.empty()

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

    st.markdown(
        "Con **Iniciar/Pausar** la simulación corre en continuo; con **Paso +1** avanzás discretamente. "
        "Ajustá la carga en vivo para ver cuándo escala a más pods."
    )


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
