import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

warnings.filterwarnings("ignore")


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Inventory Intelligence Framework",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Framework de Optimización de Inventarios")
st.caption("Pronóstico + simulación + optimización de inventarios inspirado en el caso Kroger")


# =========================================================
# FUNCIONES DE DATOS
# =========================================================
def generar_demanda_sintetica(n_productos: int = 5, dias: int = 365, seed: int = 42) -> pd.DataFrame:
    """Genera demanda diaria sintética para pruebas."""
    rng = np.random.default_rng(seed)
    fechas = pd.date_range(start="2025-01-01", periods=dias, freq="D")
    dataframes = []

    for i in range(1, n_productos + 1):
        producto = f"PROD_{i:03d}"
        base = rng.integers(30, 100)
        tendencia = rng.uniform(0.01, 0.08)
        estacionalidad = rng.uniform(5, 15)
        ruido = rng.normal(0, base * 0.20, dias)
        tiempo = np.arange(dias)

        demanda = base + tendencia * tiempo + estacionalidad * np.sin(2 * np.pi * tiempo / 7) + ruido
        demanda = np.maximum(0, np.round(demanda)).astype(int)

        dataframes.append(
            pd.DataFrame(
                {
                    "date": fechas,
                    "product_id": producto,
                    "demand_real": demanda,
                }
            )
        )

    return pd.concat(dataframes, ignore_index=True)


def leer_archivo_subido(uploaded_file) -> pd.DataFrame:
    """Lee CSV o Excel y normaliza columnas básicas."""
    nombre = uploaded_file.name.lower()

    if nombre.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif nombre.endswith(".xlsx") or nombre.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formato no soportado. Sube un archivo CSV o Excel.")

    # Normalizar nombres de columnas
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Alias frecuentes en español/inglés
    alias = {
        "fecha": "date",
        "día": "date",
        "dia": "date",
        "producto": "product_id",
        "sku": "product_id",
        "id_producto": "product_id",
        "codigo": "product_id",
        "código": "product_id",
        "demanda": "demand_real",
        "venta": "demand_real",
        "ventas": "demand_real",
        "cantidad": "demand_real",
        "unidades": "demand_real",
    }
    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

    columnas_requeridas = ["date", "product_id", "demand_real"]
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(
            "Faltan columnas obligatorias: " + ", ".join(faltantes) +
            ". Usa columnas: date, product_id, demand_real."
        )

    df = df[columnas_requeridas].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["product_id"] = df["product_id"].astype(str)
    df["demand_real"] = pd.to_numeric(df["demand_real"], errors="coerce").fillna(0)
    df["demand_real"] = df["demand_real"].clip(lower=0)
    df = df.dropna(subset=["date"])
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("El archivo no tiene datos válidos después de limpiar fechas y demanda.")

    return df


# =========================================================
# PRONÓSTICOS
# =========================================================
def forecast_regresion(serie: np.ndarray) -> np.ndarray:
    x = np.arange(len(serie)).reshape(-1, 1)
    modelo = LinearRegression()
    modelo.fit(x, serie)
    pred = modelo.predict(x)
    return np.maximum(0, pred)


def forecast_ses(serie: np.ndarray, alpha: float = 0.30) -> np.ndarray:
    if len(serie) < 3:
        return np.repeat(np.mean(serie), len(serie))

    modelo = SimpleExpSmoothing(serie, initialization_method="estimated")
    ajuste = modelo.fit(smoothing_level=alpha, optimized=False)
    pred = np.asarray(ajuste.fittedvalues)
    return np.maximum(0, pred)


def forecast_arima(serie: np.ndarray) -> np.ndarray:
    if len(serie) < 10:
        return forecast_ses(serie)

    try:
        modelo = ARIMA(serie, order=(1, 1, 1))
        ajuste = modelo.fit()
        pred = np.asarray(ajuste.fittedvalues)
        pred = np.where(np.isnan(pred), np.nanmean(serie), pred)
        return np.maximum(0, pred)
    except Exception:
        return forecast_ses(serie)


def generar_forecast(df: pd.DataFrame, metodo: str) -> pd.DataFrame:
    resultados = []

    for producto, sub in df.groupby("product_id"):
        sub = sub.sort_values("date").copy()
        serie = sub["demand_real"].to_numpy(dtype=float)

        if metodo == "Regresión lineal":
            pred = forecast_regresion(serie)
        elif metodo == "ARIMA":
            pred = forecast_arima(serie)
        else:
            pred = forecast_ses(serie)

        sub["demand_forecast"] = np.round(pred, 2)
        resultados.append(sub)

    return pd.concat(resultados, ignore_index=True)


def calcular_errores(y_real, y_pred) -> dict:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    suma_real = y_real.sum()

    if suma_real == 0:
        return {"wMAPE": 0.0, "Bias": 0.0}

    wmape = np.sum(np.abs(y_real - y_pred)) / suma_real
    bias = np.sum(y_pred - y_real) / suma_real
    return {"wMAPE": wmape, "Bias": bias}


# =========================================================
# SIMULACIÓN DE INVENTARIO
# =========================================================
@dataclass
class ParametrosInventario:
    initial_stock: int
    lead_time: int
    review_period: int
    ss_days: int
    q_fixed: int
    lot_size: int
    cost_order: float
    cost_holding: float
    cost_stockout: float


def redondear_lote(cantidad: float, lote: int) -> int:
    if cantidad <= 0:
        return 0
    lote = max(1, int(lote))
    return int(math.ceil(cantidad / lote) * lote)


def simular_producto(df_producto: pd.DataFrame, politica: str, p: ParametrosInventario) -> pd.DataFrame:
    df_producto = df_producto.sort_values("date").reset_index(drop=True).copy()
    stock_fisico = float(p.initial_stock)
    pipeline = {}  # día_llegada: cantidad
    resultados = []

    demanda_promedio = max(0.01, df_producto["demand_forecast"].mean())

    for t, fila in df_producto.iterrows():
        llegada = pipeline.pop(t, 0)
        stock_fisico += llegada

        demanda_lt = demanda_promedio * p.lead_time
        stock_seguridad = demanda_promedio * p.ss_days
        punto_reorden = demanda_lt + stock_seguridad
        nivel_objetivo = demanda_promedio * (p.lead_time + p.review_period + p.ss_days)

        posicion_inventario = stock_fisico + sum(pipeline.values())
        orden = 0

        if politica == "RS - revisión periódica":
            if t % p.review_period == 0:
                orden = max(0, nivel_objetivo - posicion_inventario)

        elif politica == "sS - punto de reorden y nivel máximo":
            if posicion_inventario <= punto_reorden:
                orden = max(0, nivel_objetivo - posicion_inventario)

        elif politica == "sQ - punto de reorden y cantidad fija":
            if posicion_inventario <= punto_reorden:
                orden = p.q_fixed

        orden = redondear_lote(orden, p.lot_size)

        if orden > 0:
            dia_llegada = t + p.lead_time
            pipeline[dia_llegada] = pipeline.get(dia_llegada, 0) + orden

        demanda_real = float(fila["demand_real"])
        venta_real = min(stock_fisico, demanda_real)
        venta_perdida = max(0, demanda_real - stock_fisico)
        stock_fisico -= venta_real

        resultados.append(
            {
                "date": fila["date"],
                "product_id": fila["product_id"],
                "demand_real": demanda_real,
                "demand_forecast": fila["demand_forecast"],
                "inventory_level": stock_fisico,
                "inventory_position": posicion_inventario,
                "order_placed": orden,
                "arrivals": llegada,
                "sales_real": venta_real,
                "sales_lost": venta_perdida,
                "reorder_point_s": punto_reorden,
                "target_level_S": nivel_objetivo,
                "is_stockout": int(venta_perdida > 0),
            }
        )

    return pd.DataFrame(resultados)


def calcular_kpis(df_sim: pd.DataFrame, p: ParametrosInventario) -> dict:
    demanda_total = df_sim["demand_real"].sum()
    ventas_perdidas = df_sim["sales_lost"].sum()
    ordenes = (df_sim["order_placed"] > 0).sum()
    inventario_promedio = df_sim["inventory_level"].mean()

    fill_rate = 1 - ventas_perdidas / demanda_total if demanda_total > 0 else 1
    costo_ordenar = ordenes * p.cost_order
    costo_mantener = inventario_promedio * p.cost_holding
    costo_quiebre = ventas_perdidas * p.cost_stockout
    costo_total = costo_ordenar + costo_mantener + costo_quiebre

    return {
        "fill_rate": fill_rate,
        "avg_inventory": inventario_promedio,
        "lost_sales_units": ventas_perdidas,
        "stockout_days": int(df_sim["is_stockout"].sum()),
        "orders": int(ordenes),
        "ordering_cost": costo_ordenar,
        "holding_cost": costo_mantener,
        "stockout_cost": costo_quiebre,
        "total_cost": costo_total,
    }


def optimizar_stock_seguridad(df_producto: pd.DataFrame, politica: str, p_base: ParametrosInventario, ss_max: int) -> pd.DataFrame:
    filas = []

    for ss in range(0, ss_max + 1):
        p = ParametrosInventario(
            initial_stock=p_base.initial_stock,
            lead_time=p_base.lead_time,
            review_period=p_base.review_period,
            ss_days=ss,
            q_fixed=p_base.q_fixed,
            lot_size=p_base.lot_size,
            cost_order=p_base.cost_order,
            cost_holding=p_base.cost_holding,
            cost_stockout=p_base.cost_stockout,
        )
        sim = simular_producto(df_producto, politica, p)
        kpis = calcular_kpis(sim, p)
        filas.append({"ss_days": ss, **kpis})

    return pd.DataFrame(filas)


# =========================================================
# VISUALIZACIONES
# =========================================================
def grafico_forecast(df_producto: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_producto["date"], y=df_producto["demand_real"], mode="lines", name="Demanda real"))
    fig.add_trace(go.Scatter(x=df_producto["date"], y=df_producto["demand_forecast"], mode="lines", name="Pronóstico"))
    fig.update_layout(title="Demanda real vs pronóstico", xaxis_title="Fecha", yaxis_title="Unidades", hovermode="x unified")
    return fig


def grafico_inventario(df_sim: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df_sim["date"], y=df_sim["inventory_level"], name="Inventario", mode="lines"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df_sim["date"], y=df_sim["reorder_point_s"], name="Punto s", mode="lines", line={"dash": "dot"}),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=df_sim["date"], y=df_sim["demand_real"], name="Demanda", opacity=0.35),
        secondary_y=True,
    )

    pedidos = df_sim[df_sim["order_placed"] > 0]
    fig.add_trace(
        go.Scatter(
            x=pedidos["date"],
            y=pedidos["order_placed"],
            name="Pedido generado",
            mode="markers",
            marker={"size": 10, "symbol": "triangle-up"},
        ),
        secondary_y=True,
    )

    fig.update_layout(title="Simulación de inventario", hovermode="x unified")
    fig.update_yaxes(title_text="Inventario", secondary_y=False)
    fig.update_yaxes(title_text="Demanda / Pedidos", secondary_y=True)
    return fig


def grafico_tradeoff(df_opt: pd.DataFrame) -> go.Figure:
    mejor = df_opt.loc[df_opt["total_cost"].idxmin()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_opt["ss_days"], y=df_opt["total_cost"], mode="lines+markers", name="Costo total"))
    fig.add_trace(go.Scatter(x=df_opt["ss_days"], y=df_opt["holding_cost"], mode="lines", name="Costo mantener"))
    fig.add_trace(go.Scatter(x=df_opt["ss_days"], y=df_opt["stockout_cost"], mode="lines", name="Costo quiebre"))
    fig.add_vline(x=int(mejor["ss_days"]), line_dash="dash", annotation_text=f"Óptimo: {int(mejor['ss_days'])} días")
    fig.update_layout(title="Trade-off de costos", xaxis_title="Días de stock de seguridad", yaxis_title="Costo", hovermode="x unified")
    return fig


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("1. Carga de datos")
modo_datos = st.sidebar.radio("Modo de datos", ["Generar datos sintéticos", "Subir CSV/Excel"])

if modo_datos == "Generar datos sintéticos":
    n_productos = st.sidebar.slider("Número de productos", 1, 50, 5)
    dias = st.sidebar.slider("Días de historial", 30, 730, 365)
    seed = st.sidebar.number_input("Semilla", min_value=1, max_value=9999, value=42)
    df_real = generar_demanda_sintetica(n_productos=n_productos, dias=dias, seed=seed)
else:
    archivo = st.sidebar.file_uploader("Sube tu archivo", type=["csv", "xlsx", "xls"])
    if archivo is None:
        st.info("Sube un CSV o Excel con columnas: date, product_id, demand_real.")
        st.stop()

    try:
        df_real = leer_archivo_subido(archivo)
    except Exception as e:
        st.error(str(e))
        st.stop()

st.sidebar.header("2. Pronóstico")
metodo_forecast = st.sidebar.selectbox("Método", ["SES", "ARIMA", "Regresión lineal"])

df_forecast = generar_forecast(df_real, metodo_forecast)
productos = sorted(df_forecast["product_id"].unique())
producto_sel = st.sidebar.selectbox("Producto a visualizar", productos)

st.sidebar.header("3. Política de inventario")
politica = st.sidebar.selectbox(
    "Política",
    [
        "RS - revisión periódica",
        "sS - punto de reorden y nivel máximo",
        "sQ - punto de reorden y cantidad fija",
    ],
)

initial_stock = st.sidebar.number_input("Stock inicial", min_value=0, value=100, step=10)
lead_time = st.sidebar.number_input("Lead time / tiempo de entrega (días)", min_value=1, value=5, step=1)
review_period = st.sidebar.number_input("Periodo de revisión R (días)", min_value=1, value=7, step=1)
ss_days = st.sidebar.number_input("Stock de seguridad inicial (días)", min_value=0, value=3, step=1)
q_fixed = st.sidebar.number_input("Cantidad fija Q", min_value=1, value=100, step=10)
lot_size = st.sidebar.number_input("Tamaño de lote / empaque", min_value=1, value=1, step=1)

st.sidebar.header("4. Costos")
cost_order = st.sidebar.number_input("Costo por orden", min_value=0.0, value=200.0, step=10.0)
cost_holding = st.sidebar.number_input("Costo de mantener inventario", min_value=0.0, value=1.5, step=0.5)
cost_stockout = st.sidebar.number_input("Costo por unidad perdida", min_value=0.0, value=500.0, step=10.0)
ss_max = st.sidebar.slider("Máximo SS para optimizar", 1, 60, 20)

parametros = ParametrosInventario(
    initial_stock=int(initial_stock),
    lead_time=int(lead_time),
    review_period=int(review_period),
    ss_days=int(ss_days),
    q_fixed=int(q_fixed),
    lot_size=int(lot_size),
    cost_order=float(cost_order),
    cost_holding=float(cost_holding),
    cost_stockout=float(cost_stockout),
)


# =========================================================
# CONTENIDO PRINCIPAL
# =========================================================
sub_forecast = df_forecast[df_forecast["product_id"] == producto_sel].copy()
sub_sim = simular_producto(sub_forecast, politica, parametros)
kpis = calcular_kpis(sub_sim, parametros)
sub_opt = optimizar_stock_seguridad(sub_forecast, politica, parametros, ss_max=ss_max)
mejor = sub_opt.loc[sub_opt["total_cost"].idxmin()]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Fill rate", f"{kpis['fill_rate']:.2%}")
col2.metric("Inventario promedio", f"{kpis['avg_inventory']:.1f}")
col3.metric("Ventas perdidas", f"{kpis['lost_sales_units']:.0f}")
col4.metric("Costo total", f"S/ {kpis['total_cost']:,.2f}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Datos y pronóstico",
    "📦 Simulación",
    "🎯 Optimización",
    "📋 Tablas",
])

with tab1:
    st.subheader("Pronóstico de demanda")

    errores = []
    for producto, sub in df_forecast.groupby("product_id"):
        err = calcular_errores(sub["demand_real"], sub["demand_forecast"])
        errores.append(
            {
                "Producto": producto,
                "wMAPE": f"{err['wMAPE']:.2%}",
                "Bias": f"{err['Bias']:.2%}",
            }
        )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(grafico_forecast(sub_forecast), use_container_width=True)
    with col_b:
        st.write("Errores por producto")
        st.dataframe(pd.DataFrame(errores), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Simulación de inventario")
    st.plotly_chart(grafico_inventario(sub_sim), use_container_width=True)

    st.write("KPIs de la simulación")
    kpi_df = pd.DataFrame([kpis]).T.reset_index()
    kpi_df.columns = ["Indicador", "Valor"]
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Optimización de stock de seguridad")
    st.info(
        f"Para el producto {producto_sel}, el stock de seguridad óptimo encontrado es "
        f"{int(mejor['ss_days'])} días, con costo total aproximado de S/ {mejor['total_cost']:,.2f}."
    )
    st.plotly_chart(grafico_tradeoff(sub_opt), use_container_width=True)

    fig_servicio = px.line(
        sub_opt,
        x="ss_days",
        y="fill_rate",
        markers=True,
        title="Nivel de servicio según días de stock de seguridad",
        labels={"ss_days": "Días de stock de seguridad", "fill_rate": "Fill rate"},
    )
    fig_servicio.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_servicio, use_container_width=True)

with tab4:
    st.subheader("Tablas de resultados")
    st.write("Datos base con pronóstico")
    st.dataframe(sub_forecast, use_container_width=True, hide_index=True)

    st.write("Simulación diaria")
    st.dataframe(sub_sim, use_container_width=True, hide_index=True)

    st.write("Resultados de optimización")
    st.dataframe(sub_opt, use_container_width=True, hide_index=True)

    csv = sub_sim.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar simulación en CSV",
        data=csv,
        file_name=f"simulacion_{producto_sel}.csv",
        mime="text/csv",
    )
