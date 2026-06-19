import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(
    page_title="Forecast & Inventory Intelligence",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Forecast & Inventory Intelligence Framework")
st.caption("Comparación Forecast Empresa vs Forecast Propuesto + simulación de inventarios por SKU")

# =========================================================
# FUNCIONES BASE
# =========================================================
def normalizar_columnas(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    alias = {
        "sku": "product_id",
        "producto": "product_id",
        "cod_producto": "product_id",
        "demanda agrupada final": "product_id",
        "descripcion": "description",
        "descripción": "description",
        "fecha": "date",
        "mes": "date",
        "ventas": "demand_real",
        "venta": "demand_real",
        "und final": "demand_real",
        "unidades": "demand_real",
        "forecast_comercial": "forecast_company",
        "forecast empresa": "forecast_company",
        "pronostico empresa": "forecast_company",
        "pronóstico empresa": "forecast_company",
        "costo_unitario": "unit_cost",
        "costo unitario": "unit_cost",
        "precio bp 2025 sin igv": "unit_cost",
        "lead_time": "lead_time_days",
        "lead time": "lead_time_days",
        "leadtime": "lead_time_days",
        "lead time dias": "lead_time_days",
        "lead time días": "lead_time_days",
    }

    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})
    return df


def cargar_excel(uploaded_file):
    maestro = pd.read_excel(uploaded_file, sheet_name="Maestro_SKU")
    ventas = pd.read_excel(uploaded_file, sheet_name="Ventas_Historicas")
    forecast_empresa = pd.read_excel(uploaded_file, sheet_name="Forecast_Comercial")

    maestro = normalizar_columnas(maestro)
    ventas = normalizar_columnas(ventas)
    forecast_empresa = normalizar_columnas(forecast_empresa)

    maestro["product_id"] = maestro["product_id"].astype(str)
    maestro["description"] = maestro.get("description", maestro["product_id"]).astype(str)
    maestro["unit_cost"] = pd.to_numeric(maestro["unit_cost"], errors="coerce").fillna(0)
    

    ventas["date"] = pd.to_datetime(ventas["date"], errors="coerce")
    ventas["product_id"] = ventas["product_id"].astype(str)
    ventas["demand_real"] = pd.to_numeric(ventas["demand_real"], errors="coerce").fillna(0)

    forecast_empresa["date"] = pd.to_datetime(forecast_empresa["date"], errors="coerce")
    forecast_empresa["product_id"] = forecast_empresa["product_id"].astype(str)
    forecast_empresa["forecast_company"] = pd.to_numeric(
        forecast_empresa["forecast_company"],
        errors="coerce"
    ).fillna(0)

    ventas = ventas.dropna(subset=["date"])
    forecast_empresa = forecast_empresa.dropna(subset=["date"])

    # Agrupar ventas reales de diario a mensual
    ventas["month"] = ventas["date"].dt.to_period("M").dt.to_timestamp()
    ventas_mensual = (
        ventas.groupby(["product_id", "month"], as_index=False)["demand_real"]
        .sum()
        .rename(columns={"month": "date"})
    )

    # Forecast empresa mensual
    forecast_empresa["month"] = forecast_empresa["date"].dt.to_period("M").dt.to_timestamp()
    forecast_mensual = (
        forecast_empresa.groupby(["product_id", "month"], as_index=False)["forecast_company"]
        .sum()
        .rename(columns={"month": "date"})
    )

    return maestro, ventas_mensual, forecast_mensual


# =========================================================
# MÉTRICAS
# =========================================================
def wmape(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sum(np.abs(y_real - y_pred)) / max(np.sum(np.abs(y_real)), 1) * 100


def bias_pct(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sum(y_pred - y_real) / max(np.sum(y_real), 1) * 100


def calcular_error_valorizado(y_real, y_pred, costo):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    exceso = np.maximum(y_pred - y_real, 0)
    faltante = np.maximum(y_real - y_pred, 0)
    error_abs = np.abs(y_pred - y_real)

    return {
        "exceso_und": exceso.sum(),
        "faltante_und": faltante.sum(),
        "error_valorizado": np.sum(error_abs * costo),
    }


# =========================================================
# MODELOS DE PRONÓSTICO
# =========================================================
def forecast_regresion(serie):
    x = np.arange(len(serie)).reshape(-1, 1)
    modelo = LinearRegression()
    modelo.fit(x, serie)
    pred = modelo.predict(x)
    return np.maximum(0, pred)


def forecast_ses(serie):
    if len(serie) < 3:
        return np.repeat(np.mean(serie), len(serie))

    modelo = SimpleExpSmoothing(serie, initialization_method="estimated")
    ajuste = modelo.fit(optimized=True)
    pred = np.asarray(ajuste.fittedvalues)
    return np.maximum(0, pred)


def forecast_arima(serie):
    if len(serie) < 6:
        return forecast_ses(serie)

    try:
        modelo = ARIMA(serie, order=(1, 1, 1))
        ajuste = modelo.fit()
        pred = np.asarray(ajuste.fittedvalues)
        pred = np.where(np.isnan(pred), np.nanmean(serie), pred)
        return np.maximum(0, pred)
    except Exception:
        return forecast_ses(serie)


def evaluar_sku(product_id, ventas_mensual, forecast_mensual, maestro):
    ventas_sku = ventas_mensual[ventas_mensual["product_id"] == product_id].copy()
    ventas_sku = ventas_sku.sort_values("date")

    forecast_sku = forecast_mensual[forecast_mensual["product_id"] == product_id].copy()
    forecast_sku = forecast_sku.sort_values("date")

    df = ventas_sku.merge(
        forecast_sku,
        on=["product_id", "date"],
        how="inner"
    )

    if len(df) < 6:
        return None

    serie = df["demand_real"].to_numpy(dtype=float)
    y_real = df["demand_real"].to_numpy(dtype=float)

    modelos = {
        "Regresión lineal": forecast_regresion(serie),
        "Suavizamiento exponencial simple": forecast_ses(serie),
        "ARIMA": forecast_arima(serie),
    }

    costo = maestro.loc[maestro["product_id"] == product_id, "unit_cost"]
    costo = float(costo.iloc[0]) if len(costo) > 0 else 0

    resultados = []

    # Forecast empresa
    y_empresa = df["forecast_company"].to_numpy(dtype=float)
    err_emp = calcular_error_valorizado(y_real, y_empresa, costo)

    resultados.append({
        "product_id": product_id,
        "modelo": "Forecast empresa",
        "wMAPE": wmape(y_real, y_empresa),
        "Bias %": bias_pct(y_real, y_empresa),
        "Exceso und": err_emp["exceso_und"],
        "Faltante und": err_emp["faltante_und"],
        "Error valorizado S/": err_emp["error_valorizado"],
    })

    # Modelos propuestos
    for nombre, pred in modelos.items():
        err = calcular_error_valorizado(y_real, pred, costo)

        resultados.append({
            "product_id": product_id,
            "modelo": nombre,
            "wMAPE": wmape(y_real, pred),
            "Bias %": bias_pct(y_real, pred),
            "Exceso und": err["exceso_und"],
            "Faltante und": err["faltante_und"],
            "Error valorizado S/": err["error_valorizado"],
        })

    tabla = pd.DataFrame(resultados)

    propuestos = tabla[tabla["modelo"] != "Forecast empresa"].copy()
    mejor = propuestos.sort_values("wMAPE").iloc[0]

    empresa = tabla[tabla["modelo"] == "Forecast empresa"].iloc[0]

    return {
        "product_id": product_id,
        "df": df,
        "modelos": modelos,
        "tabla": tabla,
        "mejor_modelo": mejor["modelo"],
        "wmape_empresa": empresa["wMAPE"],
        "wmape_propuesto": mejor["wMAPE"],
        "bias_empresa": empresa["Bias %"],
        "bias_propuesto": mejor["Bias %"],
        "error_empresa": empresa["Error valorizado S/"],
        "error_propuesto": mejor["Error valorizado S/"],
        "ahorro": empresa["Error valorizado S/"] - mejor["Error valorizado S/"],
    }


@st.cache_data
def evaluar_todos(ventas_mensual, forecast_mensual, maestro):
    detalles = {}
    resumen = []

    productos = sorted(set(ventas_mensual["product_id"]) & set(forecast_mensual["product_id"]))

    for p in productos:
        r = evaluar_sku(p, ventas_mensual, forecast_mensual, maestro)
        if r is not None:
            detalles[p] = r
            resumen.append({
                "product_id": p,
                "mejor_modelo": r["mejor_modelo"],
                "wMAPE empresa": r["wmape_empresa"],
                "wMAPE propuesto": r["wmape_propuesto"],
                "Bias empresa %": r["bias_empresa"],
                "Bias propuesto %": r["bias_propuesto"],
                "Error empresa S/": r["error_empresa"],
                "Error propuesto S/": r["error_propuesto"],
                "Ahorro potencial S/": r["ahorro"],
            })

    return pd.DataFrame(resumen), detalles


# =========================================================
# SIMULACIÓN DE INVENTARIO
# =========================================================
@dataclass
class ParametrosInventario:
    initial_stock: int
    lead_time_months: int
    review_period_months: int
    ss_months: int
    q_fixed: int
    lot_size: int
    cost_order: float
    cost_holding: float
    cost_stockout: float


def redondear_lote(cantidad, lote):
    if cantidad <= 0:
        return 0
    lote = max(1, int(lote))
    return int(math.ceil(cantidad / lote) * lote)


def simular_producto(df_producto, politica, p: ParametrosInventario):
    df_producto = df_producto.sort_values("date").reset_index(drop=True).copy()

    stock = float(p.initial_stock)
    pipeline = {}
    resultados = []

    demanda_prom = max(0.01, df_producto["demand_forecast"].mean())

    for t, fila in df_producto.iterrows():
        llegada = pipeline.pop(t, 0)
        stock += llegada

        demanda_lt = demanda_prom * p.lead_time_months
        stock_seguridad = demanda_prom * p.ss_months
        punto_reorden = demanda_lt + stock_seguridad
        nivel_objetivo = demanda_prom * (
            p.lead_time_months + p.review_period_months + p.ss_months
        )

        posicion = stock + sum(pipeline.values())
        orden = 0

        if politica == "RS - revisión periódica":
            if t % p.review_period_months == 0:
                orden = max(0, nivel_objetivo - posicion)

        elif politica == "sS - punto de reorden y nivel máximo":
            if posicion <= punto_reorden:
                orden = max(0, nivel_objetivo - posicion)

        elif politica == "sQ - punto de reorden y cantidad fija":
            if posicion <= punto_reorden:
                orden = p.q_fixed

        orden = redondear_lote(orden, p.lot_size)

        if orden > 0:
            llegada_mes = t + p.lead_time_months
            pipeline[llegada_mes] = pipeline.get(llegada_mes, 0) + orden

        demanda_real = float(fila["demand_real"])
        venta_real = min(stock, demanda_real)
        venta_perdida = max(0, demanda_real - stock)

        stock -= venta_real

        resultados.append({
            "date": fila["date"],
            "product_id": fila["product_id"],
            "demand_real": demanda_real,
            "demand_forecast": fila["demand_forecast"],
            "inventory_level": stock,
            "inventory_position": posicion,
            "order_placed": orden,
            "arrivals": llegada,
            "sales_real": venta_real,
            "sales_lost": venta_perdida,
            "reorder_point_s": punto_reorden,
            "target_level_S": nivel_objetivo,
            "is_stockout": int(venta_perdida > 0),
        })

    return pd.DataFrame(resultados)


def calcular_kpis(df_sim, p):
    demanda_total = df_sim["demand_real"].sum()
    ventas_perdidas = df_sim["sales_lost"].sum()
    ordenes = (df_sim["order_placed"] > 0).sum()
    inventario_prom = df_sim["inventory_level"].mean()

    fill_rate = 1 - ventas_perdidas / demanda_total if demanda_total > 0 else 1
    costo_ordenar = ordenes * p.cost_order
    costo_mantener = inventario_prom * p.cost_holding
    costo_quiebre = ventas_perdidas * p.cost_stockout
    costo_total = costo_ordenar + costo_mantener + costo_quiebre

    return {
        "fill_rate": fill_rate,
        "avg_inventory": inventario_prom,
        "lost_sales_units": ventas_perdidas,
        "orders": ordenes,
        "total_cost": costo_total,
    }


def optimizar_stock_seguridad(df_producto, politica, p_base, ss_max):
    filas = []

    for ss in range(0, ss_max + 1):
        p = ParametrosInventario(
            initial_stock=p_base.initial_stock,
            lead_time_months=p_base.lead_time_months,
            review_period_months=p_base.review_period_months,
            ss_months=ss,
            q_fixed=p_base.q_fixed,
            lot_size=p_base.lot_size,
            cost_order=p_base.cost_order,
            cost_holding=p_base.cost_holding,
            cost_stockout=p_base.cost_stockout,
        )

        sim = simular_producto(df_producto, politica, p)
        kpis = calcular_kpis(sim, p)
        filas.append({"ss_months": ss, **kpis})

    return pd.DataFrame(filas)


# =========================================================
# VISUALIZACIONES
# =========================================================
def grafico_comparacion(df, pred, modelo):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["demand_real"], mode="lines+markers", name="Ventas reales"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["forecast_company"], mode="lines+markers", name="Forecast empresa"))
    fig.add_trace(go.Scatter(x=df["date"], y=pred, mode="lines+markers", name=f"Forecast propuesto - {modelo}"))

    fig.update_layout(
        title=f"Ventas reales vs forecast empresa vs {modelo}",
        xaxis_title="Mes",
        yaxis_title="Unidades",
        hovermode="x unified",
    )
    return fig


def grafico_inventario(df_sim):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=df_sim["date"], y=df_sim["inventory_level"], name="Inventario", mode="lines"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_sim["date"], y=df_sim["reorder_point_s"], name="Punto s", mode="lines", line={"dash": "dot"}), secondary_y=False)
    fig.add_trace(go.Bar(x=df_sim["date"], y=df_sim["demand_real"], name="Demanda", opacity=0.35), secondary_y=True)

    fig.update_layout(title="Simulación mensual de inventario", hovermode="x unified")
    return fig


def grafico_tradeoff(df_opt):
    mejor = df_opt.loc[df_opt["total_cost"].idxmin()]
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_opt["ss_months"], y=df_opt["total_cost"], mode="lines+markers", name="Costo total"))
    fig.add_vline(x=int(mejor["ss_months"]), line_dash="dash", annotation_text=f"Óptimo: {int(mejor['ss_months'])} meses")

    fig.update_layout(
        title="Optimización de stock de seguridad",
        xaxis_title="Meses de stock de seguridad",
        yaxis_title="Costo total",
    )
    return fig


# =========================================================
# CARGA DE DATOS
# =========================================================
st.sidebar.header("1. Carga de datos")
archivo = st.sidebar.file_uploader("Sube tu Excel", type=["xlsx", "xls"])

if archivo is None:
    st.info("Sube un Excel con hojas: Maestro_SKU, Ventas_Historicas y Forecast_Comercial.")
    st.stop()

try:
    maestro, ventas_mensual, forecast_mensual = cargar_excel(archivo)
except Exception as e:
    st.error(f"Error leyendo archivo: {e}")
    st.stop()

resumen_skus, detalles = evaluar_todos(ventas_mensual, forecast_mensual, maestro)

if resumen_skus.empty:
    st.warning("No se pudo evaluar ningún SKU. Revisa que los SKU coincidan entre ventas y forecast comercial.")
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
productos = sorted(detalles.keys())
producto_sel = st.sidebar.selectbox("Producto", productos)

st.sidebar.header("2. Simulación")
politica = st.sidebar.selectbox(
    "Política",
    [
        "RS - revisión periódica",
        "sS - punto de reorden y nivel máximo",
        "sQ - punto de reorden y cantidad fija",
    ],
)

lead_months_auto = st.sidebar.number_input(
    "Lead time general (meses)",
    min_value=1,
    value=1,
    step=1
)

initial_stock = st.sidebar.number_input("Stock inicial", min_value=0, value=100, step=10)
review_period = st.sidebar.number_input("Periodo de revisión (meses)", min_value=1, value=1, step=1)
ss_months = st.sidebar.number_input("Stock de seguridad inicial (meses)", min_value=0, value=1, step=1)
q_fixed = st.sidebar.number_input("Cantidad fija Q", min_value=1, value=100, step=10)
lot_size = st.sidebar.number_input("Tamaño de lote", min_value=1, value=1, step=1)

st.sidebar.header("3. Costos")
cost_order = st.sidebar.number_input("Costo por orden", min_value=0.0, value=200.0, step=10.0)
cost_holding = st.sidebar.number_input("Costo de mantener inventario", min_value=0.0, value=1.5, step=0.5)
cost_stockout = st.sidebar.number_input("Costo por unidad faltante", min_value=0.0, value=500.0, step=10.0)
ss_max = st.sidebar.slider("Máximo SS para optimizar", 1, 12, 6)

parametros = ParametrosInventario(
    initial_stock=int(initial_stock),
    lead_time_months=int(lead_months_auto),
    review_period_months=int(review_period),
    ss_months=int(ss_months),
    q_fixed=int(q_fixed),
    lot_size=int(lot_size),
    cost_order=float(cost_order),
    cost_holding=float(cost_holding),
    cost_stockout=float(cost_stockout),
)

# =========================================================
# CONTENIDO
# =========================================================
r = detalles[producto_sel]
modelo_ganador = r["mejor_modelo"]
pred_ganador = r["modelos"][modelo_ganador]

df_sim_base = r["df"].copy()
df_sim_base["demand_forecast"] = pred_ganador

sub_sim = simular_producto(df_sim_base, politica, parametros)
kpis = calcular_kpis(sub_sim, parametros)
sub_opt = optimizar_stock_seguridad(df_sim_base, politica, parametros, ss_max)
mejor_ss = sub_opt.loc[sub_opt["total_cost"].idxmin()]

error_empresa_total = resumen_skus["Error empresa S/"].sum()
error_propuesto_total = resumen_skus["Error propuesto S/"].sum()
ahorro_total = resumen_skus["Ahorro potencial S/"].sum()

reduccion_error = (
    (error_empresa_total - error_propuesto_total) / error_empresa_total * 100
    if error_empresa_total > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("SKUs evaluados", f"{len(resumen_skus):,.0f}")
col2.metric("Reducción error valorizado", f"{reduccion_error:.1f}%")
col3.metric("Ahorro potencial total", f"S/ {ahorro_total:,.2f}")
col4.metric("Modelo ganador SKU", modelo_ganador)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Comparación de Forecast",
    "📦 Simulación",
    "🎯 Optimización",
    "📋 Tablas",
])

with tab1:
    st.subheader("Comparación Forecast Empresa vs Forecast Propuesto")

    st.write("Resumen global por SKU")
    st.dataframe(resumen_skus, use_container_width=True, hide_index=True)

    st.write(f"Detalle del SKU: {producto_sel}")

    for modelo, pred in r["modelos"].items():
        if modelo == "Forecast empresa":
            continue

        col_graf, col_tabla = st.columns([2, 1])

        with col_graf:
            st.plotly_chart(
                grafico_comparacion(r["df"], pred, modelo),
                use_container_width=True
            )

        with col_tabla:
            tabla_modelo = r["tabla"][r["tabla"]["modelo"].isin(["Forecast empresa", modelo])]
            st.dataframe(tabla_modelo, use_container_width=True, hide_index=True)

    st.success(f"Mejor modelo para {producto_sel}: {modelo_ganador}")

with tab2:
    st.subheader("Simulación mensual de inventario")
    st.plotly_chart(grafico_inventario(sub_sim), use_container_width=True)

    kpi_df = pd.DataFrame([kpis]).T.reset_index()
    kpi_df.columns = ["Indicador", "Valor"]
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Optimización de stock de seguridad")
    st.info(
        f"Para el SKU {producto_sel}, el stock de seguridad óptimo es "
        f"{int(mejor_ss['ss_months'])} meses, con costo total aproximado de "
        f"S/ {mejor_ss['total_cost']:,.2f}."
    )
    st.plotly_chart(grafico_tradeoff(sub_opt), use_container_width=True)

with tab4:
    st.subheader("Tablas")
    st.write("Base mensual del SKU")
    st.dataframe(r["df"], use_container_width=True, hide_index=True)

    st.write("Resultados por modelo")
    st.dataframe(r["tabla"], use_container_width=True, hide_index=True)

    st.write("Simulación mensual")
    st.dataframe(sub_sim, use_container_width=True, hide_index=True)

    st.write("Optimización")
    st.dataframe(sub_opt, use_container_width=True, hide_index=True)

    csv = sub_sim.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar simulación CSV",
        data=csv,
        file_name=f"simulacion_{producto_sel}.csv",
        mime="text/csv",
    )
