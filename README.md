# Inventory Intelligence Framework

Aplicación en Streamlit para pronóstico, simulación y optimización de inventarios.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Ejecutar en GitHub Codespaces

```bash
python -m pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Formato de archivo de carga

Puedes subir CSV o Excel con estas columnas:

- `date`
- `product_id`
- `demand_real`

También acepta alias como `fecha`, `producto`, `sku`, `demanda`, `ventas`, `cantidad`.
