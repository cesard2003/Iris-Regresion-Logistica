# -----------------------------------------
# ANÁLISIS DEL ARCHIVO DE MINERÍA DE DATOS
# -----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el archivo Excel
df = pd.read_excel(r"C:\Users\aguir\Downloads\S3_dataset_con_errores_Mineria_UDEC.xlsx")


# 2. Convertir la columna "Cantidad" a numérica (forzar errores a NaN)
df["Cantidad_num"] = pd.to_numeric(df["Cantidad"], errors="coerce")

# ==============================
# TABLAS
# ==============================

# Tabla 1: Estadísticas básicas de la columna Cantidad
tabla_estadisticas = df["Cantidad_num"].describe()
print("📊 Estadísticas básicas de la columna Cantidad:\n")
print(tabla_estadisticas, "\n")

# Tabla 2: Top 5 productos más vendidos
top_productos = df.groupby("Descripcion Producto")["Cantidad_num"].sum().sort_values(ascending=False).head(5)
print("📊 Top 5 productos más vendidos:\n")
print(top_productos, "\n")

# Tabla 3: Top 5 clientes con más compras
top_clientes = df.groupby("Codigo Cliente")["Cantidad_num"].sum().sort_values(ascending=False).head(5)
print("📊 Top 5 clientes con más unidades compradas:\n")
print(top_clientes, "\n")

# ==============================
# GRÁFICOS
# ==============================

# Gráfico 1: Histograma de la columna cantidad (sin outliers extremos)
df_filtrado = df[df["Cantidad_num"] < 500]  # Filtramos valores exagerados como 9999
plt.figure(figsize=(8,5))
plt.hist(df_filtrado["Cantidad_num"], bins=20, edgecolor="black")
plt.title("Distribución de la cantidad de productos comprados")
plt.xlabel("Cantidad")
plt.ylabel("Frecuencia")
plt.show()

# Gráfico 2: Barras de los 5 productos más vendidos
plt.figure(figsize=(8,5))
top_productos.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Top 5 productos más vendidos")
plt.xlabel("Producto")
plt.ylabel("Unidades vendidas")
plt.show()

# Gráfico 3: Barras de los 5 clientes con más compras
plt.figure(figsize=(8,5))
top_clientes.plot(kind="bar", color="salmon", edgecolor="black")
plt.title("Top 5 clientes con más unidades compradas")
plt.xlabel("Código Cliente")
plt.ylabel("Unidades compradas")
plt.show()

