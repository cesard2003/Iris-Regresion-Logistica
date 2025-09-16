# ============================================================================
# Universidad de Cundinamarca
# Ingeniería de Sistemas
# Asignatura: Machine Learning
# Actividad: Clasificación de Plantas (Dataset IRIS) usando Regresión Lineal
# Autor: Cesar Aguirre Hurtado
# Fecha: Septiembre 2025
# ============================================================================

# ---------------------------
# Importación de librerías
# ---------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# Semilla global para reproducibilidad
np.random.seed(42)

os.makedirs("imagenes", exist_ok=True)



# ---------------------------
# 1. Carga del dataset
# ---------------------------
iris = load_iris()
X = iris.data      # Variables independientes
y = iris.target    # Variable dependiente (clase): 0=Setosa, 1=Versicolor, 2=Virginica

caracteristicas_es = [
    "Largo del sépalo (cm)",
    "Ancho del sépalo (cm)",
    "Largo del pétalo (cm)",
    "Ancho del pétalo (cm)"
]

clases_es = ["Setosa", "Versicolor", "Virginica"]

print("Características del dataset:", caracteristicas_es)
print("Clases:", clases_es)
print("Tamaño del dataset:", X.shape)

# ---------------------------
# 2. División de datos
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\nTamaño de entrenamiento:", X_train.shape[0])
print("Tamaño de prueba:", X_test.shape[0])

# ---------------------------
# 3. Definir y entrenar modelo
# ---------------------------
modelo = LinearRegression()
modelo.fit(X_train, y_train)

print("\nCoeficientes del modelo (importancia de cada característica):")
for feature, coef in zip(caracteristicas_es, modelo.coef_):
    print(f"{feature}: {coef:.4f}")

print("Intercepto (bias):", modelo.intercept_)

# ---------------------------
# 4. Predicciones
# ---------------------------
y_pred_continuo = modelo.predict(X_test)

# Nota: Regresión Lineal da valores continuos → se redondean a clases (0,1,2)
y_pred = np.round(y_pred_continuo).astype(int)
y_pred = np.clip(y_pred, 0, 2)  # asegurar que esté entre 0 y 2

# ---------------------------
# 5. Evaluación
# ---------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nExactitud del modelo:", round(accuracy, 3))

print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=clases_es))

# ---------------------------
# 6. Validación cruzada (opcional)
# ---------------------------
scores = cross_val_score(modelo, X, y, cv=5, scoring="r2")
print("Resultados de validación cruzada (R² en cada fold):", scores)
print("Promedio R²:", scores.mean())

# ---------------------------
# 7. Visualización
# ---------------------------

# 7.1. Matriz de confusión
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=clases_es, yticklabels=clases_es)
plt.xlabel("Clase Predicha")
plt.ylabel("Clase Real")
plt.title("Matriz de Confusión - Regresión Lineal (redondeada)")
plt.savefig("imagenes/matriz_confusion.png")
plt.show()


# 7.2. Comparación entre valores reales, predicciones continuas y redondeadas
plt.figure(figsize=(8,5))
plt.scatter(range(len(y_test)), y_test, label="Clases Reales", marker="o", color="green")
plt.scatter(range(len(y_pred_continuo)), y_pred_continuo, label="Predicciones Continuas", marker="x", color="red")
plt.scatter(range(len(y_pred)), y_pred, label="Predicciones Redondeadas", marker="s", color="blue")
plt.title("Comparación entre valores reales, continuos y redondeados")
plt.xlabel("Número de muestra")
plt.ylabel("Clase (0=Setosa, 1=Versicolor, 2=Virginica)")
plt.savefig("imagenes/comparacion.png")
plt.legend()
plt.show()

# 7.3. Importancia de características (coeficientes del modelo)
plt.figure(figsize=(6,4))
plt.barh(caracteristicas_es, modelo.coef_, color="purple")
plt.title("Importancia de las características según el modelo")
plt.xlabel("Peso (coeficiente)")
plt.ylabel("Características")
plt.savefig("imagenes/importancia.png")
plt.show()

# 7.4. Distribución de clases en el dataset
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette="Set2")
plt.xticks([0,1,2], clases_es)
plt.title("Distribución de clases en el dataset Iris")
plt.xlabel("Clase")
plt.ylabel("Cantidad de muestras")
plt.savefig("imagenes/distribucion.png")
plt.show()

# ---------------------------
# 8. Predicción con nuevos datos
# ---------------------------
nuevos = np.array([
    [5.1, 3.5, 1.4, 0.2],  # parecido a Setosa
    [6.7, 3.0, 5.2, 2.3]   # parecido a Virginica
])

prediccion_nueva = np.round(modelo.predict(nuevos)).astype(int)
prediccion_nueva = np.clip(prediccion_nueva, 0, 2)

print("\nPredicción para nuevos datos:")
for i, p in enumerate(prediccion_nueva):
    print(f"Muestra {i+1}: {clases_es[p]}")
