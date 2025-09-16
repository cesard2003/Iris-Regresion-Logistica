# ============================================================================
# Universidad de Cundinamarca
# Ingeniería de Sistemas
# Asignatura: Machine Learning
# Actividad: Clasificación de Plantas (Dataset IRIS) usando Regresión Lineal
# Autor: [Cesar Aguirre Hurtado]
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

# ---------------------------
# 1. Carga del dataset
# ---------------------------
iris = load_iris()
X = iris.data      # Variables independientes: [largo_sepalo, ancho_sepalo, largo_petalo, ancho_petalo]
y = iris.target    # Variable dependiente (clase): 0=Setosa, 1=Versicolor, 2=Virginica

print("Características del dataset:", iris.feature_names)
print("Clases:", iris.target_names)
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

print("\nCoeficientes del modelo (importancia de cada feature):")
for feature, coef in zip(iris.feature_names, modelo.coef_):
    print(f"{feature}: {coef:.4f}")

print("Intercepto (bias):", modelo.intercept_)

# ---------------------------
# 4. Predicciones
# ---------------------------
y_pred_continuo = modelo.predict(X_test)

# Convertir valores continuos a clases discretas (0,1,2)
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
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ---------------------------
# 6. Validación cruzada (opcional)
# ---------------------------
scores = cross_val_score(modelo, X, y, cv=5, scoring="r2")
print("Resultados de validación cruzada (R² en cada fold):", scores)
print("Promedio R²:", scores.mean())

# ---------------------------
# 7. Visualización
# ---------------------------

# Matriz de confusión como heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.title("Matriz de Confusión - Regresión Lineal (redondeada)")
plt.show()

# Distribución de predicciones continuas vs reales
plt.figure(figsize=(8,5))
plt.scatter(range(len(y_test)), y_test, label="Clases Reales", marker="o")
plt.scatter(range(len(y_pred_continuo)), y_pred_continuo, label="Predicciones Continuas", marker="x")
plt.scatter(range(len(y_pred)), y_pred, label="Predicciones Redondeadas", marker="s")
plt.title("Comparación entre valores reales, continuos y redondeados")
plt.xlabel("Muestra")
plt.ylabel("Clase")
plt.legend()
plt.show()
