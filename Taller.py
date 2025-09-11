#!/usr/bin/env python3
"""
Actividad: Clasificación de correos electrónicos (SPAM vs HAM) con Regresión Logística en sklearn.
Autor: Estudiante Ingeniería de Sistemas - UDEC Facatativá
Materia: Machine Learning - 8º semestre

Requisitos:
  pip install pandas scikit-learn matplotlib joblib reportlab
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------
# Feature Engineering
# ---------------------------
def preparar_datos(df):
    # Texto combinado
    df["texto"] = df["Subject"].fillna("") + " " + df["Body"].fillna("")

    # Extraer dominio
    df["dominio_remitente"] = df["Sender"].astype(str).apply(lambda s: s.split("@")[-1] if "@" in s else s)
    df["dominio_destino"] = df["Recipient"].astype(str).apply(lambda s: s.split("@")[-1] if "@" in s else s)

    # Convertir fecha
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["hora"] = df["Date"].dt.hour.fillna(-1).astype(int)
    df["dia_semana"] = df["Date"].dt.dayofweek.fillna(-1).astype(int)

    # Variables binarias
    df["precio_bin"] = df["Contains_Price"].astype(str).str.contains("Sí|Si|\\$|\\d", case=False, regex=True).astype(int)
    urg_map = {"Baja":0, "Media":1, "Alta":2}
    df["urgencia_num"] = df["Urgency"].map(urg_map).fillna(0).astype(int)

    # Etiqueta
    le = LabelEncoder()
    df["y"] = le.fit_transform(df["Label"].astype(str))

    return df, le

# ---------------------------
# Modelo de regresión logística
# ---------------------------
def entrenar_modelo(df, carpeta_out="resultados_logistica"):
    # Features
    FEATURE_TEXT = "texto"
    CAT_FEATURES = ["From_Domain","dominio_remitente","dominio_destino","Call_to_Action","Language"]
    NUM_FEATURES = ["hora","dia_semana","urgencia_num","precio_bin"]

    X = df[[FEATURE_TEXT] + CAT_FEATURES + NUM_FEATURES].copy()
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Preprocesamiento
    tfidf = TfidfVectorizer(max_features=800, ngram_range=(1,2))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocesador = ColumnTransformer([
        ("texto", tfidf, FEATURE_TEXT),
        ("categ", ohe, CAT_FEATURES),
    ], remainder="passthrough")

    modelo = Pipeline([
        ("prep", preprocesador),
        ("clf", LogisticRegression(solver="liblinear", max_iter=1000))
    ])

    print("Entrenando modelo de Regresión Logística...")
    modelo.fit(X_train, y_train)

    # Predicciones con probabilidad
    y_proba = modelo.predict_proba(X_test)[:,1]
    thresholds = np.linspace(0.01,0.99,99)
    f1_scores = [f1_score(y_test,(y_proba>=t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    mejor_umbral = thresholds[best_idx]
    mejor_f1 = f1_scores[best_idx]

    y_pred = (y_proba >= mejor_umbral).astype(int)
    matriz_conf = confusion_matrix(y_test, y_pred)
    reporte = classification_report(y_test, y_pred)

    # Importancia de grupos de features
    modelo_clf = modelo.named_steps["clf"]
    prep = modelo.named_steps["prep"]

    coefs = modelo_clf.coef_[0]
    abs_coefs = np.abs(coefs)

    n_text = len(prep.named_transformers_["texto"].get_feature_names_out())
    n_cat = len(prep.named_transformers_["categ"].get_feature_names_out(CAT_FEATURES))
    n_num = len(NUM_FEATURES)

    text_imp = abs_coefs[:n_text].sum()
    cat_imp = abs_coefs[n_text:n_text+n_cat].sum()
    num_imp = abs_coefs[-n_num:].sum()
    total = text_imp+cat_imp+num_imp

    importancia = {
        "Textuales": 100*text_imp/total,
        "Categóricas": 100*cat_imp/total,
        "Numéricas": 100*num_imp/total
    }

    # Correlación
    num_df = df[NUM_FEATURES].copy()
    corr = num_df.corr()
    carpeta = Path(carpeta_out)
    carpeta.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6,5))
    plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.title("Matriz de correlación")
    plt.savefig(carpeta/"heatmap_correlacion.png", bbox_inches="tight")
    plt.close()

    # Guardar modelo
    joblib.dump(modelo, carpeta/"modelo_logistico.pkl")

    # PDF
    pdf_path = carpeta/"informe_regresion_logistica.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Informe: Regresión Logística para detección de SPAM", styles["Heading1"]))
    story.append(Spacer(1,10))
    story.append(Paragraph(f"Mejor umbral: {mejor_umbral:.2f}", styles["Normal"]))
    story.append(Paragraph(f"Mejor F1-Score: {mejor_f1:.4f}", styles["Normal"]))
    story.append(Paragraph(f"Matriz de confusión: {matriz_conf.tolist()}", styles["Normal"]))
    story.append(Spacer(1,10))
    story.append(Paragraph("Importancia de grupos de features:", styles["Heading2"]))
    for k,v in importancia.items():
        story.append(Paragraph(f"{k}: {v:.2f}%", styles["Normal"]))
    story.append(Spacer(1,10))
    story.append(RLImage(str(carpeta/"heatmap_correlacion.png"), width=400, height=280))
    doc.build(story)

    print("=== RESULTADOS ===")
    print("Mejor umbral:", mejor_umbral)
    print("F1:", mejor_f1)
    print("Matriz de confusión:\n", matriz_conf)
    print("PDF generado en:", pdf_path)

    return {
        "umbral": mejor_umbral,
        "f1": mejor_f1,
        "matriz": matriz_conf,
        "reporte": reporte,
        "pdf": pdf_path
    }

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    # Ruta al dataset en el escritorio
    ruta_dataset = Path("C:/Users/aguir/OneDrive/Escritorio/Machine Learning/dataset_correos_1000_instancias.csv")

    df = pd.read_csv(ruta_dataset)
    df, le = preparar_datos(df)
    resultados = entrenar_modelo(df)
