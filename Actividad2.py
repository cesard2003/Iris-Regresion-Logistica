import argparse
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
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# NLTK para stopwords en español
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
spanish_stopwords = stopwords.words("spanish")

# ---------------------------
# Feature Engineering
# ---------------------------
def feature_engineering(df):
    expected = ["Subject","Body","Sender","Recipient","Date",
                "From_Domain","Language","Urgency",
                "Contains_Price","Call_to_Action","Label"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    # Texto combinado
    df["text"] = df["Subject"].fillna("") + " " + df["Body"].fillna("")

    # Domains
    df["sender_domain_extracted"] = df["Sender"].astype(str).apply(lambda s: s.split("@")[-1] if "@" in s else s)
    df["recipient_domain"] = df["Recipient"].astype(str).apply(lambda s: s.split("@")[-1] if "@" in s else s)

    # Date -> hour, dayofweek
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["hour"] = df["Date"].dt.hour.fillna(-1).astype(int)
    df["dayofweek"] = df["Date"].dt.dayofweek.fillna(-1).astype(int)

    # Contains_Price binario
    df["contains_price_bin"] = df["Contains_Price"].astype(str).str.contains("Sí|Si|\\$|\\d", case=False, regex=True).astype(int)

    # Urgency numérico
    urg_map = {"Baja":0, "Media":1, "Alta":2}
    df["urgency_num"] = df["Urgency"].map(urg_map).fillna(0).astype(int)

    # Label encoding
    le = LabelEncoder()
    df["label_bin"] = le.fit_transform(df["Label"].astype(str))

    return df, le

# ---------------------------
# Modelo y entrenamiento
# ---------------------------
def build_and_train(df, out_dir):
    FEATURE_TEXT = "text"
    CAT_FEATURES = ["From_Domain","sender_domain_extracted","recipient_domain","Call_to_Action","Language"]
    NUM_FEATURES = ["hour","dayofweek","urgency_num","contains_price_bin"]

    X = df[[FEATURE_TEXT] + CAT_FEATURES + NUM_FEATURES].copy()
    y = df["label_bin"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    # Preprocesamiento con stopwords en español
    tfidf = TfidfVectorizer(max_features=800, ngram_range=(1,2), stop_words=spanish_stopwords)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(transformers=[
        ("text", tfidf, FEATURE_TEXT),
        ("cat", ohe, CAT_FEATURES),
    ], remainder="passthrough")

    clf = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(solver="liblinear", max_iter=1000))
    ])

    print("Entrenando modelo...")
    clf.fit(X_train, y_train)

    # Probabilidades y umbral
    y_proba = clf.predict_proba(X_test)[:,1]
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    y_pred_best = (y_proba >= best_threshold).astype(int)

    conf_mat = confusion_matrix(y_test, y_pred_best)
    report = classification_report(y_test, y_pred_best)

    # Importancia de features
    model = clf.named_steps["clf"]
    pre = clf.named_steps["pre"]
    tfidf_feat_names = pre.named_transformers_["text"].get_feature_names_out()
    ohe_feat_names = pre.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES)
    coefs = model.coef_[0]
    abs_coefs = np.abs(coefs)

    text_importance = abs_coefs[:len(tfidf_feat_names)].sum()
    cat_importance = abs_coefs[len(tfidf_feat_names):len(tfidf_feat_names)+len(ohe_feat_names)].sum()
    num_importance = abs_coefs[-len(NUM_FEATURES):].sum()
    total = text_importance + cat_importance + num_importance
    group_percentages = {
        "textual": 100 * text_importance / total if total>0 else 0.0,
        "categorical": 100 * cat_importance / total if total>0 else 0.0,
        "numeric": 100 * num_importance / total if total>0 else 0.0
    }

    # Top tokens
    token_coefs = list(zip(tfidf_feat_names, coefs[:len(tfidf_feat_names)]))
    token_coefs_sorted = sorted(token_coefs, key=lambda x: x[1], reverse=True)
    top_spam_tokens = token_coefs_sorted[:15]

    # Correlación simple
    num_df = df[NUM_FEATURES].copy()
    for col in ["From_Domain","Call_to_Action"]:
        freq = df[col].value_counts(normalize=True).to_dict()
        num_df[f"{col}_freq"] = df[col].map(freq)
    corr = num_df.corr()

    # Crear carpeta
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Graficar correlación
    plt.figure(figsize=(8,6))
    plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="left")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Matriz de correlación")
    plt.savefig(out_dir / "correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    # ---------------------------
    # NUEVA GRÁFICA DE REGRESIÓN LOGÍSTICA
    # ---------------------------
    plt.figure(figsize=(8,6))
    plt.hist(y_proba[y_test==0], bins=30, alpha=0.6, label="HAM", color="blue")
    plt.hist(y_proba[y_test==1], bins=30, alpha=0.6, label="SPAM", color="red")
    plt.axvline(best_threshold, color="green", linestyle="--", label=f"Umbral óptimo ({best_threshold:.2f})")
    plt.xlabel("Probabilidad predicha de ser SPAM")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de probabilidades - Regresión Logística")
    plt.legend()
    plt.savefig(out_dir / "logistic_regression_plot.png", bbox_inches="tight")
    plt.close()

    # Guardar modelo
    joblib.dump(clf, out_dir / "pipeline_logistic_spam.pkl")

    # PDF resumen
    pdf_path = out_dir / "informe_modelo_regresion_logistica.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Informe: Clasificación de correos - Regresión Logística", styles["Heading1"]))
    story.append(Spacer(1,8))
    story.append(Paragraph(f"Umbral óptimo: {best_threshold:.2f}", styles["Normal"]))
    story.append(Paragraph(f"F1-Score: {best_f1:.4f}", styles["Normal"]))
    story.append(Paragraph(f"Matriz de confusión: {conf_mat.tolist()}", styles["Normal"]))
    story.append(Spacer(1,6))
    story.append(Paragraph("<b>Importancia de grupos de features:</b>", styles["Heading2"]))
    for k,v in group_percentages.items():
        story.append(Paragraph(f"{k}: {v:.1f} %", styles["Normal"]))
    story.append(Spacer(1,6))
    story.append(Paragraph("<b>Top tokens asociados a SPAM:</b>", styles["Heading2"]))
    for tok,coef in top_spam_tokens:
        story.append(Paragraph(f"{tok} ({coef:.3f})", styles["Normal"]))
    story.append(Spacer(1,6))
    story.append(RLImage(str(out_dir / "correlation_heatmap.png"), width=400, height=280))
    story.append(Spacer(1,6))
    story.append(Paragraph("<b>Distribución de probabilidades del modelo:</b>", styles["Heading2"]))
    story.append(RLImage(str(out_dir / "logistic_regression_plot.png"), width=400, height=280))
    doc.build(story)

    return {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "confusion_matrix": conf_mat,
        "report": report,
        "group_percentages": group_percentages,
        "pdf": pdf_path
    }

# ---------------------------
# MAIN
# ---------------------------
def main():
    # Cambia la ruta al dataset según tu caso
    default_path = Path("C:/Users/aguir/OneDrive/Escritorio/Machine Learning/dataset_correos_1000_instancias.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(default_path),
                        help="Ruta al dataset CSV")
    parser.add_argument("--out", type=str, default="spam_results", help="Carpeta de salida")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df, le = feature_engineering(df)
    results = build_and_train(df, args.out)

    print("==== RESULTADOS ====")
    print("Umbral óptimo:", results["best_threshold"])
    print("F1:", results["best_f1"])
    print("Matriz de confusión:\n", results["confusion_matrix"])
    print("PDF generado en:", results["pdf"])

if __name__ == "__main__":
    main()

