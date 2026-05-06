"""
Entraînement et évaluation des 4 modèles ML.

Modèles :
- Logistic Regression (baseline interprétable)
- Random Forest (ensemble bagging)
- Gradient Boosting (ensemble boosting, équivalent XGBoost natif sklearn)
- MLP (Multi-Layer Perceptron, réseau de neurones / Deep Learning)

Évaluation :
- Cross-validation 5-fold stratifiée (robustesse)
- Train/test split 80/20 (résultat publié)
- Accuracy, classification report, matrice de confusion
"""

import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

warnings.filterwarnings("ignore")

from data_preparation import build_dataset

RANDOM_STATE = 2024  # seed reproductible


def get_models():
    """Retourne les 4 modèles à entraîner."""
    return {
        "Logistic Regression": LogisticRegression(C=0.5, max_iter=2000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
        "MLP (Deep Learning)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=3000, alpha=0.01, random_state=42),
    }


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    """Entraîne et évalue un modèle sur un split donné."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }


def cross_validate_model(model, X, y, n_splits=5) -> tuple:
    """Cross-validation 5-fold stratifiée."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()


def train_all(data_dir: str = "data", output_dir: str = "outputs"):
    """Pipeline complet : prepare data, train models, save results."""
    print("=" * 70)
    print("  PIPELINE D'ENTRAÎNEMENT — Prédiction Bac")
    print("=" * 70)

    # 1. Préparer les données
    X, y, df_full = build_dataset(data_dir=data_dir)
    feature_names = X.columns.tolist()
    X = X.values
    y = y.values
    print(f"\n[1] Dataset : {X.shape[0]} étudiants × {X.shape[1]} features")
    print(f"    Taux de réussite : {y.mean()*100:.1f}% ({y.sum()}/{len(y)})")

    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"\n[2] Split : {len(X_train)} train / {len(X_test)} test")

    # 3. Entraîner et évaluer chaque modèle
    results = {}
    print(f"\n[3] Entraînement des 4 modèles ML\n")
    print(f"{'Modèle':25s} {'Accuracy':>10s} {'AUC':>8s} {'CV moy':>10s}")
    print("-" * 60)

    for name, model in get_models().items():
        # Évaluation single split
        eval_result = evaluate_model(model, X_train_s, X_test_s, y_train, y_test)
        # Cross-validation
        cv_mean, cv_std = cross_validate_model(model, scaler.transform(X), y)

        results[name] = {
            **eval_result,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }
        print(f"{name:25s} {eval_result['accuracy']*100:9.1f}% {eval_result['auc']:8.3f} {cv_mean*100:8.1f}±{cv_std*100:.1f}%")

    # 4. Identifier le meilleur modèle
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_acc = results[best_name]["accuracy"]
    print(f"\n[4] >> MEILLEUR MODÈLE : {best_name} → {best_acc*100:.1f}% accuracy <<")

    # 5. Sauvegarde des résultats
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)

    # JSON résultats
    results_json = {
        name: {
            "accuracy":  r["accuracy"],
            "precision": r["precision"],
            "recall":    r["recall"],
            "f1":        r["f1"],
            "auc":       r["auc"],
            "cv_mean":   r["cv_mean"],
            "cv_std":    r["cv_std"],
            "confusion_matrix": r["confusion_matrix"],
        } for name, r in results.items()
    }
    results_json["_meta"] = {
        "best_model": best_name,
        "best_accuracy": best_acc,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "baseline_accuracy": float(y.mean()),
    }
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"\n[5] Résultats sauvegardés : {out / 'results.json'}")

    # Visualisations
    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#F8F9FA"})

    # Graph 1 : Comparaison des modèles
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]
    cv_means = [results[n]["cv_mean"] * 100 for n in names]
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy (test)", color="#6366F1", edgecolor="white")
    bars2 = ax.bar(x + w/2, cv_means, w, label="Accuracy (CV 5-fold)", color="#A855F7", edgecolor="white")
    ax.axhline(y=87, color="green", linestyle="--", alpha=0.7, label="Cible : 87%")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.legend()
    ax.set_title("Comparaison des 4 modèles ML — Prédiction Bac", fontweight="bold", pad=15)
    for bar, v in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "01_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Graph 2 : Confusion matrix du meilleur modèle
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array(results[best_name]["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="BuPu", cbar=False,
                xticklabels=["Échec", "Réussite"], yticklabels=["Échec", "Réussite"], ax=ax)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de confusion — {best_name}", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out / "02_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Graph 3 : Feature importance (Random Forest)
    rf = get_models()["Random Forest"]
    rf.fit(X_train_s, y_train)
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#6366F1" if i == len(importances)-1 else "#A5B4FC" for i in range(len(importances))]
    importances.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Importance"); ax.set_title("Top features — Random Forest", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out / "03_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[6] Visualisations sauvegardées : {out}/")
    print(f"    - 01_model_comparison.png")
    print(f"    - 02_confusion_matrix.png")
    print(f"    - 03_feature_importance.png")

    # Sauvegarde du meilleur modèle pour l'app Streamlit
    best_model = get_models()[best_name]
    best_model.fit(scaler.transform(X), y)
    joblib.dump({"model": best_model, "scaler": scaler, "feature_names": feature_names},
                out / "best_model.pkl")
    print(f"[7] Modèle entraîné sauvegardé : {out / 'best_model.pkl'}")

    print("\n" + "=" * 70)
    print(f"  ✓ TERMINÉ — Meilleur modèle : {best_name} = {best_acc*100:.1f}%")
    print("=" * 70)

    return results, best_name


if __name__ == "__main__":
    train_all()
