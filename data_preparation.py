"""
Préparation et nettoyage des données.

Pipeline :
1. Charger data_lycee.csv (43k+ lignes, ~600 élèves) — historique notes lycée
2. Charger data_bac.csv (440 lignes, 84 élèves) — résultats au bac
3. Nettoyer les marks ("18,50" → 18.5, "****" → NaN, "Abs." → NaN)
4. Agréger les notes lycée par élève (mean, std, min, max, count par matière)
5. Calculer la moyenne au bac → cible binaire (réussite si moyenne >= 10)
6. Joindre lycée + bac sur le code étudiant
7. Renvoyer X (features) et y (cible)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(data_dir: str = "data"):
    """Charge les CSV bruts."""
    data_dir = Path(data_dir)
    df_lycee = pd.read_csv(data_dir / "data_lycee.csv")
    df_bac = pd.read_csv(data_dir / "data_bac.csv")
    return df_lycee, df_bac


def clean_marks(series: pd.Series) -> pd.Series:
    """Nettoie les notes :
    - Remplace ',' par '.' (format français)
    - Remplace '****' et 'Abs.' par NaN
    - Convertit en float
    """
    if series.dtype == object:
        series = (
            series.astype(str)
                  .str.strip()
                  .replace({"****": np.nan, "Abs.": np.nan, "nan": np.nan})
                  .str.replace(",", ".", regex=False)
        )
    return pd.to_numeric(series, errors="coerce")


def clean_lycee(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le dataset lycée."""
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["mark"] = clean_marks(df["mark"])

    # Strip whitespace partout
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Drop NaN sur la note (sinon agg sera faussée)
    df = df.dropna(subset=["mark"])
    return df


def clean_bac(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le dataset bac."""
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["mark"] = clean_marks(df["mark"])

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def aggregate_lycee_per_student(df: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque élève, calcule des features agrégées sur ses notes lycée :
    - moyenne globale, écart-type, min, max, nb d'évaluations
    - moyenne par grand domaine (scientifique, littéraire, langues, autre)
    """
    # Mapping matière → domaine
    SCIENTIFIC = ["MATHEMATIQUES", "PHYSIQUE CHIMIE", "SVT", "SC. DE LA VIE"]
    LITERARY = ["PHILOSOPHIE", "FRANCAIS", "HIST", "ARABE", "EDUCATION ISLAMIQUE"]
    LANGUAGES = ["LANGUE", "ANGLAIS", "ANGLAISE", "ESPAGNOL", "ALLEMAND"]

    def domaine(subject: str) -> str:
        s = subject.upper()
        if any(k in s for k in SCIENTIFIC):
            return "scientifique"
        if any(k in s for k in LANGUAGES):
            return "langues"
        if any(k in s for k in LITERARY):
            return "litteraire"
        return "autre"

    df = df.copy()
    df["domaine"] = df["subject"].apply(domaine)

    # Aggregations globales
    agg_global = df.groupby("code")["mark"].agg(
        lycee_mean="mean",
        lycee_std="std",
        lycee_min="min",
        lycee_max="max",
        lycee_count="count",
    ).reset_index()

    # Aggregations par domaine
    agg_domaine = df.groupby(["code", "domaine"])["mark"].mean().unstack(fill_value=0).reset_index()
    agg_domaine.columns = ["code"] + [f"lycee_mean_{c}" for c in agg_domaine.columns[1:]]

    # Niveau le plus récent (le bac year)
    last_level = df.groupby("code")["level"].last().reset_index()
    last_level.columns = ["code", "last_level"]

    # Age
    last_age = df.groupby("code")["age"].last().reset_index()
    last_age.columns = ["code", "age"]

    # Merge tous les aggregates
    features = (agg_global
                .merge(agg_domaine, on="code", how="left")
                .merge(last_level, on="code", how="left")
                .merge(last_age, on="code", how="left"))

    return features


def aggregate_bac_per_student(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule la moyenne du bac par étudiant + la cible binaire (réussite >= 10)."""
    avg = df.groupby("code")["mark"].mean().reset_index()
    avg.columns = ["code", "bac_mean"]
    avg["bac_pass"] = (avg["bac_mean"] >= 10).astype(int)
    return avg


def build_dataset(data_dir: str = "data"):
    """Pipeline complet : retourne X (features) et y (cible binaire)."""
    df_lycee, df_bac = load_raw_data(data_dir)

    df_lycee = clean_lycee(df_lycee)
    df_bac = clean_bac(df_bac)

    features = aggregate_lycee_per_student(df_lycee)
    targets = aggregate_bac_per_student(df_bac)

    # Merge inner : on garde uniquement les élèves présents dans les 2 datasets
    df = features.merge(targets, on="code", how="inner")

    if len(df) == 0:
        raise ValueError("Aucun élève en commun entre lycée et bac. Vérifie les codes.")

    # Encode last_level (catégoriel)
    df = pd.get_dummies(df, columns=["last_level"], prefix="level", drop_first=True)

    # Drop colonnes non-features
    feature_cols = [c for c in df.columns if c not in ["code", "bac_mean", "bac_pass"]]
    X = df[feature_cols].fillna(0)
    y = df["bac_pass"]

    return X, y, df  # df complet pour analyses


if __name__ == "__main__":
    X, y, df = build_dataset()
    print(f"Dataset construit : {X.shape[0]} étudiants, {X.shape[1]} features")
    print(f"Cible : {y.sum()} réussites / {len(y)} ({y.mean()*100:.1f}% taux de réussite)")
    print(f"\nFeatures :\n{list(X.columns)}")
    print(f"\nAperçu X :\n{X.head()}")
