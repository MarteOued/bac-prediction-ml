<h1 align="center">⚡ Bac Prediction ML</h1>

<p align="center">
  <i>Modèle de Machine Learning pour <b>prédire la réussite au baccalauréat</b>
  à partir de l'historique des notes du lycée — 4 algorithmes comparés.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Models-4%20ML-success?style=flat" />
  <img src="https://img.shields.io/badge/Best%20Accuracy-88.9%25-brightgreen?style=flat" />
  <img src="https://img.shields.io/badge/AUC-0.908-blueviolet?style=flat" />
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" />
</p>

<p align="center">
  <a href="#-quick-start">🚀 Quick Start</a> ·
  <a href="#-résultats">🏆 Résultats</a> ·
  <a href="#-démo">🎬 Démo</a> ·
  <a href="https://portfoliomarte.vercel.app">🌐 Portfolio</a>
</p>

---

## 🎯 Aperçu

Ce projet construit un **système de prédiction de la réussite au baccalauréat** à partir
des notes obtenues au lycée par 86 étudiants marocains. Le but : permettre aux équipes
pédagogiques d'**identifier en amont les élèves à risque** et de leur proposer un
accompagnement renforcé.

> **Ce projet remplace** une version antérieure basée sur une régression linéaire avec R² ≈ 0.
> La nouvelle approche en **classification binaire** atteint **88.9% d'accuracy** avec
> Logistic Regression.

## 🏆 Résultats

| Modèle | Accuracy (test) | AUC | CV 5-fold |
|--------|----------------:|----:|----------:|
| **Logistic Regression** ⭐ | **88.9%** | **0.908** | 81.5% ± 6.4% |
| Random Forest | 77.8% | 0.831 | 78.1% ± 10.7% |
| Gradient Boosting (~XGBoost) | 77.8% | 0.862 | 73.5% ± 9.2% |
| MLP (Deep Learning) | 72.2% | 0.862 | 73.4% ± 8.9% |

> Le **baseline** (toujours prédire "réussite") donnerait 70.9% — on gagne **+18 pp** d'accuracy.

## ✨ Fonctionnalités

- 🧹 **Pipeline de nettoyage** robuste (notes au format français, valeurs spéciales, NaN)
- 📊 **Feature engineering** : agrégations par étudiant + moyennes par matière
- 🤖 **4 algorithmes ML** entraînés et comparés
- 📈 **Cross-validation 5-fold** stratifiée pour la robustesse
- 🎨 **Interface Streamlit dark** avec navigation latérale (vue d'ensemble, comparaison, prédiction, méthodologie)
- 🔮 **Outil de prédiction interactif** : renseigne les notes d'un élève, obtient sa probabilité de réussite (3 niveaux)
- 📁 **Visualisations** générées (PNG haute qualité) : comparaison modèles, matrice de confusion, feature importance
- 💾 **Modèle persistant** (joblib) prêt à servir

## 📊 Données

- **`data_lycee.csv`** : 43 108 lignes — notes du lycée (1ère/2ème année Bac, tronc commun) sur 186 élèves
- **`data_bac.csv`** : 440 lignes — notes au bac national sur 86 élèves
- **86 élèves communs** = dataset d'entraînement final

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/MarteOued/bac-prediction-ml.git
cd bac-prediction-ml
pip install -r requirements.txt
```

### Entraîner les modèles

```bash
python train_models.py
```

Cela génère :
- `outputs/results.json` — toutes les métriques
- `outputs/01_model_comparison.png` — comparaison des 4 modèles
- `outputs/02_confusion_matrix.png` — matrice de confusion du meilleur modèle
- `outputs/03_feature_importance.png` — importance des features (Random Forest)
- `outputs/best_model.pkl` — modèle sérialisé pour l'app Streamlit

### Lancer l'app interactive

```bash
streamlit run app.py
```

L'app s'ouvre sur `http://localhost:8501` avec 4 sections accessibles via la sidebar :
- 🏠 **Vue d'ensemble** : KPIs animés + barres de performance + visualisations
- 📊 **Comparaison interactive** : graphiques Plotly dark sur toutes les métriques
- 🎯 **Outil de prédiction** : formulaire avec labels en français, 3 profils exemples, résultat à 3 niveaux
- 📚 **Méthodologie & Auteure** : pipeline, stack technique, présentation

## 📂 Structure du projet

```
bac-prediction-ml/
├── data/
│   ├── data_lycee.csv          # Notes lycée (43 108 lignes)
│   └── data_bac.csv            # Notes bac (440 lignes)
├── data_preparation.py         # Pipeline de nettoyage + feature engineering
├── train_models.py             # Entraînement + évaluation des 4 modèles
├── app.py                      # Interface Streamlit
├── outputs/
│   ├── results.json            # Métriques des 4 modèles
│   ├── best_model.pkl          # Modèle entraîné (prêt à servir)
│   ├── 01_model_comparison.png
│   ├── 02_confusion_matrix.png
│   └── 03_feature_importance.png
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧠 Méthodologie

### 1. Nettoyage des données

- Marks au **format français** : `"18,50"` → `18.5`
- Valeurs spéciales : `"****"` et `"Abs."` → NaN
- Strip whitespace sur toutes les colonnes texte
- Drop lignes avec `mark` manquant

### 2. Feature engineering

Pour chaque élève, on agrège ses notes lycée en :
- `lycee_mean`, `lycee_std`, `lycee_min`, `lycee_max`, `lycee_count`, `lycee_median`
- `avg_<matière>` pour le top 5-15 matières les plus fréquentes
- `age` (dernier connu)

### 3. Cible

```python
bac_pass = (mean_bac >= 10).astype(int)
```

Cible **binaire** : réussite (1) ou échec (0).

### 4. Modèles

| Modèle | Type | Hyperparamètres |
|--------|------|-----------------|
| Logistic Regression | Linéaire | C=0.5, max_iter=2000 |
| Random Forest | Bagging | n=300, max_depth=6 |
| Gradient Boosting | Boosting | n=200, lr=0.05, depth=3 |
| MLP (Deep Learning) | Réseau de neurones | hidden=(64,32), alpha=0.01 |

### 5. Évaluation

- **Split** train/test 80/20 stratifié (random_state=2024)
- **Standardisation** des features (StandardScaler)
- **Métriques** : Accuracy, Precision, Recall, F1, AUC
- **Cross-validation** 5-fold stratifiée pour vérifier la robustesse

## 🛠️ Stack technique

| Catégorie | Technologie |
|-----------|-------------|
| **Langage** | Python 3.10+ |
| **ML** | Scikit-Learn (LogReg, RF, GBM, MLP) |
| **Data** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **UI** | Streamlit |
| **Persistance** | Joblib |

## 🎬 Démo

> **➡ [Voir la démo complète sur YouTube](https://youtu.be/q1m8X7rNJnU)**

La vidéo montre les 4 sections de l'interface, l'outil de prédiction interactif avec les 3 profils exemples (élève en difficulté, niveau moyen, excellent), ainsi que les visualisations Plotly animées.

## 👩‍💻 Auteure

**Martine Ouedraogo** — Master 1 Informatique, spécialisation Data Science & Machine Learning
Université Lumière Lyon 2 · 2026

[LinkedIn](https://www.linkedin.com/in/marte-oued) · [Portfolio](https://portfoliomarte.vercel.app) · [GitHub](https://github.com/MarteOued)

## 📜 Licence

MIT — Projet de recherche académique. Données utilisées avec autorisation pédagogique.
