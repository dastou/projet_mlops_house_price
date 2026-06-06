# Fichier de suivi — Projet MLOps Laplace Immo

> **Rôle de ce fichier** : suivi pas-à-pas du projet pour ne pas perdre le fil entre les sessions. À relire au début de chaque nouvelle session.

---

## 0. État actuel

- **Date de démarrage** : 2026-06-05
- **Dernière mise à jour** : 2026-06-06 (étape 1 terminée — env `mlops_immo` créé avec Python 3.11.15)
- **Étape en cours** : Étape 2 — Structure projet + Git + GitHub
- **Prochaine action** : Créer la structure de dossiers et les fichiers initiaux du projet

---

## 1. Contexte du projet

### Énoncé (Laplace Immo — sujet EPT/DIC3)
Construire un **simulateur de prix de maisons** pour le réseau d'agences Laplace Immo. Approche : régression supervisée sur le dataset Kaggle "House Prices Advanced Regression Techniques" (Ames, Iowa).

### Livrables exigés par le sujet (consolidés)

**Obligatoires** :
- [ ] **Dépôt GitHub public** avec README détaillé
- [ ] Code Python structuré (pas un notebook monolithique)
- [ ] **DVC configuré avec pipeline reproductible** (`dvc.yaml`)
- [ ] **Dockerfile fonctionnel**
- [ ] Expériences trackées sur MLflow — **minimum 10 runs comparés**
- [ ] **API REST déployée et documentée** (Swagger auto via FastAPI)
- [ ] Pipeline CI/CD fonctionnel sur GitHub Actions
- [ ] Notebook d'analyse exploratoire propre et annoté
- [ ] Code/notebook des essais de modèles + identification du modèle final
- [ ] Tests unitaires Pytest exécutés automatiquement via GitHub Actions
- [ ] Respect des conventions PEP 8
- [ ] Support de présentation 25 slides max → **fait sur Canva** (hors repo)
- [ ] Dossier `notebooks/` dans le repo GitHub
- [ ] Nommage des notebooks : `house_price_01_*.ipynb`, etc.

**Optionnels** :
- [ ] Dashboard de monitoring avec détection de drift (Evidently) — **on ne le fait pas dans le scope initial**
- [ ] Documentation d'architecture (schéma + justification des choix) — **on le fait** (gain facile, soutenance)

**Bonus que j'ajoute (non exigé mais fort impact soutenance)** :
- [ ] Interface utilisateur Streamlit (consommant l'API FastAPI)

### Évaluation
- Fonctionnement technique : **40 %**
- Qualité du code : **30 %**
- Présentation orale : **30 %**

### Format de soutenance
- 15–20 min présentation + 5–10 min Q&A
- Plan : problématique → cleaning + FE + exploration → essais de modèles → modèle final + améliorations

---

## 2. Dataset (vérifié)

- **Source officielle** : Kaggle (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Source pratique** : OpenML id `42165` via `sklearn.datasets.fetch_openml("house_prices")` → **pas besoin de compte Kaggle**
- **Train** : 1 460 lignes × 81 colonnes (Id + 79 features + SalePrice)
- **Test (Kaggle uniquement)** : 1 459 lignes sans SalePrice — non nécessaire pour Laplace Immo
- **Cible** : `SalePrice` (continu, en dollars)
- **Métrique officielle Kaggle** : RMSE sur `log(SalePrice)` — on l'utilisera comme métrique principale
- **Métriques métier** (pour le jury) : R², MAE en dollars

### Décision : source de données
On utilise **OpenML via sklearn** pour la portabilité (CI/CD sans token Kaggle).

---

## 3. Stack technique (révisée)

| Besoin | Outil | Version cible |
|---|---|---|
| Env Python | conda | Python 3.11 |
| Manipulation | pandas, numpy | latest stable |
| Modèles | scikit-learn, xgboost, lightgbm | latest stable |
| Pipeline ML | `sklearn.pipeline.Pipeline` + `ColumnTransformer` | — |
| Tracking | MLflow (local) | 2.x |
| **Versioning données + pipeline** | **DVC** | 3.x |
| **Containerisation** | **Docker** | latest stable |
| **API REST** | **FastAPI + Uvicorn + Pydantic** | latest stable |
| **UI démo (bonus)** | **Streamlit** | latest stable |
| Tests | Pytest + httpx (pour tester API) | latest |
| Lint/format | flake8 + black | latest |
| CI/CD | GitHub Actions | — |
| Visualisations EDA | matplotlib, seaborn, missingno | — |

### Outils écartés (et raison)
- **ZenML / Prefect / Airflow** : DVC + scripts Python suffisent
- **Kubernetes** : Docker seul suffit
- **Optuna / Hyperopt** : `GridSearchCV` ou `RandomizedSearchCV` suffisent
- **ydata-profiling** : trop lourd, notebook EDA manuel plus défendable
- **Evidently AI (drift)** : sorti du scope initial, on l'ajoute seulement si tout fini à J-3

---

## 4. Décisions méthodologiques

### Décision A — Variables exclues (data leakage métier)
Variables exclues car non disponibles au moment de l'estimation Laplace Immo (avant la vente) :
- `SaleType`, `SaleCondition` — décrivent la transaction, pas la maison
- `MoSold`, `YrSold` — date de vente future, inconnue lors du conseil

**Argument soutenance** : décision de rigueur métier, perte minime de RMSE mais modèle réellement utilisable en production.

### Décision B — Gestion des NaN structurels
NaN "non applicable" (≠ donnée manquante) à remplacer par `"None"` :
`Alley`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `PoolQC`, `Fence`, `MiscFeature`, `MasVnrType`.

NaN numériques liés → remplacer par `0` :
`GarageYrBlt`, `GarageCars`, `GarageArea`, `MasVnrArea`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, `BsmtHalfBath`.

Vrais NaN → imputation contextuelle :
- `LotFrontage` → médiane par `Neighborhood`
- `Electrical` → mode

Features binaires complémentaires : `has_pool`, `has_garage`, `has_basement`, `has_fireplace`, `has_2ndfloor`, `has_porch`.

### Décision C — Anti-leakage technique
- Split AVANT preprocessing (toujours)
- `sklearn.Pipeline` + `ColumnTransformer` → fit-sur-train-only garanti
- Cross-validation via `cross_val_score` (isolation par fold)

### Décision D — Transformation cible
`np.log1p(SalePrice)` avant entraînement, `np.expm1` pour revenir aux dollars.
Aligne avec la métrique Kaggle officielle (RMSE log).

### Décision E — Feature engineering ciblé
5 axes :
1. **Âges** : `building_age`, `remodel_age`, `garage_age`, `was_remodeled`
2. **Surfaces totales** : `TotalSF`, `TotalPorchSF`
3. **Comptages composites** : `TotalBathrooms` (avec pondération 0.5 sur HalfBath)
4. **Booléens has_X** (cf. Décision B)
5. **Log1p sur features avec |skew| > 0.75** (LotArea, GrLivArea, etc.)

Bonus : encodage ordinal manuel pour les variables de qualité (`Po < Fa < TA < Gd < Ex`).

### Décision F — API : FastAPI + Pydantic (nouveau)
- FastAPI pour la doc Swagger auto à `/docs`
- Pydantic `BaseModel` pour valider les inputs (types + ranges)
- 2 endpoints minimum : `POST /predict` + `GET /health`
- Tests via `pytest` + `TestClient` ou `httpx`

### Décision G — Pipeline reproductible avec DVC (nouveau)
Stages dans `dvc.yaml` :
1. `load` : télécharge depuis OpenML → `data/raw/train.csv`
2. `preprocess` : nettoyage + FE → `data/processed/train.csv`
3. `train` : entraîne le modèle → `models/model.pkl`
4. `evaluate` : calcule les métriques → `metrics.json`

`dvc repro` réexécute uniquement les stages affectés par un changement.

### Décision H — Containerisation Docker (nouveau)
- Image basée sur `python:3.11-slim`
- Contient l'API FastAPI + le modèle entraîné
- Expose le port 8000
- Buildable en local et en CI

### Décision I — Bonus interface utilisateur Streamlit (nouveau)
- Une page formulaire pour saisir les caractéristiques d'une maison
- Appelle l'API FastAPI via `requests.post`
- Affiche le prix prédit en dollars
- Justification : impact soutenance > Evidently (visible, démontrable, métier-aligné)

---

## 5. Architecture cible du repo

```
projet_mlops/
├── data/                          # DVC-tracked, gitignored
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── house_price_01_analyse.ipynb         # EDA
│   ├── house_price_02_preprocessing.ipynb   # cleaning + FE
│   ├── house_price_03_essais.ipynb          # essais 10+ modèles + MLflow
│   └── house_price_04_final.ipynb           # modèle final + tuning
├── src/
│   ├── __init__.py
│   ├── data.py                    # load_data via OpenML
│   ├── preprocessing.py           # fonctions de cleaning
│   ├── features.py                # feature engineering
│   ├── pipeline.py                # Pipeline sklearn complet
│   ├── train.py                   # script DVC stage : entraînement
│   └── evaluate.py                # script DVC stage : évaluation
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app
│   └── schemas.py                 # Pydantic models
├── ui/
│   └── app.py                     # Streamlit (bonus)
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_pipeline.py
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci_cd.yml              # lint + test + Docker build
├── docs/
│   └── architecture.md            # schéma + justification (optionnel-fait)
├── models/                        # DVC-tracked
├── mlruns/                        # gitignored
├── dvc.yaml                       # pipeline DVC
├── dvc.lock                       # auto-généré
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── config.yaml
├── .gitignore
├── README.md
└── PROGRESSION.md                 # ce fichier
```

---

## 6. Plan d'exécution (étapes)

| # | Étape | Livrable | Statut |
|---|---|---|---|
| 1 | Setup environnement conda | env `mlops_immo` activable | ✅ |
| 2 | Structure projet + Git + GitHub public | repo initialisé, structure créée | ⏳ En cours |
| 3 | `src/data.py` : chargement OpenML | fonction `load_data()` testée | ⬜ |
| 4 | Notebook 01 : EDA | `house_price_01_analyse.ipynb` propre | ⬜ |
| 5 | `src/preprocessing.py` + Notebook 02 | preprocessing modulaire + notebook annoté | ⬜ |
| 6 | `src/features.py` (feature engineering) | fonctions FE testables | ⬜ |
| 7 | `src/pipeline.py` : Pipeline sklearn | Pipeline anti-leakage complet | ⬜ |
| 8 | Notebook 03 : essais **min 10 runs** MLflow | tracking MLflow visible dans `mlflow ui` | ⬜ |
| 9 | Notebook 04 : modèle final + tuning | meilleur modèle sauvegardé en `.pkl` | ⬜ |
| 10 | **DVC : `dvc.yaml` pipeline reproductible** | `dvc repro` fonctionne end-to-end | ⬜ |
| 11 | **FastAPI `api/main.py` + Swagger** | API testable sur `/docs` | ⬜ |
| 12 | Tests Pytest (incl. test_api.py) | `pytest tests/` passe en local | ⬜ |
| 13 | **Dockerfile + test build local** | `docker run` lance l'API | ⬜ |
| 14 | GitHub Actions CI/CD (lint + test + Docker build) | workflow vert sur GitHub | ⬜ |
| 15 | **Streamlit `ui/app.py` (bonus)** | démo locale consommant l'API | ⬜ |
| 16 | **Schéma d'architecture (`docs/architecture.md`)** | diagramme + justifications | ⬜ |
| 17 | README final détaillé | doc claire pour le jury | ⬜ |
| 18 | Préparation soutenance (slides Canva) | 25 slides max | ⬜ |

Légende : ⬜ à faire — ⏳ en cours — ✅ terminé — ❌ bloqué

---

## 7. Insights de la recherche web (à appliquer)

### Feature engineering
- Grouper variables : espace (SF), sous-sol, équipements, construction, garage
- Log1p sur features avec skew > 0.75
- Stacking (ElasticNet + RF + GBM) → top Kaggle
- FE > choix du modèle

### Pipeline anti-leakage
- `ColumnTransformer` pour traiter colonnes numériques et catégorielles séparément
- `Pipeline(steps=[("prep", ColumnTransformer(...)), ("model", XGBRegressor())])`
- Toujours `train_test_split` AVANT preprocessing
- CV via `cross_val_score(pipeline, X, y)` — isolation par fold

### MLflow + sklearn
- `mlflow.sklearn.autolog()` capture params + métriques + modèle automatiquement
- Sinon manuel : `mlflow.start_run()` + `log_params/log_metric/log_model`
- Tracking local → `mlruns/` à gitignorer
- `infer_signature()` pour documenter input/output du modèle

### Tests data science
- Schema tests (colonnes + types)
- Row-level (pas de NaN dans colonnes critiques)
- Column-level (ranges plausibles)
- Pipeline-level (n_in == n_out)
- API tests (status codes, payload validation, prédiction dans range)

### GitHub Actions Python
- Trigger : `on: [push, pull_request]`
- Runner : `ubuntu-latest`
- Cache pip via `actions/setup-python@v5` + `cache: pip`
- Jobs : `lint` + `test` + `docker-build`
- Pas de matrix Python multiple pour projet de cours

### DVC
- `dvc init` → crée `.dvc/`
- `dvc add data/raw/train.csv` → tracking d'un fichier
- `dvc.yaml` → définit stages avec deps/outs
- `dvc repro` → exécute les stages nécessaires (cache intelligent)
- `dvc dag` → affiche le graphe du pipeline

### FastAPI
- Pydantic `BaseModel` valide automatiquement les inputs
- `/docs` génère Swagger UI automatiquement
- `/redoc` génère doc ReDoc alternative
- Tester avec `TestClient` de `fastapi.testclient`

### Docker
- Base : `python:3.11-slim` (image légère)
- `WORKDIR /app`
- Copier `requirements.txt` AVANT le code → cache Docker efficace
- `.dockerignore` pour exclure `.git`, `mlruns/`, `data/`, etc.

---

## 8. Ressources utiles

### Documentation officielle
- [OpenML House Prices (id 42165)](https://www.openml.org/d/42165)
- [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- [MLflow sklearn guide](https://mlflow.org/docs/latest/ml/traditional-ml/sklearn/guide/)
- [sklearn Pipelines + ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html)
- [sklearn common pitfalls (data leakage)](https://scikit-learn.org/stable/common_pitfalls.html)
- [DVC docs](https://dvc.org/doc)
- [FastAPI docs](https://fastapi.tiangolo.com/)
- [Docker docs](https://docs.docker.com/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
- [Streamlit docs](https://docs.streamlit.io/)

### Articles techniques
- [Avoiding leakage with Pipeline + CV](https://medium.com/@wing.y.tse.jobs/a-step-by-step-guide-to-avoid-data-leakage-by-pipeline-randomizedsearchcv-and-votingregressor-in-68b01aca6109)
- [Markaicode — Leak-free pipelines](https://markaicode.com/sklearn-pipeline-no-leakage/)
- [eugeneyan — Writing robust ML tests](https://eugeneyan.com/writing/testing-pipelines/)
- [Top 100 Kaggle solution](https://blog.finxter.com/how-i-cracked-the-top-100-in-the-kaggle-house-prices-competition/)
- [NYC Data Science Academy — Creative FE](https://nycdatascience.com/blog/student-works/house-price-prediction-with-creative-feature-engineering-and-advanced-regression-techniques/)

### Repos GitHub pédagogiques
- [mlopsbootcamp/house-price-predictor](https://github.com/mlopsbootcamp/house-price-predictor) — template MLOps complet
- [InseeFrLab/formation-mlops](https://github.com/InseeFrLab/formation-mlops) — formation MLOps en français
- [chriskhanhtran/kaggle-house-price](https://github.com/chriskhanhtran/kaggle-house-price) — notebook Ames très propre

---

## 9. Journal des décisions

| Date | Décision | Raison |
|---|---|---|
| 2026-06-05 | Utiliser OpenML au lieu de Kaggle | Pas de token, portable en CI |
| 2026-06-05 | Python 3.11 | Stabilité + compatibilité libs ML |
| 2026-06-05 | Exclure SaleType/SaleCondition/MoSold/YrSold | Data leakage métier |
| 2026-06-05 | Slides sur Canva | Choix utilisateur |
| 2026-06-06 | Ajout DVC, Docker, FastAPI au stack | Nouvelles consignes du sujet (livrables attendus) |
| 2026-06-06 | Minimum 10 runs MLflow | Exigence explicite du sujet |
| 2026-06-06 | Streamlit comme bonus (pas Evidently) | Plus visuel pour soutenance, lien direct cas d'usage Laplace Immo |
| 2026-06-06 | Schéma d'architecture inclus | Optionnel mais gain facile pour points bonus |
| 2026-06-06 | Evidently / drift exclu du scope initial | À rajouter seulement si tout fini à J-3 |

---

## 10. Notes pour Claude (à lire au début de chaque session)

- L'utilisateur préfère **étape par étape**, sans donner trop à la fois.
- Donner les explications **avant** les commandes (pourquoi → quoi → comment).
- L'utilisateur est sur **Windows + Anaconda + VSCode**.
- Slides en **Canva** (pas dans le repo, mais on prépare le contenu).
- L'utilisateur veut **comprendre chaque ligne**, pas juste copier-coller.
- Toujours **mettre à jour ce fichier** après chaque étape complétée :
  - Cocher la case dans la section 6
  - Mettre à jour "État actuel" (section 0)
  - Ajouter une entrée au "Journal des décisions" si décision notable
- Si une exigence nouvelle apparaît (comme DVC/Docker/FastAPI au tour 2026-06-06), **réviser proactivement** l'architecture et les décisions concernées avant d'avancer.
