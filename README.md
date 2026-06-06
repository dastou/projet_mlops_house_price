# Laplace Immo — Simulateur d'estimation de prix de maisons

Projet MLOps de prédiction de prix immobiliers sur le dataset Ames Iowa (Kaggle / OpenML).

> Ce README sera enrichi à la fin du projet. Pour le suivi de progression, voir [PROGRESSION.md](./PROGRESSION.md).

## Stack technique

Python 3.11 · scikit-learn · XGBoost · MLflow · DVC · FastAPI · Docker · Pytest · GitHub Actions · Streamlit

## Installation rapide

```bash
conda create -n mlops_immo python=3.11 -y
conda activate mlops_immo
pip install -r requirements.txt
```

## Structure du projet

```
projet_mlops/
├── data/           # données (DVC-tracked)
├── notebooks/      # analyses exploratoires
├── src/            # code source modulaire
├── api/            # API REST FastAPI
├── ui/             # interface Streamlit (bonus)
├── tests/          # tests unitaires Pytest
├── models/         # modèles entraînés (DVC-tracked)
├── docs/           # documentation architecture
└── .github/        # workflows CI/CD
```
