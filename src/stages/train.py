"""Stage DVC : entrainement du modele CatBoost final.

Lit les donnees preprocessed et les hyperparametres CatBoost depuis
params.yaml, entraine le pipeline complet (preprocessing + modele) et
sauvegarde le modele dans models/final_model.pkl.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import yaml
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

from src.pipeline import construire_pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    chemin_projet = Path(__file__).resolve().parent.parent.parent

    with open(chemin_projet / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    params_train = params["train"]
    logger.info("Parametres CatBoost : %s", params_train)

    # Chargement des donnees preprocessed
    X_train = joblib.load(chemin_projet / "data" / "processed" / "X_train.pkl")
    y_train = joblib.load(chemin_projet / "data" / "processed" / "y_train.pkl")
    logger.info("Donnees train chargees : %s", X_train.shape)

    # Construction du pipeline complet : preprocessing + CatBoost
    pipeline = Pipeline([
        ("preprocessing", construire_pipeline(X_train)),
        ("modele", CatBoostRegressor(
            iterations=params_train["iterations"],
            depth=params_train["depth"],
            learning_rate=params_train["learning_rate"],
            l2_leaf_reg=params_train["l2_leaf_reg"],
            random_state=params_train["random_state"],
            verbose=False,
            allow_writing_files=False,
        )),
    ])

    pipeline.fit(X_train, y_train)
    logger.info("Entrainement termine")

    # Sauvegarde du pipeline complet
    sortie = chemin_projet / "models"
    sortie.mkdir(parents=True, exist_ok=True)
    chemin_modele = sortie / "final_model.pkl"
    joblib.dump(pipeline, chemin_modele)

    taille_mo = chemin_modele.stat().st_size / (1024 * 1024)
    logger.info("Modele sauvegarde : %s (%.2f Mo)", chemin_modele, taille_mo)


if __name__ == "__main__":
    main()
