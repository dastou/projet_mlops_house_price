"""Stage DVC : evaluation du modele final sur le jeu de test.

Calcule les metriques (RMSE log, R2, RMSE dollars, MAE dollars) et les
ecrit dans metrics/metrics.json. DVC suit ce fichier pour comparer les
performances entre runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    chemin_projet = Path(__file__).resolve().parent.parent.parent

    # Chargement du modele et du jeu de test
    pipeline = joblib.load(chemin_projet / "models" / "final_model.pkl")
    X_test = joblib.load(chemin_projet / "data" / "processed" / "X_test.pkl")
    y_test = joblib.load(chemin_projet / "data" / "processed" / "y_test.pkl")
    logger.info("Modele et test charges. X_test : %s", X_test.shape)

    # Predictions en log
    y_pred_log = pipeline.predict(X_test)

    # Metriques en log (metrique Kaggle officielle)
    test_rmse_log = float(np.sqrt(mean_squared_error(y_test, y_pred_log)))
    test_r2 = float(r2_score(y_test, y_pred_log))

    # Metriques en dollars (lisibilite metier)
    y_test_dollars = np.expm1(y_test)
    y_pred_dollars = np.expm1(y_pred_log)
    test_rmse_dollars = float(np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars)))
    test_mae_dollars = float(mean_absolute_error(y_test_dollars, y_pred_dollars))

    metriques = {
        "test_rmse_log": round(test_rmse_log, 6),
        "test_r2": round(test_r2, 6),
        "test_rmse_dollars": round(test_rmse_dollars, 2),
        "test_mae_dollars": round(test_mae_dollars, 2),
    }

    # Sauvegarde dans metrics/metrics.json (tracke par DVC, garde en git)
    sortie = chemin_projet / "metrics"
    sortie.mkdir(parents=True, exist_ok=True)

    chemin_metriques = sortie / "metrics.json"
    with open(chemin_metriques, "w", encoding="utf-8") as f:
        json.dump(metriques, f, indent=2)

    logger.info("Metriques sauvegardees dans %s", chemin_metriques)
    for cle, valeur in metriques.items():
        logger.info("  %s : %s", cle, valeur)


if __name__ == "__main__":
    main()
