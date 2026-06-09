"""Stage DVC : preparation des donnees (load + clean + features + split).

Lit les parametres depuis params.yaml, applique la chaine deterministe de
preprocessing et produit les jeux train/test dans data/processed/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from src.data import charger_donnees, NOM_CIBLE
from src.features import creer_features
from src.preprocessing import nettoyer_donnees

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    chemin_projet = Path(__file__).resolve().parent.parent.parent

    with open(chemin_projet / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    test_size = params["prepare"]["test_size"]
    random_state = params["prepare"]["random_state"]
    logger.info("Parametres : test_size=%s, random_state=%s", test_size, random_state)

    # Chaine deterministe de preprocessing
    df = charger_donnees()
    df = nettoyer_donnees(df)
    df = creer_features(df)

    # Separation X / y avec log1p sur la cible
    X = df.drop(columns=[NOM_CIBLE])
    y = np.log1p(df[NOM_CIBLE])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Sauvegarde des jeux train/test
    sortie = chemin_projet / "data" / "processed"
    sortie.mkdir(parents=True, exist_ok=True)

    joblib.dump(X_train, sortie / "X_train.pkl")
    joblib.dump(X_test, sortie / "X_test.pkl")
    joblib.dump(y_train, sortie / "y_train.pkl")
    joblib.dump(y_test, sortie / "y_test.pkl")

    logger.info(
        "Donnees preparees : X_train %s, X_test %s",
        X_train.shape,
        X_test.shape,
    )


if __name__ == "__main__":
    main()
