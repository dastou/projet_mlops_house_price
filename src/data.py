"""
Module de chargement des donnees Ames Iowa pour le projet Laplace Immo.

Ce module expose deux fonctions :
- charger_donnees() : telecharge le dataset depuis OpenML
- separer_features_cible() : separe les features (X) et la cible (y)

Source du dataset :
    OpenML id 42165 (mirror de la competition Kaggle "House Prices").
    1460 maisons d'Ames (Iowa, USA) decrites par 79 variables explicatives
    + la cible SalePrice (prix de vente en dollars).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

# Logger nomme du module : permet d'activer/desactiver les logs sans
# toucher au code et de les rediriger vers un fichier en CI/CD.
logger = logging.getLogger(__name__)

# Constante du nom de la colonne cible. Centralisee ici pour eviter
# les fautes de frappe et faciliter un eventuel renommage.
NOM_CIBLE = "SalePrice"


def charger_donnees(
    sauvegarder: bool = False,
    chemin_sortie: Path | None = None,
) -> pd.DataFrame:
    """Charge le dataset Ames Iowa depuis OpenML.

    Le dataset est mis en cache localement par scikit-learn apres le
    premier telechargement (dans ~/scikit_learn_data/), donc les appels
    suivants sont instantanes.

    Args:
        sauvegarder: si True, ecrit le DataFrame en CSV pour le pipeline DVC.
        chemin_sortie: chemin du CSV de sortie. Par defaut, data/raw/ames.csv.

    Returns:
        DataFrame de 1460 lignes contenant les 79 features + SalePrice.

    Raises:
        RuntimeError: si le telechargement OpenML echoue (souvent reseau).
    """
    logger.info("Telechargement du dataset Ames depuis OpenML (id=42165)...")

    try:
        dataset = fetch_openml(
            name="house_prices",
            version="active",
            as_frame=True,
            parser="auto",
        )
    except Exception as e:
        raise RuntimeError("Echec du telechargement OpenML. Verifier la connexion internet.") from e

    # On combine les features et la cible en un seul DataFrame.
    # Pratique pour l'EDA : tout est explorable d'un coup.
    df = dataset.data.copy()
    df[NOM_CIBLE] = pd.to_numeric(dataset.target, errors="raise")

    logger.info(
        "Dataset charge : %d lignes x %d colonnes",
        df.shape[0],
        df.shape[1],
    )

    if sauvegarder:
        if chemin_sortie is None:
            chemin_sortie = Path("data/raw/ames.csv")
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(chemin_sortie, index=False)
        logger.info("Donnees sauvegardees dans %s", chemin_sortie)

    return df


def separer_features_cible(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separe le DataFrame en features (X) et cible (y).

    Args:
        df: DataFrame contenant les features et la colonne SalePrice.

    Returns:
        Tuple (X, y) ou X est le DataFrame des features (sans SalePrice)
        et y la Series du prix de vente.

    Raises:
        ValueError: si la colonne SalePrice est absente du DataFrame.
    """
    if NOM_CIBLE not in df.columns:
        raise ValueError(f"La colonne cible '{NOM_CIBLE}' est absente du DataFrame.")

    X = df.drop(columns=[NOM_CIBLE])
    y = df[NOM_CIBLE]
    return X, y


if __name__ == "__main__":
    # Permet de tester rapidement depuis la ligne de commande :
    #   python -m src.data
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    donnees = charger_donnees(sauvegarder=True)
    X, y = separer_features_cible(donnees)
    print(f"\nShape de X : {X.shape}")
    print(f"Shape de y : {y.shape}")
    print(f"\nPrix moyen  : {y.mean():>10,.0f} $")
    print(f"Prix median : {y.median():>10,.0f} $")
    print(f"Prix min    : {y.min():>10,.0f} $")
    print(f"Prix max    : {y.max():>10,.0f} $")
