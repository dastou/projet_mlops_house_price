"""
Module de creation de features pour le projet Laplace Immo.

Ce module enrichit le DataFrame nettoye avec de nouvelles features deriees
des variables existantes. Ces features ont ete identifiees en analyse
exploratoire comme ameliorant le pouvoir predictif des modeles.

Fonctions exposees :
    - creer_totalsf : surface totale habitable
    - creer_totalporchsf : surface totale des porches
    - creer_totalbathrooms : nombre equivalent de salles de bain completes
    - creer_ages : building_age, remodel_age, garage_age
    - creer_binaires : has_pool, has_garage, has_basement, has_fireplace, has_2ndfloor
    - creer_features : orchestrateur qui enchaine les 5 fonctions

Note importante sur les ages :
    Pendant l'entrainement, on calcule les ages avec YrSold (annee de vente reelle)
    pour rester fidele a la distribution historique. Pendant l'inference,
    on utilise une annee d'estimation fournie en parametre (ou la constante
    ANNEE_ESTIMATION_DEFAUT). YrSold est supprimee apres calcul des ages.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constantes : configuration du feature engineering
# ----------------------------------------------------------------------------

# Annee par defaut pour calculer les ages quand YrSold est absent (inference).
# A surcharger en production pour rester aligne avec l'annee courante.
ANNEE_ESTIMATION_DEFAUT = 2026

# Colonnes utilisees pour calculer TotalSF
COLS_SURFACES_INTERIEURES = ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]

# Colonnes utilisees pour calculer TotalPorchSF
COLS_PORCHES = [
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "WoodDeckSF",
]

# Colonnes utilisees pour calculer TotalBathrooms
COLS_BATHROOMS_FULL = ["FullBath", "BsmtFullBath"]
COLS_BATHROOMS_HALF = ["HalfBath", "BsmtHalfBath"]


# ----------------------------------------------------------------------------
# Fonctions de creation de features (pures)
# ----------------------------------------------------------------------------

def creer_totalsf(df: pd.DataFrame) -> pd.DataFrame:
    """Cree la feature TotalSF : surface habitable totale.

    TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF.
    Combine 3 features fortement correlees en une seule plus predictive.

    Args:
        df: DataFrame contenant les colonnes de surfaces interieures.

    Returns:
        Nouveau DataFrame avec la colonne TotalSF ajoutee.
    """
    df = df.copy()
    df["TotalSF"] = df[COLS_SURFACES_INTERIEURES].sum(axis=1)
    logger.info("creer_totalsf : feature TotalSF ajoutee")
    return df


def creer_totalporchsf(df: pd.DataFrame) -> pd.DataFrame:
    """Cree la feature TotalPorchSF : surface exterieure totale.

    TotalPorchSF = somme des 5 types de porches et terrasses
    (OpenPorch, EnclosedPorch, 3SsnPorch, ScreenPorch, WoodDeck).

    Args:
        df: DataFrame contenant les colonnes de porches.

    Returns:
        Nouveau DataFrame avec la colonne TotalPorchSF ajoutee.
    """
    df = df.copy()
    df["TotalPorchSF"] = df[COLS_PORCHES].sum(axis=1)
    logger.info("creer_totalporchsf : feature TotalPorchSF ajoutee")
    return df


def creer_totalbathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """Cree la feature TotalBathrooms : nombre equivalent de SdB completes.

    TotalBathrooms = FullBath + BsmtFullBath + 0.5 * (HalfBath + BsmtHalfBath).
    Une demi salle de bain est ponderee 0.5 dans le total.

    Args:
        df: DataFrame contenant les 4 colonnes de salles de bain.

    Returns:
        Nouveau DataFrame avec la colonne TotalBathrooms ajoutee.
    """
    df = df.copy()
    df["TotalBathrooms"] = (
        df[COLS_BATHROOMS_FULL].sum(axis=1)
        + 0.5 * df[COLS_BATHROOMS_HALF].sum(axis=1)
    )
    logger.info("creer_totalbathrooms : feature TotalBathrooms ajoutee")
    return df


def creer_ages(
    df: pd.DataFrame,
    annee_estimation: int | None = None,
) -> pd.DataFrame:
    """Cree les features d'age et supprime YrSold.

    Trois ages sont calcules :
        - building_age = annee_ref - YearBuilt
        - remodel_age  = annee_ref - YearRemodAdd
        - garage_age   = annee_ref - GarageYrBlt
            (convention : garage_age = building_age si pas de garage,
            cad si GarageYrBlt == 0 apres imputation)

    L'annee de reference est :
        - YrSold si la colonne est presente (cas entrainement)
        - annee_estimation si fourni (cas inference)
        - ANNEE_ESTIMATION_DEFAUT sinon

    YrSold est supprimee a la fin pour eviter le data leakage metier.

    Args:
        df: DataFrame contenant YearBuilt, YearRemodAdd, GarageYrBlt et
            optionnellement YrSold.
        annee_estimation: annee a utiliser si YrSold est absente.

    Returns:
        Nouveau DataFrame avec les 3 ages et sans YrSold.
    """
    df = df.copy()

    if "YrSold" in df.columns:
        annee_ref = df["YrSold"]
        source = "YrSold"
    elif annee_estimation is not None:
        annee_ref = annee_estimation
        source = f"annee_estimation={annee_estimation}"
    else:
        annee_ref = ANNEE_ESTIMATION_DEFAUT
        source = f"ANNEE_ESTIMATION_DEFAUT={ANNEE_ESTIMATION_DEFAUT}"

    df["building_age"] = annee_ref - df["YearBuilt"]
    df["remodel_age"] = annee_ref - df["YearRemodAdd"]
    # Pour les maisons sans garage (GarageYrBlt == 0 apres imputation),
    # on utilise building_age comme convention raisonnable.
    df["garage_age"] = np.where(
        df["GarageYrBlt"] == 0,
        df["building_age"],
        annee_ref - df["GarageYrBlt"],
    )

    if "YrSold" in df.columns:
        df = df.drop(columns=["YrSold"])

    logger.info(
        "creer_ages : building_age, remodel_age, garage_age ajoutes (source: %s)",
        source,
    )
    return df


def creer_binaires(df: pd.DataFrame) -> pd.DataFrame:
    """Cree des features binaires de presence/absence d'equipements.

    Variables creees :
        - has_pool       : 1 si PoolArea > 0
        - has_garage     : 1 si GarageArea > 0
        - has_basement   : 1 si TotalBsmtSF > 0
        - has_fireplace  : 1 si Fireplaces > 0
        - has_2ndfloor   : 1 si 2ndFlrSF > 0

    Ces features donnent un signal explicite au modele sur la presence
    d'equipements, en complement des variables continues correspondantes.

    Args:
        df: DataFrame contenant les colonnes de reference.

    Returns:
        Nouveau DataFrame avec les 5 features binaires ajoutees.
    """
    df = df.copy()
    df["has_pool"] = (df["PoolArea"] > 0).astype(int)
    df["has_garage"] = (df["GarageArea"] > 0).astype(int)
    df["has_basement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["has_fireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["has_2ndfloor"] = (df["2ndFlrSF"] > 0).astype(int)
    logger.info("creer_binaires : 5 features binaires ajoutees")
    return df


# ----------------------------------------------------------------------------
# Orchestrateur
# ----------------------------------------------------------------------------

def creer_features(
    df: pd.DataFrame,
    annee_estimation: int | None = None,
) -> pd.DataFrame:
    """Applique l'ensemble du feature engineering.

    Ordre d'execution :
        1. creer_totalsf
        2. creer_totalporchsf
        3. creer_totalbathrooms
        4. creer_binaires (avant les ages : pas de dependance avec YrSold)
        5. creer_ages (en dernier : supprime YrSold)

    Args:
        df: DataFrame nettoye (typiquement issu de nettoyer_donnees).
        annee_estimation: annee a utiliser pour les ages en mode inference.

    Returns:
        Nouveau DataFrame enrichi de 11 nouvelles features.
    """
    logger.info("Demarrage du feature engineering...")
    logger.info("Shape avant FE : %d lignes x %d colonnes", df.shape[0], df.shape[1])

    df = creer_totalsf(df)
    df = creer_totalporchsf(df)
    df = creer_totalbathrooms(df)
    df = creer_binaires(df)
    df = creer_ages(df, annee_estimation=annee_estimation)

    logger.info("Shape apres FE : %d lignes x %d colonnes", df.shape[0], df.shape[1])
    return df


# ----------------------------------------------------------------------------
# Test rapide en standalone : python -m src.features
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from src.data import charger_donnees
    from src.preprocessing import nettoyer_donnees

    df_brut = charger_donnees()
    df_clean = nettoyer_donnees(df_brut)
    df_feat = creer_features(df_clean)

    nouvelles_features = [
        "TotalSF",
        "TotalPorchSF",
        "TotalBathrooms",
        "building_age",
        "remodel_age",
        "garage_age",
        "has_pool",
        "has_garage",
        "has_basement",
        "has_fireplace",
        "has_2ndfloor",
    ]

    print(f"\nShape final : {df_feat.shape}")
    print(f"YrSold encore presente ? {'YrSold' in df_feat.columns}")

    print(f"\nStatistiques des nouvelles features :")
    print(df_feat[nouvelles_features].describe().T.round(2))

    print(f"\nPart de maisons avec chaque equipement :")
    for col in ["has_pool", "has_garage", "has_basement", "has_fireplace", "has_2ndfloor"]:
        taux = df_feat[col].mean() * 100
        print(f"  {col:18s} : {taux:5.1f} %")
