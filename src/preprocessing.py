"""
Module de nettoyage deterministe des donnees Ames Iowa.

Ce module applique uniquement les operations de nettoyage qui ne dependent
pas des donnees elles-memes (pas de calcul de statistiques) et peuvent donc
etre appliquees avant le split train/test sans risque de data leakage.

Les operations qui apprennent des statistiques (imputation par mediane par
groupe, encodage, scaling) sont implementees dans le module pipeline.

Fonctions exposees :
    - supprimer_outliers : retire les ventes Partial avec GrLivArea > 4000
    - imputer_nan_categorielles : remplace NaN par "None" sur 15 colonnes
    - imputer_nan_numeriques : remplace NaN par 0 sur 2 colonnes
    - supprimer_id : retire la colonne identifiant
    - exclure_variables_leakage : retire 4 variables connues seulement apres vente
    - nettoyer_donnees : orchestrateur qui enchaine les 5 fonctions ci-dessus
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constantes : variables a traiter
# ----------------------------------------------------------------------------

# Variables categorielles dont le NaN signifie "non applicable" (= absence
# d'equipement). Elles sont imputees par la string "None" pour creer une
# categorie "absence" exploitable par les encodeurs.
COLS_NAN_STRUCTURELLES_CATEGORIELLES = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
]

# Variables numeriques liees aux equipements absents : NaN signifie 0.
# Exemple : pas de garage implique GarageYrBlt = 0, MasVnrArea = 0.
COLS_NAN_STRUCTURELLES_NUMERIQUES = [
    "GarageYrBlt",
    "MasVnrArea",
]

# Variables connues seulement au moment de la signature de la vente (T5).
# Inutilisables pour estimer un prix avant mise en vente (T1) : data leakage metier.
VARS_LEAKAGE_METIER = [
    "SaleType",
    "SaleCondition",
    "MoSold",
    "YrSold",
]

# Variables sans pouvoir predictif a retirer du dataset.
VARS_INUTILES = ["Id"]


# ----------------------------------------------------------------------------
# Fonctions de nettoyage (deterministes)
# ----------------------------------------------------------------------------

def supprimer_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les outliers identifies en EDA.

    Critere : GrLivArea > 4000 sq ft ET SaleCondition = "Partial".
    Ces ventes correspondent a des constructions inachevees vendues
    prematurement et ne representent pas le marche normal.

    Args:
        df: DataFrame contenant au minimum les colonnes GrLivArea et SaleCondition.

    Returns:
        Nouveau DataFrame sans les outliers.
    """
    avant = len(df)
    masque = (df["GrLivArea"] > 4000) & (df["SaleCondition"] == "Partial")
    df_clean = df[~masque].copy()
    logger.info(
        "supprimer_outliers : %d lignes supprimees (avant %d, apres %d)",
        avant - len(df_clean),
        avant,
        len(df_clean),
    )
    return df_clean


def imputer_nan_categorielles(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les NaN structurels categoriels par la string 'None'.

    Pour les variables decrivant un equipement (piscine, garage, sous-sol),
    NaN signifie 'pas d'equipement'. On le transforme en categorie 'None'
    afin que les encodeurs (OneHot, Ordinal) puissent la traiter comme
    une modalite a part entiere.

    Args:
        df: DataFrame contenant les colonnes structurelles categorielles.

    Returns:
        Nouveau DataFrame avec NaN remplaces par 'None'.
    """
    df = df.copy()
    cols_presentes = [c for c in COLS_NAN_STRUCTURELLES_CATEGORIELLES if c in df.columns]
    for col in cols_presentes:
        df[col] = df[col].fillna("None")
    logger.info(
        "imputer_nan_categorielles : %d colonnes traitees", len(cols_presentes)
    )
    return df


def imputer_nan_numeriques(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les NaN structurels numeriques par 0.

    Pour les variables numeriques liees a un equipement absent (annee de
    construction du garage, surface du parement maconne), NaN signifie 0.

    Args:
        df: DataFrame contenant les colonnes structurelles numeriques.

    Returns:
        Nouveau DataFrame avec NaN remplaces par 0.
    """
    df = df.copy()
    cols_presentes = [c for c in COLS_NAN_STRUCTURELLES_NUMERIQUES if c in df.columns]
    for col in cols_presentes:
        df[col] = df[col].fillna(0)
    logger.info(
        "imputer_nan_numeriques : %d colonnes traitees", len(cols_presentes)
    )
    return df


def supprimer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime la colonne Id si presente.

    L'identifiant n'a aucune valeur predictive et risque meme de creer
    du bruit (correlation avec l'ordre d'enregistrement).

    Args:
        df: DataFrame avec ou sans colonne Id.

    Returns:
        Nouveau DataFrame sans colonne Id.
    """
    if "Id" in df.columns:
        return df.drop(columns=["Id"])
    return df.copy()


def exclure_variables_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les 4 variables identifiees comme data leakage metier.

    Variables exclues : SaleType, SaleCondition, MoSold, YrSold.
    Elles decrivent la transaction et sont inconnues au moment ou
    Laplace Immo souhaite estimer le prix (avant mise en vente).

    Args:
        df: DataFrame pouvant contenir ces variables.

    Returns:
        Nouveau DataFrame sans les variables de leakage.
    """
    cols_a_drop = [c for c in VARS_LEAKAGE_METIER if c in df.columns]
    if not cols_a_drop:
        return df.copy()
    logger.info("exclure_variables_leakage : %s", cols_a_drop)
    return df.drop(columns=cols_a_drop)


# ----------------------------------------------------------------------------
# Orchestrateur
# ----------------------------------------------------------------------------

def nettoyer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """Applique l'ensemble du nettoyage deterministe dans le bon ordre.

    Ordre d'execution (important) :
        1. supprimer_outliers : avant l'exclusion de SaleCondition, car le
           filtre des outliers utilise cette colonne.
        2. imputer_nan_categorielles : transforme les NaN en string "None".
        3. imputer_nan_numeriques : transforme les NaN en 0.
        4. supprimer_id : retire la colonne Id.
        5. exclure_variables_leakage : en dernier, retire les 4 variables.

    Args:
        df: DataFrame brut tel que retourne par charger_donnees().

    Returns:
        Nouveau DataFrame nettoye, pret pour le feature engineering.
    """
    logger.info("Demarrage du nettoyage deterministe...")
    logger.info("Shape initial : %d lignes x %d colonnes", df.shape[0], df.shape[1])

    df = supprimer_outliers(df)
    df = imputer_nan_categorielles(df)
    df = imputer_nan_numeriques(df)
    df = supprimer_id(df)
    df = exclure_variables_leakage(df)

    logger.info("Shape final : %d lignes x %d colonnes", df.shape[0], df.shape[1])
    return df


# ----------------------------------------------------------------------------
# Test rapide en standalone : python -m src.preprocessing
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from src.data import charger_donnees

    donnees_brutes = charger_donnees()
    print(f"\nAvant nettoyage : {donnees_brutes.shape}")
    print(f"Nb NaN total avant : {donnees_brutes.isna().sum().sum()}")

    donnees_propres = nettoyer_donnees(donnees_brutes)
    print(f"\nApres nettoyage : {donnees_propres.shape}")
    print(f"Nb NaN total apres : {donnees_propres.isna().sum().sum()}")

    # Verification : seuls LotFrontage et Electrical doivent encore avoir des NaN
    # (traites dans le pipeline statistique a l'etape 7).
    nan_restants = donnees_propres.isna().sum()
    nan_restants = nan_restants[nan_restants > 0].sort_values(ascending=False)
    print(f"\nColonnes avec NaN restants (traitees a l'etape pipeline) :")
    print(nan_restants)
