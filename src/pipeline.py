"""
Module de construction du pipeline sklearn anti-leakage pour Laplace Immo.

Ce module construit le preprocesseur statistique qui complete les fonctions
deterministes de preprocessing.py et features.py. Il s'integre dans un
sklearn Pipeline pour garantir que les statistiques (medianes, modes, ecarts
types, modalites OneHot) sont apprises uniquement sur les donnees
d'entrainement et appliquees identiquement sur le test.

Architecture cible :

    Pipeline([
        ("lotfrontage_imputer", LotFrontageImputer()),  # mediane par quartier
        ("column_transformer", ColumnTransformer([
            ("num", num_pipeline, cols_num),    # SimpleImputer + StandardScaler
            ("ord", ord_pipeline, cols_ord),    # SimpleImputer + OrdinalEncoder
            ("nom", nom_pipeline, cols_nom),    # SimpleImputer + OneHotEncoder
        ])),
    ])

Aucun modele n'est inclus dans ce pipeline : il retourne une matrice numerique
prete pour n'importe quel estimateur sklearn. Le modele est ajoute plus tard
via Pipeline([("preprocessing", pipeline), ("modele", ...)])  pour permettre
de tester plusieurs modeles avec le meme preprocessing.

Fonctions et classes exposees :
    - LotFrontageImputer : custom transformer pour LotFrontage
    - identifier_colonnes : retourne les listes (num, ord, nom)
    - construire_preprocessor : construit le ColumnTransformer
    - construire_pipeline : construit le Pipeline complet (sans modele)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constantes : configuration des encodages
# ----------------------------------------------------------------------------

# Variables de qualite ordinales : echelle Po (Poor) < Fa (Fair) <
# TA (Typical) < Gd (Good) < Ex (Excellent).
# "None" est ajoute en premier pour les maisons sans equipement correspondant
# (apres imputation par preprocessing.imputer_nan_categorielles).
ORDINAL_QUALITES = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "PoolQC",
    "GarageQual",
    "GarageCond",
]

# Ordre commun pour les variables ci-dessus.
ORDRE_QUALITES = ["None", "Po", "Fa", "TA", "Gd", "Ex"]


# ----------------------------------------------------------------------------
# Custom transformer : imputation de LotFrontage par mediane de quartier
# ----------------------------------------------------------------------------


class LotFrontageImputer(BaseEstimator, TransformerMixin):
    """Impute les NaN de LotFrontage par la mediane de chaque quartier.

    Pendant fit(), calcule la mediane de LotFrontage pour chaque valeur de
    Neighborhood, ainsi qu'une mediane globale (fallback pour les quartiers
    inconnus a l'inference).

    Pendant transform(), remplace les NaN de LotFrontage par la mediane du
    quartier correspondant. Si le quartier n'a pas ete vu au fit, on utilise
    la mediane globale.

    Si LotFrontage ou Neighborhood sont absents du DataFrame, l'imputer fait
    un no-op (passe les donnees sans modification).

    Attributes:
        medianes_ : pd.Series indexee par Neighborhood, mediane apprise sur train.
        mediane_globale_ : float, mediane globale apprise sur train.
    """

    def fit(self, X, y=None):
        """Apprend les medianes par quartier sur X (train)."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LotFrontageImputer attend un DataFrame pandas.")

        if "LotFrontage" not in X.columns or "Neighborhood" not in X.columns:
            self.medianes_ = pd.Series(dtype="float64")
            self.mediane_globale_ = float("nan")
            return self

        self.medianes_ = X.groupby("Neighborhood")["LotFrontage"].median()
        self.mediane_globale_ = float(X["LotFrontage"].median())

        logger.info(
            "LotFrontageImputer fit : %d quartiers appris, mediane globale = %.1f",
            len(self.medianes_),
            self.mediane_globale_,
        )
        return self

    def transform(self, X):
        """Applique les medianes apprises pour imputer les NaN de LotFrontage."""
        check_is_fitted(self, attributes=["medianes_", "mediane_globale_"])

        if not isinstance(X, pd.DataFrame):
            raise TypeError("LotFrontageImputer attend un DataFrame pandas.")

        if "LotFrontage" not in X.columns or "Neighborhood" not in X.columns:
            return X.copy()

        X = X.copy()
        masque_nan = X["LotFrontage"].isna()
        nb_nan = int(masque_nan.sum())

        if nb_nan == 0:
            return X

        # Map quartier -> mediane apprise. Fallback sur la mediane globale.
        quartiers_nan = X.loc[masque_nan, "Neighborhood"]
        valeurs_imputees = quartiers_nan.map(self.medianes_).fillna(self.mediane_globale_)
        X.loc[masque_nan, "LotFrontage"] = valeurs_imputees.values

        logger.info("LotFrontageImputer transform : %d NaN imputes", nb_nan)
        return X


# ----------------------------------------------------------------------------
# Identification des groupes de colonnes
# ----------------------------------------------------------------------------


def identifier_colonnes(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Identifie les trois groupes de colonnes pour le ColumnTransformer.

    Args:
        X: DataFrame d'entree (sans la colonne cible).

    Returns:
        Tuple (cols_num, cols_ord, cols_nom) :
            - cols_num : colonnes numeriques (int, float)
            - cols_ord : colonnes ordinales (10 qualites Po-Ex presentes dans X)
            - cols_nom : autres colonnes catégorielles (object non ordinales)
    """
    cols_ord = [c for c in ORDINAL_QUALITES if c in X.columns]
    cols_nom = [c for c in X.select_dtypes(include="object").columns if c not in cols_ord]
    cols_num = list(X.select_dtypes(include=["number"]).columns)
    return cols_num, cols_ord, cols_nom


# ----------------------------------------------------------------------------
# Construction du ColumnTransformer et du Pipeline complet
# ----------------------------------------------------------------------------


def construire_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construit le ColumnTransformer pour le preprocessing statistique.

    Trois sous-pipelines :
        - num : SimpleImputer(median) + StandardScaler
        - ord : SimpleImputer(constant="None") + OrdinalEncoder(echelle Po-Ex)
        - nom : SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown="ignore")

    Args:
        X: DataFrame echantillon pour identifier les colonnes (typiquement X_train).

    Returns:
        ColumnTransformer pret a etre fitte.
    """
    cols_num, cols_ord, cols_nom = identifier_colonnes(X)

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    ord_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[ORDRE_QUALITES] * len(cols_ord),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    nom_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, cols_num),
            ("ord", ord_pipeline, cols_ord),
            ("nom", nom_pipeline, cols_nom),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def construire_pipeline(X: pd.DataFrame) -> Pipeline:
    """Construit le pipeline complet de preprocessing (sans modele).

    Le pipeline enchaine :
        1. LotFrontageImputer (custom : mediane par quartier)
        2. ColumnTransformer (imputation + encodage + scaling sur tous les groupes)

    Aucun modele n'est inclus. Pour l'entrainement, wrapper dans un autre
    Pipeline avec le modele de son choix :

        pipeline_complet = Pipeline([
            ("preprocessing", construire_pipeline(X_train)),
            ("modele", Ridge(alpha=1.0)),
        ])

    Args:
        X: DataFrame echantillon pour identifier les colonnes.

    Returns:
        Pipeline sklearn fittable sur X_train.
    """
    return Pipeline(
        [
            ("lotfrontage_imputer", LotFrontageImputer()),
            ("column_transformer", construire_preprocessor(X)),
        ]
    )


# ----------------------------------------------------------------------------
# Test rapide en standalone : python -m src.pipeline
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    from src.data import charger_donnees, NOM_CIBLE
    from src.features import creer_features
    from src.preprocessing import nettoyer_donnees

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 1. Chargement et preparation deterministe
    df = charger_donnees()
    df = nettoyer_donnees(df)
    df = creer_features(df)

    # 2. Separation X / y (log1p sur la cible)
    X = df.drop(columns=[NOM_CIBLE])
    y = np.log1p(df[NOM_CIBLE])
    print(f"\nShape X : {X.shape}, shape y : {y.shape}")

    # 3. Split train / test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train : {X_train.shape}, Test : {X_test.shape}")

    # 4. Identification des groupes de colonnes
    cols_num, cols_ord, cols_nom = identifier_colonnes(X_train)
    print("\nGroupes de colonnes identifies :")
    print(f"  Numeriques : {len(cols_num)} colonnes")
    print(f"  Ordinales  : {len(cols_ord)} colonnes")
    print(f"    -> {cols_ord}")
    print(f"  Nominales  : {len(cols_nom)} colonnes")

    # 5. Construction et application du pipeline de preprocessing
    pipeline = construire_pipeline(X_train)
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    print("\nApres preprocessing :")
    print(f"  X_train_processed : shape {X_train_processed.shape}, dtype {X_train_processed.dtype}")
    print(f"  X_test_processed  : shape {X_test_processed.shape}")
    print(f"  NaN restants dans X_train_processed : {int(np.isnan(X_train_processed).sum())}")

    # 6. Sanity check : entrainer un Ridge baseline
    print("\nSanity check avec Ridge :")
    pipeline_complet = Pipeline(
        [
            ("preprocessing", construire_pipeline(X_train)),
            ("modele", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    pipeline_complet.fit(X_train, y_train)
    y_pred_log = pipeline_complet.predict(X_test)

    rmse_log = float(np.sqrt(mean_squared_error(y_test, y_pred_log)))
    r2 = float(r2_score(y_test, y_pred_log))

    y_test_dollars = np.expm1(y_test)
    y_pred_dollars = np.expm1(y_pred_log)
    rmse_dollars = float(np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars)))

    print(f"  RMSE (log1p) : {rmse_log:.4f}")
    print(f"  RMSE (dollars) : {rmse_dollars:,.0f} $")
    print(f"  R2 : {r2:.4f}")
