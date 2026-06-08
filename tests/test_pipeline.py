"""Tests unitaires pour src/pipeline.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.pipeline import (
    ORDINAL_QUALITES,
    LotFrontageImputer,
    construire_pipeline,
    identifier_colonnes,
)


def test_identifier_colonnes_retourne_trois_groupes(donnees_features):
    """identifier_colonnes doit retourner (num, ord, nom) tous non vides."""
    X = donnees_features.drop(columns=["SalePrice"])
    cols_num, cols_ord, cols_nom = identifier_colonnes(X)
    assert len(cols_num) > 0
    assert len(cols_ord) > 0
    assert len(cols_nom) > 0
    # Toutes les ordinales identifiees doivent etre dans la liste de reference
    for col in cols_ord:
        assert col in ORDINAL_QUALITES


def test_lotfrontage_imputer_impute_par_mediane_quartier():
    """LotFrontageImputer doit imputer les NaN par la mediane du quartier appris en fit."""
    df_train = pd.DataFrame({
        "Neighborhood": ["A", "A", "A", "B", "B"],
        "LotFrontage": [50.0, 60.0, 70.0, 100.0, 120.0],
    })
    df_test = pd.DataFrame({
        "Neighborhood": ["A", "B"],
        "LotFrontage": [np.nan, np.nan],
    })

    imputer = LotFrontageImputer()
    imputer.fit(df_train)
    result = imputer.transform(df_test)

    # Mediane de A = 60, mediane de B = 110
    assert result["LotFrontage"].tolist() == [60.0, 110.0]


def test_pipeline_end_to_end_produit_predictions_plausibles(donnees_features):
    """Le pipeline complet (preprocessing + Ridge) doit predire dans une plage log raisonnable."""
    X = donnees_features.drop(columns=["SalePrice"]).head(200)
    y = np.log1p(donnees_features["SalePrice"]).head(200)

    pipeline = Pipeline([
        ("preprocessing", construire_pipeline(X)),
        ("modele", Ridge(alpha=1.0)),
    ])
    pipeline.fit(X, y)
    predictions = pipeline.predict(X.head(5))

    # Plage realiste : log(35k) ~ 10.5, log(800k) ~ 13.6
    assert all(8 <= p <= 14 for p in predictions)
