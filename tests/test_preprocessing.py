"""Tests unitaires pour src/preprocessing.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import (
    VARS_LEAKAGE_METIER,
    exclure_variables_leakage,
    imputer_nan_categorielles,
    imputer_nan_numeriques,
    supprimer_outliers,
)


def test_supprimer_outliers_retire_partial_grande_surface():
    """Les ventes Partial avec GrLivArea > 4000 doivent etre supprimees."""
    df = pd.DataFrame({
        "GrLivArea": [5000, 3000, 4500, 5500],
        "SaleCondition": ["Partial", "Normal", "Normal", "Partial"],
    })
    result = supprimer_outliers(df)
    # On garde les 2 maisons Normal et on retire les 2 Partial > 4000
    assert len(result) == 2
    assert all(result["SaleCondition"] == "Normal")


def test_imputer_nan_categorielles_remplace_par_none():
    """Les NaN sur les colonnes structurelles doivent devenir 'None'."""
    df = pd.DataFrame({
        "PoolQC": [None, "Ex", None],
        "Alley": [None, None, "Pave"],
        "AutreCol": [1, 2, 3],  # ne doit pas etre touchee
    })
    result = imputer_nan_categorielles(df)
    assert result["PoolQC"].tolist() == ["None", "Ex", "None"]
    assert result["Alley"].tolist() == ["None", "None", "Pave"]
    assert result["AutreCol"].tolist() == [1, 2, 3]


def test_imputer_nan_numeriques_remplace_par_zero():
    """Les NaN sur GarageYrBlt et MasVnrArea doivent devenir 0."""
    df = pd.DataFrame({
        "GarageYrBlt": [2003.0, np.nan, 1995.0],
        "MasVnrArea": [np.nan, 100.0, np.nan],
    })
    result = imputer_nan_numeriques(df)
    assert result["GarageYrBlt"].tolist() == [2003.0, 0.0, 1995.0]
    assert result["MasVnrArea"].tolist() == [0.0, 100.0, 0.0]


def test_exclure_variables_leakage_conserve_yrsold():
    """SaleType, SaleCondition, MoSold sont retires (mais PAS YrSold)."""
    df = pd.DataFrame({
        "SaleType": ["WD"],
        "SaleCondition": ["Normal"],
        "MoSold": [6],
        "YrSold": [2008],
        "GrLivArea": [1500],
    })
    result = exclure_variables_leakage(df)
    # Les 3 variables de leakage sont supprimees
    for var in VARS_LEAKAGE_METIER:
        assert var not in result.columns
    # YrSold est conserve volontairement pour le calcul des ages
    assert "YrSold" in result.columns


def test_nettoyer_donnees_shape_finale(donnees_propres):
    """Apres nettoyage complet : 1458 lignes (2 outliers Partial) x 77 colonnes."""
    assert donnees_propres.shape == (1458, 77)
