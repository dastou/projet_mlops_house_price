"""Tests unitaires pour src/features.py."""
from __future__ import annotations

import pandas as pd

from src.features import (
    ANNEE_ESTIMATION_DEFAUT,
    creer_ages,
    creer_binaires,
    creer_totalbathrooms,
    creer_totalsf,
)


def test_creer_totalsf_somme_correcte():
    """TotalSF doit etre la somme de TotalBsmtSF + 1stFlrSF + 2ndFlrSF."""
    df = pd.DataFrame({
        "TotalBsmtSF": [100, 200],
        "1stFlrSF": [500, 600],
        "2ndFlrSF": [0, 700],
    })
    result = creer_totalsf(df)
    assert result["TotalSF"].tolist() == [600, 1500]


def test_creer_totalbathrooms_ponderation_demi_sdb():
    """Les demi-sdb doivent etre ponderees par 0.5 dans TotalBathrooms."""
    df = pd.DataFrame({
        "FullBath": [2],
        "HalfBath": [1],
        "BsmtFullBath": [1],
        "BsmtHalfBath": [0],
    })
    result = creer_totalbathrooms(df)
    # 2 (FullBath) + 0.5 (HalfBath) + 1 (BsmtFullBath) + 0 (BsmtHalfBath) = 3.5
    assert result["TotalBathrooms"].iloc[0] == 3.5


def test_creer_binaires_indique_presence_equipements():
    """has_X doit valoir 1 si l'equipement est present, 0 sinon."""
    df = pd.DataFrame({
        "PoolArea": [0, 100],
        "GarageArea": [548, 0],
        "TotalBsmtSF": [856, 0],
        "Fireplaces": [0, 2],
        "2ndFlrSF": [854, 0],
    })
    result = creer_binaires(df)
    assert result["has_pool"].tolist() == [0, 1]
    assert result["has_garage"].tolist() == [1, 0]
    assert result["has_basement"].tolist() == [1, 0]
    assert result["has_fireplace"].tolist() == [0, 1]
    assert result["has_2ndfloor"].tolist() == [1, 0]


def test_creer_ages_avec_yrsold_utilise_yrsold():
    """En presence de YrSold : building_age = YrSold - YearBuilt, puis YrSold est supprime."""
    df = pd.DataFrame({
        "YearBuilt": [2000, 1980],
        "YearRemodAdd": [2005, 1990],
        "GarageYrBlt": [2000, 1980],
        "YrSold": [2008, 2010],
    })
    result = creer_ages(df)
    assert result["building_age"].tolist() == [8, 30]
    assert result["remodel_age"].tolist() == [3, 20]
    # YrSold doit etre supprime apres usage (pour eviter le leakage final)
    assert "YrSold" not in result.columns


def test_creer_ages_sans_yrsold_utilise_annee_defaut():
    """En absence de YrSold : on utilise ANNEE_ESTIMATION_DEFAUT (2026)."""
    df = pd.DataFrame({
        "YearBuilt": [2003],
        "YearRemodAdd": [2003],
        "GarageYrBlt": [2003],
    })
    result = creer_ages(df)
    age_attendu = ANNEE_ESTIMATION_DEFAUT - 2003
    assert result["building_age"].iloc[0] == age_attendu
    assert result["remodel_age"].iloc[0] == age_attendu
