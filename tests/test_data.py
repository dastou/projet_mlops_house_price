"""Tests unitaires pour src/data.py."""
from __future__ import annotations

import pytest

from src.data import NOM_CIBLE, separer_features_cible


def test_charger_donnees_shape(donnees_brutes):
    """Le dataset doit contenir 1460 lignes et 81 colonnes (Id + 79 features + SalePrice)."""
    assert donnees_brutes.shape == (1460, 81)


def test_charger_donnees_contient_cible(donnees_brutes):
    """La colonne cible SalePrice doit etre presente."""
    assert NOM_CIBLE in donnees_brutes.columns


def test_separer_features_cible_split_correct(donnees_brutes):
    """La separation doit retourner (X sans cible, y = cible)."""
    X, y = separer_features_cible(donnees_brutes)
    assert NOM_CIBLE not in X.columns
    assert y.name == NOM_CIBLE
    assert len(X) == len(y) == 1460
