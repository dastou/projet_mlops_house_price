"""Fixtures partagees pour les tests pytest.

Les fixtures avec scope='session' sont chargees une seule fois pour toute
la session de tests, ce qui evite de recharger les donnees a chaque test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ajout de la racine du projet a sys.path pour permettre les imports src.*
chemin_projet = Path(__file__).resolve().parent.parent
if str(chemin_projet) not in sys.path:
    sys.path.insert(0, str(chemin_projet))

from src.data import charger_donnees
from src.features import creer_features
from src.preprocessing import nettoyer_donnees


@pytest.fixture(scope="session")
def donnees_brutes() -> pd.DataFrame:
    """Donnees Ames brutes depuis OpenML (charge une fois pour la session)."""
    return charger_donnees()


@pytest.fixture(scope="session")
def donnees_propres(donnees_brutes) -> pd.DataFrame:
    """Donnees apres nettoyage deterministe (preprocessing.py)."""
    return nettoyer_donnees(donnees_brutes)


@pytest.fixture(scope="session")
def donnees_features(donnees_propres) -> pd.DataFrame:
    """Donnees apres preprocessing complet et feature engineering."""
    return creer_features(donnees_propres)
