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

from src.features import creer_features
from src.preprocessing import nettoyer_donnees

# Echantillon stratifie (98 lignes, 25 quartiers) genere une fois via
# tests/fixtures/_generer_sample.py et committe dans Git.
# Permet aux tests de tourner sans dependance reseau (OpenML).
CHEMIN_ECHANTILLON = Path(__file__).parent / "fixtures" / "house_prices_sample.csv"


@pytest.fixture(scope="session")
def donnees_brutes() -> pd.DataFrame:
    """Donnees Ames sur echantillon local (charge une fois pour la session).

    Lit tests/fixtures/house_prices_sample.csv au lieu d'appeler fetch_openml.
    Tests rapides, deterministes et independants d'Internet.
    """
    return pd.read_csv(CHEMIN_ECHANTILLON)


@pytest.fixture(scope="session")
def donnees_propres(donnees_brutes) -> pd.DataFrame:
    """Donnees apres nettoyage deterministe (preprocessing.py)."""
    return nettoyer_donnees(donnees_brutes)


@pytest.fixture(scope="session")
def donnees_features(donnees_propres) -> pd.DataFrame:
    """Donnees apres preprocessing complet et feature engineering."""
    return creer_features(donnees_propres)
