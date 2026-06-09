"""Genere un echantillon stratifie du dataset Ames pour les tests Pytest.

Ce script est execute UNE SEULE FOIS en local pour creer le fichier
tests/fixtures/house_prices_sample.csv (~100 lignes, ~50 KB).

Le CSV genere est ensuite committe dans Git et utilise par les fixtures
pytest (voir tests/conftest.py) afin que les tests ne dependent plus
d'un appel reseau a OpenML.

La stratification se fait sur la colonne Neighborhood : on garde quelques
maisons de chaque quartier pour que les tests couvrent toutes les zones
(important pour LotFrontageImputer qui calcule des medianes par quartier).

Usage (depuis la racine du projet) :
    python -m tests.fixtures._generer_sample
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Ajout de la racine du projet a sys.path pour permettre l'import de src.data.
chemin_projet = Path(__file__).resolve().parent.parent.parent
if str(chemin_projet) not in sys.path:
    sys.path.insert(0, str(chemin_projet))

from src.data import charger_donnees  # noqa: E402

logger = logging.getLogger(__name__)

# 4 maisons par quartier x 25 quartiers = environ 100 lignes au total.
MAISONS_PAR_QUARTIER = 4
GRAINE_ALEATOIRE = 42
CHEMIN_SORTIE = Path(__file__).parent / "house_prices_sample.csv"


def generer_echantillon() -> pd.DataFrame:
    """Construit un echantillon stratifie sur Neighborhood.

    Strategie : pour chaque quartier, on tire aleatoirement
    MAISONS_PAR_QUARTIER lignes (ou moins si le quartier en contient moins).
    La graine aleatoire est fixee a 42 pour que l'echantillon soit
    deterministe d'une execution a l'autre.

    Returns:
        DataFrame de ~100 lignes avec les memes colonnes que le dataset complet.
    """
    df = charger_donnees()
    logger.info("Dataset complet charge : %d lignes x %d colonnes", df.shape[0], df.shape[1])

    echantillon = (
        df.groupby("Neighborhood", group_keys=False)
        .apply(
            lambda groupe: groupe.sample(
                n=min(MAISONS_PAR_QUARTIER, len(groupe)),
                random_state=GRAINE_ALEATOIRE,
            )
        )
        .sort_values("Id")
        .reset_index(drop=True)
    )

    logger.info(
        "Echantillon genere : %d lignes, %d quartiers representes",
        len(echantillon),
        echantillon["Neighborhood"].nunique(),
    )
    return echantillon


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    echantillon = generer_echantillon()
    echantillon.to_csv(CHEMIN_SORTIE, index=False)
    taille_ko = CHEMIN_SORTIE.stat().st_size / 1024
    logger.info("Echantillon sauvegarde : %s (%.1f KB)", CHEMIN_SORTIE, taille_ko)
