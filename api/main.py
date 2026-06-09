"""API REST FastAPI pour l'estimation de prix de maisons Laplace Immo.

L'API expose trois endpoints :
- GET  /         : page d'accueil avec les routes disponibles
- GET  /health   : verification du statut + chargement du modele
- POST /predict  : prediction du prix a partir des caracteristiques d'une maison

Le modele final (Pipeline preprocessing + CatBoost) est charge une fois au
demarrage via le lifespan event, puis sert toutes les requetes.

Lancer en local :

    uvicorn api.main:app --reload

Puis ouvrir http://localhost:8000/docs (Swagger UI).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, MaisonInput, PrixPrediction
from src.features import creer_features

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# Conteneur d'etat de l'application (modele charge au startup)
class EtatApp:
    """Singleton pour partager le modele entre requetes."""

    model = None


etat = EtatApp()
RACINE_PROJET = Path(__file__).resolve().parent.parent
CHEMIN_MODELE = RACINE_PROJET / "models" / "final_model.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modele au demarrage de l'API.

    L'event lifespan remplace les anciens @app.on_event("startup"). Il
    permet egalement un cleanup propre a l'arret (yield separe les deux
    phases).
    """
    logger.info("Chargement du modele depuis %s", CHEMIN_MODELE)
    if not CHEMIN_MODELE.exists():
        raise RuntimeError(
            f"Modele introuvable a {CHEMIN_MODELE}. "
            "Lancer 'dvc repro' pour generer final_model.pkl."
        )
    etat.model = joblib.load(CHEMIN_MODELE)
    logger.info("Modele charge avec succes")
    yield
    # Cleanup eventuel (rien a faire ici)
    logger.info("Arret de l'API")


app = FastAPI(
    title="Laplace Immo API",
    description=(
        "API d'estimation de prix de maisons pour le reseau Laplace Immo. "
        "Le modele utilise est un Pipeline scikit-learn complet "
        "(preprocessing + CatBoostRegressor) entraine sur le dataset Ames Iowa."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def racine():
    """Page d'accueil de l'API : liste les endpoints disponibles."""
    return {
        "message": "Bienvenue sur l'API Laplace Immo",
        "endpoints": {
            "documentation_swagger": "/docs",
            "documentation_redoc": "/redoc",
            "verification_etat": "/health",
            "prediction_prix": "POST /predict",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Verifie que l'API tourne et que le modele est charge."""
    return HealthResponse(
        status="ok",
        model_loaded=etat.model is not None,
    )


@app.post("/predict", response_model=PrixPrediction)
def predict(maison: MaisonInput):
    """Predit le prix d'une maison a partir de ses caracteristiques.

    Le pipeline complet est applique :
    1. Conversion de l'input Pydantic en DataFrame pandas
    2. Feature engineering (TotalSF, ages, has_X, etc.)
    3. Prediction en log1p via le Pipeline (preprocessing + CatBoost)
    4. Conversion en dollars via expm1
    """
    if etat.model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    # 1. Pydantic -> dict (avec aliases pour 1stFlrSF, 2ndFlrSF, 3SsnPorch) -> DataFrame
    donnees = maison.model_dump(by_alias=True)
    df = pd.DataFrame([donnees])

    # 2. Feature engineering : creation des 11 features derivees.
    # creer_features gere l'absence de YrSold via ANNEE_ESTIMATION_DEFAUT (2026).
    df = creer_features(df)

    # 3. Prediction (echelle log1p)
    try:
        pred_log = float(etat.model.predict(df)[0])
    except Exception as exc:
        logger.exception("Erreur lors de la prediction")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prediction : {exc}",
        ) from exc

    # 4. Retour a l'echelle dollars pour la lisibilite metier
    prix_dollars = float(np.expm1(pred_log))

    return PrixPrediction(
        prix_estime_dollars=round(prix_dollars, 2),
        prix_log=round(pred_log, 4),
    )
