"""Tests d'integration pour l'API FastAPI (api/main.py)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Client de test FastAPI.

    Le 'with' declenche le lifespan event qui charge final_model.pkl en memoire,
    une seule fois pour tout le module de tests.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture
def maison_id1_payload():
    """Payload JSON valide correspondant a la maison Id=1 du dataset Ames.

    Cette maison s'est reellement vendue 208 500 dollars. Notre modele
    devrait predire entre 180 000 et 240 000 dollars (erreur < 15%).
    """
    return {
        "MSSubClass": 60, "MSZoning": "RL", "LotFrontage": 65,
        "LotArea": 8450, "Street": "Pave", "Alley": "None", "LotShape": "Reg",
        "LandContour": "Lvl", "Utilities": "AllPub", "LotConfig": "Inside",
        "LandSlope": "Gtl", "Neighborhood": "CollgCr", "Condition1": "Norm",
        "Condition2": "Norm", "BldgType": "1Fam", "HouseStyle": "2Story",
        "OverallQual": 7, "OverallCond": 5, "YearBuilt": 2003,
        "YearRemodAdd": 2003, "RoofStyle": "Gable", "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd",
        "MasVnrType": "BrkFace", "MasVnrArea": 196.0,
        "ExterQual": "Gd", "ExterCond": "TA", "Foundation": "PConc",
        "BsmtQual": "Gd", "BsmtCond": "TA", "BsmtExposure": "No",
        "BsmtFinType1": "GLQ", "BsmtFinSF1": 706, "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0, "BsmtUnfSF": 150, "TotalBsmtSF": 856,
        "Heating": "GasA", "HeatingQC": "Ex", "CentralAir": "Y",
        "Electrical": "SBrkr", "1stFlrSF": 856, "2ndFlrSF": 854,
        "LowQualFinSF": 0, "GrLivArea": 1710,
        "BsmtFullBath": 1, "BsmtHalfBath": 0, "FullBath": 2, "HalfBath": 1,
        "BedroomAbvGr": 3, "KitchenAbvGr": 1, "KitchenQual": "Gd",
        "TotRmsAbvGrd": 8, "Functional": "Typ", "Fireplaces": 0,
        "FireplaceQu": "None", "GarageType": "Attchd", "GarageYrBlt": 2003.0,
        "GarageFinish": "RFn", "GarageCars": 2, "GarageArea": 548,
        "GarageQual": "TA", "GarageCond": "TA", "PavedDrive": "Y",
        "WoodDeckSF": 0, "OpenPorchSF": 61, "EnclosedPorch": 0,
        "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0,
        "PoolQC": "None", "Fence": "None", "MiscFeature": "None", "MiscVal": 0,
    }


def test_root_endpoint_retourne_structure_attendue(client):
    """GET / doit retourner 200 et lister les endpoints disponibles."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_health_endpoint_confirme_modele_charge(client):
    """GET /health doit retourner 200 avec model_loaded=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid_input_retourne_prix_realiste(client, maison_id1_payload):
    """POST /predict avec maison valide retourne un prix dans une plage realiste."""
    response = client.post("/predict", json=maison_id1_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prix_estime_dollars" in data
    assert "prix_log" in data
    # Plage realiste pour le dataset Ames (10k - 1M dollars)
    assert 10_000 < data["prix_estime_dollars"] < 1_000_000


def test_predict_input_invalide_retourne_422(client):
    """POST /predict avec un payload incomplet retourne 422 (Pydantic validation)."""
    payload_incomplet = {"MSSubClass": 60, "MSZoning": "RL"}
    response = client.post("/predict", json=payload_incomplet)
    assert response.status_code == 422
