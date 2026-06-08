"""Modeles Pydantic pour l'API Laplace Immo.

Ce module definit les structures d'entree et de sortie de l'API :
- MaisonInput : les 75 caracteristiques brutes attendues pour predire un prix
- PrixPrediction : la reponse retournee par /predict
- HealthResponse : la reponse retournee par /health

Les NaN structurels (PoolQC, Alley, Fence, etc.) sont representes par la
chaine "None" pour rester coherent avec le preprocessing du pipeline.
LotFrontage peut etre None (NaN) : la mediane par quartier sera appliquee
par le custom transformer LotFrontageImputer.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class MaisonInput(BaseModel):
    """Caracteristiques brutes d'une maison.

    L'utilisateur fournit les valeurs telles qu'observees sur la maison
    (avant feature engineering). Les variables 1stFlrSF, 2ndFlrSF et
    3SsnPorch sont exposees avec ces noms exacts grace aux alias Pydantic.
    Les exemples ci-dessous correspondent a la maison Id=1 du dataset Ames.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # ----- Identification et type -----
    MSSubClass: int = Field(..., description="Type de batiment (code numerique)", examples=[60])
    MSZoning: str = Field(..., description="Zonage general", examples=["RL"])

    # ----- Terrain -----
    LotFrontage: Optional[float] = Field(
        default=None,
        description="Facade en pieds (peut etre null, imputation par quartier)",
        examples=[65.0],
    )
    LotArea: int = Field(..., description="Surface du terrain (sq ft)", examples=[8450])
    Street: str = Field(..., examples=["Pave"])
    Alley: str = Field(..., description="None si pas d'allee", examples=["None"])
    LotShape: str = Field(..., examples=["Reg"])
    LandContour: str = Field(..., examples=["Lvl"])
    Utilities: str = Field(..., examples=["AllPub"])
    LotConfig: str = Field(..., examples=["Inside"])
    LandSlope: str = Field(..., examples=["Gtl"])

    # ----- Localisation -----
    Neighborhood: str = Field(..., description="Quartier d'Ames", examples=["CollgCr"])
    Condition1: str = Field(..., examples=["Norm"])
    Condition2: str = Field(..., examples=["Norm"])

    # ----- Type de logement -----
    BldgType: str = Field(..., examples=["1Fam"])
    HouseStyle: str = Field(..., examples=["2Story"])

    # ----- Qualite generale -----
    OverallQual: int = Field(..., ge=1, le=10, description="Qualite generale (1-10)", examples=[7])
    OverallCond: int = Field(..., ge=1, le=10, description="Etat general (1-10)", examples=[5])

    # ----- Construction et renovation -----
    YearBuilt: int = Field(..., ge=1800, le=2030, description="Annee de construction", examples=[2003])
    YearRemodAdd: int = Field(..., ge=1800, le=2030, description="Annee de derniere renovation", examples=[2003])

    # ----- Toit -----
    RoofStyle: str = Field(..., examples=["Gable"])
    RoofMatl: str = Field(..., examples=["CompShg"])

    # ----- Exterieur -----
    Exterior1st: str = Field(..., examples=["VinylSd"])
    Exterior2nd: str = Field(..., examples=["VinylSd"])
    MasVnrType: str = Field(..., description="None si pas de parement maconne", examples=["BrkFace"])
    MasVnrArea: float = Field(..., ge=0, description="Surface du parement (sq ft)", examples=[196.0])
    ExterQual: str = Field(..., description="Qualite materiau exterieur", examples=["Gd"])
    ExterCond: str = Field(..., examples=["TA"])

    # ----- Fondation et sous-sol -----
    Foundation: str = Field(..., examples=["PConc"])
    BsmtQual: str = Field(..., description="None si pas de sous-sol", examples=["Gd"])
    BsmtCond: str = Field(..., description="None si pas de sous-sol", examples=["TA"])
    BsmtExposure: str = Field(..., description="None si pas de sous-sol", examples=["No"])
    BsmtFinType1: str = Field(..., description="None si pas de sous-sol", examples=["GLQ"])
    BsmtFinSF1: int = Field(..., ge=0, examples=[706])
    BsmtFinType2: str = Field(..., description="None si pas de sous-sol", examples=["Unf"])
    BsmtFinSF2: int = Field(..., ge=0, examples=[0])
    BsmtUnfSF: int = Field(..., ge=0, examples=[150])
    TotalBsmtSF: int = Field(..., ge=0, description="Surface totale du sous-sol (sq ft)", examples=[856])

    # ----- Chauffage et electricite -----
    Heating: str = Field(..., examples=["GasA"])
    HeatingQC: str = Field(..., examples=["Ex"])
    CentralAir: str = Field(..., examples=["Y"])
    Electrical: str = Field(..., examples=["SBrkr"])

    # ----- Surfaces interieures -----
    # Aliases pour les noms commencant par un chiffre (impossible en Python)
    firstFlrSF: int = Field(..., alias="1stFlrSF", ge=0, description="Surface rdc (sq ft)", examples=[856])
    secondFlrSF: int = Field(..., alias="2ndFlrSF", ge=0, description="Surface etage (sq ft)", examples=[854])
    LowQualFinSF: int = Field(..., ge=0, examples=[0])
    GrLivArea: int = Field(..., ge=0, description="Surface habitable totale (sq ft)", examples=[1710])

    # ----- Salles de bain -----
    BsmtFullBath: int = Field(..., ge=0, examples=[1])
    BsmtHalfBath: int = Field(..., ge=0, examples=[0])
    FullBath: int = Field(..., ge=0, description="SdB completes au-dessus du sol", examples=[2])
    HalfBath: int = Field(..., ge=0, description="Demi SdB au-dessus du sol", examples=[1])

    # ----- Chambres et pieces -----
    BedroomAbvGr: int = Field(..., ge=0, description="Nombre de chambres", examples=[3])
    KitchenAbvGr: int = Field(..., ge=0, description="Nombre de cuisines", examples=[1])
    KitchenQual: str = Field(..., description="Qualite cuisine (Po/Fa/TA/Gd/Ex)", examples=["Gd"])
    TotRmsAbvGrd: int = Field(..., ge=0, description="Nombre total de pieces hors SdB", examples=[8])
    Functional: str = Field(..., examples=["Typ"])

    # ----- Cheminees -----
    Fireplaces: int = Field(..., ge=0, examples=[0])
    FireplaceQu: str = Field(..., description="None si pas de cheminee", examples=["None"])

    # ----- Garage -----
    GarageType: str = Field(..., description="None si pas de garage", examples=["Attchd"])
    GarageYrBlt: float = Field(..., ge=0, description="0 si pas de garage", examples=[2003.0])
    GarageFinish: str = Field(..., description="None si pas de garage", examples=["RFn"])
    GarageCars: int = Field(..., ge=0, examples=[2])
    GarageArea: int = Field(..., ge=0, examples=[548])
    GarageQual: str = Field(..., description="None si pas de garage", examples=["TA"])
    GarageCond: str = Field(..., description="None si pas de garage", examples=["TA"])

    # ----- Voirie -----
    PavedDrive: str = Field(..., examples=["Y"])

    # ----- Surfaces exterieures (porches et terrasses) -----
    WoodDeckSF: int = Field(..., ge=0, examples=[0])
    OpenPorchSF: int = Field(..., ge=0, examples=[61])
    EnclosedPorch: int = Field(..., ge=0, examples=[0])
    threeSsnPorch: int = Field(..., alias="3SsnPorch", ge=0, examples=[0])
    ScreenPorch: int = Field(..., ge=0, examples=[0])

    # ----- Piscine et divers -----
    PoolArea: int = Field(..., ge=0, examples=[0])
    PoolQC: str = Field(..., description="None si pas de piscine", examples=["None"])
    Fence: str = Field(..., description="None si pas de cloture", examples=["None"])
    MiscFeature: str = Field(..., description="None si rien", examples=["None"])
    MiscVal: int = Field(..., ge=0, examples=[0])


class PrixPrediction(BaseModel):
    """Reponse du endpoint /predict.

    Contient le prix estime en dollars (echelle metier, lisible) et la
    prediction brute du modele en log1p (utile pour le debug).
    """

    prix_estime_dollars: float = Field(
        ...,
        description="Prix estime en dollars (apres expm1)",
        examples=[208500.0],
    )
    prix_log: float = Field(
        ...,
        description="Prediction brute en log1p(SalePrice)",
        examples=[12.247],
    )


class HealthResponse(BaseModel):
    """Reponse du endpoint /health.

    Permet de verifier que l'API tourne et que le modele est correctement
    charge en memoire.
    """

    status: str = Field(..., examples=["ok"])
    model_loaded: bool = Field(..., examples=[True])
