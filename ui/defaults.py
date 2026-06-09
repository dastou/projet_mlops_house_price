"""Valeurs par defaut et constantes pour l'interface Streamlit Laplace Immo.

Les valeurs medianes/modales proviennent du dataset Ames complet (1460 maisons).
Permettent de remplir automatiquement les champs techniques que l'agent
immobilier ne renseigne pas a la main.
"""

from __future__ import annotations

# 25 quartiers d'Ames, ordre alphabetique
NEIGHBORHOODS = [
    "Blmngtn",
    "Blueste",
    "BrDale",
    "BrkSide",
    "ClearCr",
    "CollgCr",
    "Crawfor",
    "Edwards",
    "Gilbert",
    "IDOTRR",
    "MeadowV",
    "Mitchel",
    "NAmes",
    "NPkVill",
    "NWAmes",
    "NoRidge",
    "NridgHt",
    "OldTown",
    "SWISU",
    "Sawyer",
    "SawyerW",
    "Somerst",
    "StoneBr",
    "Timber",
    "Veenker",
]

# Echelle ordinale de qualite (Po=Poor a Ex=Excellent)
ORDRE_QUALITES = ["Po", "Fa", "TA", "Gd", "Ex"]

QUALITES_LABELS = {
    "Po": "Mediocre",
    "Fa": "Acceptable",
    "TA": "Standard",
    "Gd": "Bon",
    "Ex": "Excellent",
}

# Defauts pour les 75 champs de l'API (medianes / modes du dataset Ames).
# Le formulaire surcharge les champs que l'agent renseigne explicitement.
DEFAULTS = {
    # Identification
    "MSSubClass": 20,
    "MSZoning": "RL",
    # Terrain
    "LotFrontage": 70.0,
    "LotArea": 9500,
    "Street": "Pave",
    "Alley": "None",
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    # Localisation
    "Neighborhood": "NAmes",
    "Condition1": "Norm",
    "Condition2": "Norm",
    # Type
    "BldgType": "1Fam",
    "HouseStyle": "1Story",
    # Qualite
    "OverallQual": 5,
    "OverallCond": 5,
    # Construction
    "YearBuilt": 1970,
    "YearRemodAdd": 1994,
    # Toit
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    # Exterieur
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "MasVnrArea": 0.0,
    "ExterQual": "TA",
    "ExterCond": "TA",
    # Fondation et sous-sol
    "Foundation": "PConc",
    "BsmtQual": "TA",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "Unf",
    "BsmtFinSF1": 0,
    "BsmtFinType2": "Unf",
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 477,
    "TotalBsmtSF": 991,
    # Chauffage et electricite
    "Heating": "GasA",
    "HeatingQC": "TA",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    # Surfaces interieures
    "1stFlrSF": 1087,
    "2ndFlrSF": 0,
    "LowQualFinSF": 0,
    "GrLivArea": 1464,
    # Salles de bain
    "BsmtFullBath": 0,
    "BsmtHalfBath": 0,
    "FullBath": 2,
    "HalfBath": 0,
    # Chambres et pieces
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "KitchenQual": "TA",
    "TotRmsAbvGrd": 6,
    "Functional": "Typ",
    # Cheminees
    "Fireplaces": 0,
    "FireplaceQu": "None",
    # Garage
    "GarageType": "Attchd",
    "GarageYrBlt": 1980.0,
    "GarageFinish": "Unf",
    "GarageCars": 2,
    "GarageArea": 480,
    "GarageQual": "TA",
    "GarageCond": "TA",
    # Voirie
    "PavedDrive": "Y",
    # Surfaces exterieures
    "WoodDeckSF": 0,
    "OpenPorchSF": 25,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    # Piscine et divers
    "PoolArea": 0,
    "PoolQC": "None",
    "Fence": "None",
    "MiscFeature": "None",
    "MiscVal": 0,
}
