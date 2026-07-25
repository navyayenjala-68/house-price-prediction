"""Reusable feature engineering for the Ames House Prices training data.

Run this transformation before selecting features and training a new model. It
does not mutate the source DataFrame supplied by the caller.
"""

import pandas as pd


ENGINEERED_FEATURES = (
    "TotalSF",
    "TotalBathrooms",
    "HouseAge",
    "YearsSinceRemodel",
    "TotalOutdoorSF",
)


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable size, bathroom, age, renovation, and outdoor-space features."""
    df = data.copy()

    df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)
    df["TotalBathrooms"] = (
        df["FullBath"].fillna(0)
        + .5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + .5 * df["BsmtHalfBath"].fillna(0)
    )
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["YearsSinceRemodel"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    df["TotalOutdoorSF"] = (
        df["WoodDeckSF"].fillna(0)
        + df["OpenPorchSF"].fillna(0)
        + df["EnclosedPorch"].fillna(0)
        + df["3SsnPorch"].fillna(0)
        + df["ScreenPorch"].fillna(0)
    )
    return df
