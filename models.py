"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

import features


class CatBoostOffMarket:
    def __init__(self):
        self.historical = features.HistoricalFeatures()

        self.model = CatBoostRegressor(
            iterations=1200,
            depth=8,
            learning_rate=0.04,
            loss_function="RMSE",
            l2_leaf_reg=5,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

    def fit(self, train):
        train = train.copy()

        # Build historical statistics from the training sales.
        self.historical.fit(train)

        # Each training home's features use only sales before
        # that home's listing date.
        X = features.off_market_features(
            train, self.historical
        )

        y = np.log(
            pd.to_numeric(train["sold_price"], errors="coerce")
        )

        valid = (
            y.notna()
            & np.isfinite(y)
            & (train["sold_price"] > 0)
        )

        X = X.loc[valid].copy()
        y = y.loc[valid]

        self.cat_features = X.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        # CatBoost accepts missing numeric values.
        # Categorical missing values have already become strings.
        for col in self.cat_features:
            X[col] = X[col].fillna("Unknown").astype(str)

        self.model.fit(
            X,
            y,
            cat_features=self.cat_features,
        )

        return self

    def predict(self, homes):
        X = features.off_market_features(
            homes, self.historical
        )

        for col in self.cat_features:
            X[col] = X[col].fillna("Unknown").astype(str)

        predictions = np.exp(self.model.predict(X))

        return np.maximum(predictions, 1)


def make_model(track):
    """Return a fresh model for the requested competition track."""

    if track == "off_market":
        return CatBoostOffMarket()

    # Keep the original list-price baseline for now.
    return ListPriceBaseline()


class ListPriceBaseline:
    def fit(self, train):
        return self

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)
