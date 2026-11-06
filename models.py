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
            iterations=1500,
            depth=10,
            learning_rate=0.03,
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

        self.spatial = features.SpatialComparableFeatures()
        self.spatial.fit(train)

        X = features.off_market_features(
            train, self.historical
        )

        X = pd.concat(
            [X, self.spatial.transform(train)],
            axis=1,
            copy=False
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

        X = pd.concat(
            [X, self.spatial.transform(homes)],
            axis=1
        )

        for col in self.cat_features:
            X[col] = X[col].fillna("Unknown").astype(str)

        predictions = np.exp(self.model.predict(X))

        return np.maximum(predictions, 1)


class NearestSalesBaseline:
    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = [
            "house_type_name",
            "date_sold",
            "latitude",
            "longitude",
            "sold_price",
        ]

        self.sales = train[columns].copy()
        self.sales = self.sales.sort_values("date_sold")

        self.sales_by_type = dict(
            list(self.sales.groupby("house_type_name"))
        )

        return self

    def predict(self, homes):
        predictions = []

        for _, home in homes.iterrows():
            sales = self.sales_by_type.get(
                home["house_type_name"],
                self.sales
            )

            earlier = sales[
                sales["date_sold"] < home["date_listed"]
                ]

            if len(earlier) == 0:
                earlier = self.sales[
                    self.sales["date_sold"] < home["date_listed"]
                    ]

            if len(earlier) == 0:
                predictions.append(np.nan)
                continue

            dx = (
                         earlier["longitude"].to_numpy(dtype=float)
                         - float(home["longitude"])
                 ) * 0.72

            dy = (
                    earlier["latitude"].to_numpy(dtype=float)
                    - float(home["latitude"])
            )

            distance = np.sqrt(dx ** 2 + dy ** 2)

            nearest = np.argsort(
                distance, kind="stable"
            )[:self.k]

            predictions.append(
                np.median(
                    earlier["sold_price"].to_numpy(dtype=float)[nearest]
                )
            )

        return np.asarray(predictions, dtype=float)


def make_model(track):
    """Return a fresh model for the requested competition track."""

    if track == "off_market":
            return CatBoostOffMarket()

    if track == "list_aware":
            return CatBoostListAware()

    raise ValueError(f"Unknown track: {track}")


class ListPriceBaseline:
    def fit(self, train):
        return self

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)



class CatBoostListAware:

    def __init__(self):
        self.historical = features.HistoricalFeatures()
        self.spatial = features.SpatialComparableFeatures()

        self.model = CatBoostRegressor(
            iterations=1000,
            depth=7,
            learning_rate=0.05,
            loss_function="MAE",
            eval_metric="MAE",
            l2_leaf_reg=7,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

    def build_features(self, homes):

        X = pd.concat(
            [
                features.base_features(homes),
                self.historical.transform(homes),
                self.spatial.transform(homes),
            ],
            axis=1
        )

        # 3. List price features
        list_price = pd.to_numeric(
            homes["list_price"], errors="coerce"
        )

        area = features.square_feet(homes["house_area"])

        X["list_price"] = list_price
        X["log_list_price"] = np.log(list_price.clip(lower=1))

        X["list_price_per_sqft"] = (
            list_price / area.replace(0, np.nan)
        )

        # 4. Compare asking price with historical estimates

        comparison_features = [
            "geo_median_price_5",
            "geo_median_price_10",
            "geo_median_price_20",
            "community_type_median_price",
            "community_median_price",
            "type_median_price",
            "community_estimated_value",
            "type_estimated_value",
        ]

        for col in comparison_features:
            if col in X.columns:
                X[f"list_vs_{col}"] = (
                    list_price / X[col].replace(0, np.nan)
                )

                X[f"log_list_vs_{col}"] = np.log(
                    X[f"list_vs_{col}"].clip(lower=0.01)
                )

        # 5. Compare price per square foot
        for col in [
            "geo_median_ppsf_10",
            "geo_median_ppsf_20",
            "community_type_median_ppsf",
            "community_median_ppsf",
            "type_median_ppsf",
        ]:
            if col in X.columns:
                X[f"list_ppsf_vs_{col}"] = (
                    X["list_price_per_sqft"]
                    / X[col].replace(0, np.nan)
                )

        # 6. Additional property interactions
        X["area_x_bedrooms"] = (
            X["area"] * X["bedrooms"]
        )

        X["bathroom_count"] = X["washrooms"]

        X["lot_to_house_ratio"] = (
            X["lot_area"] / X["area"].replace(0, np.nan)
        )

        X["geo_estimated_value_10"] = (
                X["geo_median_ppsf_10"] * X["area"]
        )

        X["geo_estimated_value_20"] = (
                X["geo_median_ppsf_20"] * X["area"]
        )

        X["geo_value_vs_list"] = (
                X["geo_estimated_value_10"]
                / X["list_price"].replace(0, np.nan)
        )

        remarks = homes["remarks"].fillna("").astype(str).str.lower()

        keywords = {
            "renovated": ["renovated", "remodelled", "remodeled"],
            "luxury": ["luxury", "luxurious", "high-end"],
            "new_kitchen": ["new kitchen", "updated kitchen"],
            "basement": ["finished basement", "basement"],
            "needs_work": ["as-is", "handyman", "needs work", "fixer"],
            "new_build": ["new build", "newly built", "never lived"],
            "upgraded": ["upgraded", "upgrades", "premium finishes"],
            "outdoor": ["pool", "garden", "terrace", "landscaped"],
            "view": ["waterfront", "lake view", "panoramic view"],
            "parking": ["private driveway", "double garage"],
        }

        for feature, terms in keywords.items():
            pattern = "|".join(terms)
            X[f"remark_{feature}"] = (
                remarks.str.contains(pattern, regex=True, na=False)
                .astype(int)
            )

        X["remarks_length"] = remarks.str.len()

        X["remarks_word_count"] = (
            remarks.str.split().str.len()
        )

        # Is the home priced unusually high relative to its size
        X["list_price_per_sqft_log"] = np.log(
            X["list_price_per_sqft"].clip(lower=1)
        )

        # Is the home larger than typical comparable properties
        X["area_vs_geo_median"] = (
                X["area"] / X["geo_median_area_10"].replace(0, np.nan)
        )

        # Does the asking price reflect neighbourhood pricing
        X["list_vs_community_ppsf"] = (
                X["list_price_per_sqft"]
                / X["community_type_median_ppsf"].replace(0, np.nan)
        )

        return X

    def fit(self, train):

        train = train.copy()

        train["list_price"] = pd.to_numeric(
            train["list_price"], errors="coerce"
        )

        train["sold_price"] = pd.to_numeric(
            train["sold_price"], errors="coerce"
        )

        valid = (
            train["list_price"].notna()
            & train["sold_price"].notna()
            & (train["list_price"] > 0)
            & (train["sold_price"] > 0)
        )

        train = train.loc[valid].copy()

        # Fit historical feature generators on training sales only.
        self.historical.fit(train)
        self.spatial.fit(train)

        X = self.build_features(train)

        # Predict the log sale/list ratio.
        y = np.log(
            train["sold_price"] / train["list_price"]
        )

        self.cat_features = X.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        for col in self.cat_features:
            X[col] = X[col].fillna("Unknown").astype(str)

        self.model.fit(
            X,
            y,
            cat_features=self.cat_features,
        )

        return self

    def predict(self, homes):

        X = self.build_features(homes)

        for col in self.cat_features:
            X[col] = X[col].fillna("Unknown").astype(str)

        predicted_ratio = np.exp(
            self.model.predict(X)
        )

        list_price = pd.to_numeric(
            homes["list_price"], errors="coerce"
        ).to_numpy(dtype=float)

        predictions = list_price * predicted_ratio

        return np.maximum(predictions, 1)
