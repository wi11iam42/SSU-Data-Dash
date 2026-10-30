"""EDIT ME: turn a table of homes into numbers a model can learn from.

Each function gets a table of homes with the columns of its test file (so never
sold_price or date_sold) and returns a table of numbers, one row per home.
FeatureModel in models.py uses them; switch it on in make_model() to try your features.

Ideas to try:
- lot size: lot_front * lot_depth (empty for condos, which is fine for tree models)
- garage, house_style, municipality
- how pricey a community is: the median price per square foot of its past sales
  (work it out from the training sales only, never from the homes you predict)
- how prices have moved recently in the area or for that property type
- words in the remarks, such as "renovated" or "needs work"
- off-market and list-aware can use different features: list-aware has list_price
"""
import numpy as np
import pandas as pd

PROPERTY_TYPES = [
    "Condo Apt", "Detached", "Semi-Detached",
    "Condo Townhouse", "Freehold Townhouse", "Co-Op Apt"
]


def square_feet(values):
    """Convert '900-1099' or '622' into numeric square feet."""
    parts = values.fillna("").astype(str).str.extract(
        r"^\s*([\d.]+)\s*(?:-\s*([\d.]+))?"
    )
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def base_features(homes):
    """Features available when the home is listed."""
    X = pd.DataFrame(index=homes.index)

    X["area"] = square_feet(homes["house_area"])
    X["bedrooms"] = pd.to_numeric(homes["bedrooms"], errors="coerce")
    X["washrooms"] = pd.to_numeric(homes["washrooms"], errors="coerce")
    X["parking"] = pd.to_numeric(homes["parking_total"], errors="coerce")

    X["latitude"] = pd.to_numeric(homes["latitude"], errors="coerce")
    X["longitude"] = pd.to_numeric(homes["longitude"], errors="coerce")

    X["lot_front"] = pd.to_numeric(homes["lot_front"], errors="coerce")
    X["lot_depth"] = pd.to_numeric(homes["lot_depth"], errors="coerce")
    X["lot_area"] = X["lot_front"] * X["lot_depth"]

    X["area_per_bedroom"] = X["area"] / X["bedrooms"].clip(lower=1)
    X["bathrooms_per_bedroom"] = (
        X["washrooms"] / X["bedrooms"].clip(lower=1)
    )

    # Date features
    dates = pd.to_datetime(homes["date_listed"])
    X["year"] = dates.dt.year
    X["month"] = dates.dt.month
    X["quarter"] = dates.dt.quarter
    X["days_from_2021"] = (
        dates - pd.Timestamp("2021-01-01")
    ).dt.days

    # Cyclical seasonality
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

    # Categorical variables for CatBoost
    categorical = [
        "house_type_name",
        "house_style",
        "garage",
        "municipality",
        "community",
    ]

    for col in categorical:
        if col in homes:
            X[col] = homes[col].fillna("Unknown").astype(str)
        else:
            X[col] = "Unknown"

    return X


class HistoricalFeatures:
    """
    Calculate comparable-sale statistics using only sales
    completed before each property's listing date.

    Uses community/type, community-wide and type-wide comparisons.
    """

    def fit(self, sales):
        self.groups = {}

        history = sales.copy()
        history["date_sold"] = pd.to_datetime(history["date_sold"])

        history["area"] = square_feet(history["house_area"])
        history["sold_price"] = pd.to_numeric(
            history["sold_price"], errors="coerce"
        )

        history["price_per_sqft"] = (
            history["sold_price"] / history["area"].replace(0, np.nan)
        )

        history = history.dropna(
            subset=["date_sold", "sold_price"]
        )

        history["community"] = (
            history["community"].fillna("Unknown").astype(str)
        )
        history["house_type_name"] = (
            history["house_type_name"].fillna("Unknown").astype(str)
        )

        for level in ["community_type", "community", "type"]:
            if level == "community_type":
                keys = ["community", "house_type_name"]
            elif level == "community":
                keys = ["community"]
            else:
                keys = ["house_type_name"]

            for key, group in history.groupby(keys, dropna=False):
                if not isinstance(key, tuple):
                    key = (key,)

                group = group.sort_values("date_sold")

                self.groups[(level, key)] = {
                    "dates": group["date_sold"].to_numpy(
                        dtype="datetime64[ns]"
                    ),
                    "prices": group["sold_price"].to_numpy(dtype=float),
                    "pps": group["price_per_sqft"].to_numpy(dtype=float),
                }

        return self

    def _stats(self, level, key, cutoff):
        data = self.groups.get((level, key))

        if data is None:
            return [np.nan, np.nan, 0, np.nan, np.nan]

        dates = data["dates"]
        prices = data["prices"]
        pps = data["pps"]

        # Strictly earlier than the listing date.
        end = np.searchsorted(
            dates, np.datetime64(cutoff), side="left"
        )

        if end == 0:
            return [np.nan, np.nan, 0, np.nan, np.nan]

        start_365 = np.searchsorted(
            dates,
            np.datetime64(cutoff - pd.Timedelta(days=365)),
            side="left",
        )

        start_90 = np.searchsorted(
            dates,
            np.datetime64(cutoff - pd.Timedelta(days=90)),
            side="left",
        )

        recent_prices = prices[start_365:end]
        recent_pps = pps[start_365:end]

        valid_prices = recent_prices[np.isfinite(recent_prices)]
        valid_pps = recent_pps[np.isfinite(recent_pps)]

        median_price = (
            np.median(valid_prices) if len(valid_prices) else np.nan
        )
        median_pps = (
            np.median(valid_pps) if len(valid_pps) else np.nan
        )

        # Recent 90-day trend relative to the previous 365-day median.
        last_prices = prices[start_90:end]
        last_prices = last_prices[np.isfinite(last_prices)]

        recent_median = (
            np.median(last_prices) if len(last_prices) >= 2 else np.nan
        )

        trend = (
            recent_median / median_price
            if np.isfinite(recent_median)
            and np.isfinite(median_price)
            and median_price > 0
            else np.nan
        )

        return [
            median_price,
            median_pps,
            len(valid_prices),
            trend,
            end,
        ]

    def transform(self, homes):
        result = pd.DataFrame(index=homes.index)

        communities = (
            homes["community"].fillna("Unknown").astype(str)
        )
        types = (
            homes["house_type_name"].fillna("Unknown").astype(str)
        )
        dates = pd.to_datetime(homes["date_listed"])

        levels = ["community_type", "community", "type"]

        for level in levels:
            result[f"{level}_median_price"] = np.nan
            result[f"{level}_median_ppsf"] = np.nan
            result[f"{level}_sales_count"] = 0
            result[f"{level}_recent_trend"] = np.nan

        for i, (idx, date) in enumerate(dates.items()):
            community = communities.loc[idx]
            kind = types.loc[idx]

            if pd.isna(date):
                continue

            keys = {
                "community_type": (community, kind),
                "community": (community,),
                "type": (kind,),
            }

            for level, key in keys.items():
                stats = self._stats(level, key, date)

                result.at[idx, f"{level}_median_price"] = stats[0]
                result.at[idx, f"{level}_median_ppsf"] = stats[1]
                result.at[idx, f"{level}_sales_count"] = stats[2]
                result.at[idx, f"{level}_recent_trend"] = stats[3]

        # Relative neighbourhood pricing.
        result["community_vs_type"] = (
            result["community_median_price"]
            / result["type_median_price"].replace(0, np.nan)
        )

        result["community_type_vs_type"] = (
            result["community_type_median_price"]
            / result["type_median_price"].replace(0, np.nan)
        )

        # Size-adjusted comparable estimates.
        area = square_feet(homes["house_area"])
        result["community_estimated_value"] = (
            result["community_type_median_ppsf"] * area
        )
        result["type_estimated_value"] = (
            result["type_median_ppsf"] * area
        )

        return result


def off_market_features(homes, historical):
    """Complete off-market feature matrix."""
    X = base_features(homes)

    comparable = historical.transform(homes)

    X = pd.concat([X, comparable], axis=1)

    return X


def list_aware_features(homes, historical=None):
    X = base_features(homes)

    if historical is not None:
        X = pd.concat([X, historical.transform(homes)], axis=1)

    X["list_price"] = pd.to_numeric(
        homes["list_price"], errors="coerce"
    )

    return X
