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
from scipy.spatial import cKDTree


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
        """Fast historical features using NumPy arrays."""

        n = len(homes)

        # Extract columns once instead of repeatedly using .loc.
        communities = (
            homes["community"].fillna("Unknown").astype(str).to_numpy()
        )
        types = (
            homes["house_type_name"].fillna("Unknown").astype(str).to_numpy()
        )
        dates = pd.to_datetime(
            homes["date_listed"]
        ).to_numpy(dtype="datetime64[ns]")

        levels = {
            "community_type": [
                (communities[i], types[i]) for i in range(n)
            ],
            "community": [
                (communities[i],) for i in range(n)
            ],
            "type": [
                (types[i],) for i in range(n)
            ],
        }

        # Preallocate NumPy arrays.
        output = {}

        for level in levels:
            output[level] = {
                "median_price": np.full(n, np.nan),
                "median_ppsf": np.full(n, np.nan),
                "sales_count": np.zeros(n),
                "recent_trend": np.full(n, np.nan),
            }

        # Process each home.
        for i in range(n):

            if np.isnat(dates[i]):
                continue

            cutoff = pd.Timestamp(dates[i])

            for level, keys in levels.items():

                stats = self._stats(
                    level,
                    keys[i],
                    cutoff
                )

                output[level]["median_price"][i] = stats[0]
                output[level]["median_ppsf"][i] = stats[1]
                output[level]["sales_count"][i] = stats[2]
                output[level]["recent_trend"][i] = stats[3]

        # Build the DataFrame once.
        result = pd.DataFrame(index=homes.index)

        for level in levels:
            for feature in output[level]:
                result[f"{level}_{feature}"] = output[level][feature]

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
        area = square_feet(homes["house_area"]).to_numpy()

        result["community_estimated_value"] = (
            result["community_type_median_ppsf"].to_numpy() * area
        )

        result["type_estimated_value"] = (
            result["type_median_ppsf"].to_numpy() * area
        )

        return result


def off_market_features(homes, historical):
    """Complete off-market feature matrix."""
    X = base_features(homes)

    comparable = historical.transform(homes)

    X = pd.concat([X, comparable], axis=1)

    return X



def list_aware_features(homes, historical=None):
    """Fast list-aware feature generation."""

    X = base_features(homes)

    # Add historical features.
    if historical is not None:
        historical_features = historical.transform(homes)
        X = pd.concat(
            [X, historical_features],
            axis=1,
            copy=False
        )

    # Convert list price once.
    price = pd.to_numeric(
        homes["list_price"],
        errors="coerce"
    ).to_numpy(dtype=float)

    area = X["area"].to_numpy(dtype=float)
    bedrooms = X["bedrooms"].to_numpy(dtype=float)

    X["list_price"] = price

    # Additional list-aware features.
    X["log_list_price"] = np.log1p(
        np.maximum(price, 0)
    )

    X["list_price_per_sqft"] = (
        price / np.where(area > 0, area, np.nan)
    )

    X["list_price_per_bedroom"] = (
        price / np.where(bedrooms > 0, bedrooms, np.nan)
    )

    return X


class SpatialComparableFeatures:
    """Fast batch-based geographically comparable sales features."""

    def fit(self, sales):
        self.data = {}

        df = sales.copy()
        df["area"] = square_feet(df["house_area"])
        df["date_sold"] = pd.to_datetime(df["date_sold"])

        for kind, group in df.groupby("house_type_name"):
            group = group.dropna(
                subset=[
                    "latitude", "longitude",
                    "date_sold", "sold_price"
                ]
            ).copy()

            coords = np.column_stack([
                group["latitude"].to_numpy(dtype=float),
                group["longitude"].to_numpy(dtype=float) * 0.72,
            ])

            self.data[kind] = {
                "tree": cKDTree(coords),
                "dates": group["date_sold"].to_numpy(
                    dtype="datetime64[ns]"
                ),
                "prices": group["sold_price"].to_numpy(dtype=float),
                "areas": group["area"].to_numpy(dtype=float),
            }

        return self

    def transform(self, homes):

        columns = [
            "geo_median_price_5",
            "geo_median_price_10",
            "geo_median_ppsf_10",
            "geo_nearest_distance",
            "geo_median_area_10",
            "geo_comparable_count",
            "geo_median_price_20",
            "geo_median_ppsf_20",
        ]

        results = np.full(
            (len(homes), len(columns)),
            np.nan,
            dtype=np.float32
        )

        # Process homes in smaller batches
        batch_size = 500
        positions = {}

        for kind, data in self.data.items():
            positions[kind] = data

        for start in range(0, len(homes), batch_size):

            batch = homes.iloc[start:start + batch_size]

            for j, (_, home) in enumerate(batch.iterrows()):

                data = positions.get(home["house_type_name"])

                if data is None:
                    continue

                if pd.isna(home["latitude"]) or pd.isna(home["longitude"]):
                    continue

                cutoff = np.datetime64(
                    pd.Timestamp(home["date_listed"])
                )

                point = np.array([
                    float(home["latitude"]),
                    float(home["longitude"]) * 0.72
                ])

                # Never request thousands of neighbours
                k = min(100, len(data["dates"]))

                distances, indices = data["tree"].query(
                    point,
                    k=k,
                    workers=1
                )

                indices = np.atleast_1d(indices)
                distances = np.atleast_1d(distances)

                mask = data["dates"][indices] < cutoff

                selected = indices[mask]
                selected_distances = distances[mask]

                if len(selected) == 0:
                    continue

                prices = data["prices"][selected]
                areas = data["areas"][selected]

                def median(values):
                    values = values[np.isfinite(values)]
                    return np.median(values) if len(values) else np.nan

                pps = prices / np.where(
                    areas > 0,
                    areas,
                    np.nan
                )

                results[start + j] = [
                    median(prices[:5]),
                    median(prices[:10]),
                    median(pps[:10]),
                    float(selected_distances[0]),
                    median(areas[:10]),
                    len(selected),
                    median(prices[:20]),
                    median(pps[:20])
                ]

            print(
                f"Spatial features: "
                f"{min(start + batch_size, len(homes)):,}/"
                f"{len(homes):,}",
                end="\r"
            )

        return pd.DataFrame(
            results,
            columns=columns,
            index=homes.index
        )
