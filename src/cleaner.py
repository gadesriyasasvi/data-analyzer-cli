from collections import Counter

import pandas as pd
import numpy as np


class DataCleaner:
    def _standardize_categorical_case(
        self,
        series
    ):
        values = (
            series
            .astype("string")
            .str.strip()
        )

        groups = {}

        for value in values.dropna().unique():
            key = value.casefold()

            groups.setdefault(
                key,
                []
            ).append(value)

        canonical = {}

        for key, items in groups.items():
            canonical[key] = (
                Counter(items)
                .most_common(1)[0][0]
            )

        return values.map(
            lambda value:
            canonical.get(
                value.casefold(),
                value
            )
            if pd.notna(value)
            else value
        )

    def clean(
        self,
        df,
        profile
    ):
        cleaned = df.copy()

        rows_before = len(cleaned)

        missing_before = int(
            cleaned.isna().sum().sum()
        )

        duplicate_count = int(
            cleaned.duplicated().sum()
        )

        object_columns = (
            cleaned
            .select_dtypes(
                include=[
                    "object",
                    "string"
                ]
            )
            .columns
        )

        for column in object_columns:
            cleaned[column] = (
                cleaned[column]
                .astype("string")
                .str.strip()
            )

        categorical_columns = (
            profile.loc[
                profile["role"] == "categorical",
                "column"
            ].tolist()
        )

        for column in categorical_columns:
            cleaned[column] = (
                self._standardize_categorical_case(
                    cleaned[column]
                )
            )

        numeric_columns = (
            profile.loc[
                profile["role"].isin(
                    [
                        "numerical",
                        "discrete"
                    ]
                ),
                "column"
            ].tolist()
        )

        for column in numeric_columns:
            if cleaned[column].isna().any():
                median_value = (
                    cleaned[column].median()
                )

                if pd.notna(median_value):
                    cleaned[column] = (
                        cleaned[column]
                        .fillna(median_value)
                    )

        for column in categorical_columns:
            if not cleaned[column].isna().any():
                continue

            missing_pct = (
                cleaned[column]
                .isna()
                .mean()
                * 100
            )

            if missing_pct >= 40:
                cleaned[column] = (
                    cleaned[column]
                    .fillna("Unknown")
                )
            else:
                modes = (
                    cleaned[column]
                    .mode(dropna=True)
                )

                if len(modes):
                    cleaned[column] = (
                        cleaned[column]
                        .fillna(modes.iloc[0])
                    )
                else:
                    cleaned[column] = (
                        cleaned[column]
                        .fillna("Unknown")
                    )

        text_columns = (
            profile.loc[
                profile["role"] == "text",
                "column"
            ].tolist()
        )

        for column in text_columns:
            cleaned[column] = (
                cleaned[column]
                .fillna("Unknown")
            )

        datetime_columns = (
            profile.loc[
                profile["role"] == "datetime",
                "column"
            ].tolist()
        )

        for column in datetime_columns:
            cleaned[column] = pd.to_datetime(
                cleaned[column],
                errors="coerce"
            )

        cleaned = (
            cleaned
            .drop_duplicates()
            .reset_index(drop=True)
        )

        missing_after = int(
            cleaned.isna().sum().sum()
        )

        return cleaned, {
            "rows_before": int(rows_before),
            "rows_after": int(len(cleaned)),
            "missing_before": int(missing_before),
            "missing_after": int(missing_after),
            "duplicates_removed": int(
                duplicate_count
            )
        }


class DataTransformer:
    def min_max_normalize(
        self,
        df,
        profile
    ):
        transformed = df.copy()
        normalized_columns = []

        numerical_columns = (
            profile.loc[
                profile["role"] == "numerical",
                "column"
            ].tolist()
        )

        for column in numerical_columns:
            values = pd.to_numeric(
                transformed[column],
                errors="coerce"
            ).to_numpy()

            if len(values) == 0:
                continue

            minimum = np.nanmin(values)
            maximum = np.nanmax(values)

            if not (
                np.isfinite(minimum)
                and np.isfinite(maximum)
            ):
                continue

            new_column = f"norm_{column}"

            if maximum == minimum:
                transformed[new_column] = 0.0
            else:
                transformed[new_column] = (
                    (values - minimum)
                    / (maximum - minimum)
                )

            normalized_columns.append(
                new_column
            )

        return (
            transformed,
            normalized_columns
        )


class OutlierDetector:
    def detect(
        self,
        df,
        profile
    ):
        results = []

        numerical_columns = (
            profile.loc[
                profile["role"] == "numerical",
                "column"
            ].tolist()
        )

        for column in numerical_columns:
            values = pd.to_numeric(
                df[column],
                errors="coerce"
            ).dropna()

            if len(values) < 4:
                continue

            q1 = float(
                np.percentile(values, 25)
            )

            q3 = float(
                np.percentile(values, 75)
            )

            iqr = q3 - q1

            if iqr == 0:
                lower_bound = q1
                upper_bound = q3
                outlier_count = 0
            else:
                lower_bound = (
                    q1 - 1.5 * iqr
                )

                upper_bound = (
                    q3 + 1.5 * iqr
                )

                outlier_count = int(
                    (
                        (values < lower_bound)
                        |
                        (values > upper_bound)
                    ).sum()
                )

            results.append({
                "column": column,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_bound": round(
                    lower_bound,
                    4
                ),
                "upper_bound": round(
                    upper_bound,
                    4
                ),
                "outlier_count": outlier_count
            })

        return pd.DataFrame(results)
