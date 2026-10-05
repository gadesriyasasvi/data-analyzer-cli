import pandas as pd


class DatasetValidator:

    def validate(self, df):

        if df.empty:
            raise ValueError(
                "The dataset is empty."
            )

        if len(df.columns) == 0:
            raise ValueError(
                "The dataset has no columns."
            )

        if not df.columns.is_unique:
            raise ValueError(
                "Column names must be unique."
            )

        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "missing_cells": int(
                df.isna().sum().sum()
            ),
            "duplicate_rows": int(
                df.duplicated().sum()
            ),
            "all_null_columns": [
                str(column)
                for column in df.columns
                if df[column].isna().all()
            ],
            "memory_mb": round(
                float(
                    df.memory_usage(
                        deep=True
                    ).sum() / (1024 ** 2)
                ),
                3
            )
        }


class DatasetProfiler:

    def _looks_like_datetime(
        self,
        column,
        series
    ):

        if series.dtype != "object":
            return False

        sample = (
            series
            .dropna()
            .astype(str)
            .str.strip()
            .head(300)
        )

        if len(sample) == 0:
            return False

        name = str(column).lower()

        name_hint = any(
            word in name
            for word in [
                "date",
                "time",
                "month"
            ]
        )

        pattern_ratio = sample.str.contains(
            r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}"
            r"|[A-Za-z]{3,12}\s+\d{1,2}",
            regex=True
        ).mean()

        if not name_hint and pattern_ratio < 0.5:
            return False

        parsed = pd.to_datetime(
            sample,
            errors="coerce"
        )

        return (
            float(
                parsed.notna().mean()
            ) >= 0.8
        )

    def classify_column(
        self,
        column,
        series
    ):

        name = str(column).lower().strip()
        non_null = series.dropna()

        if self._looks_like_datetime(
            column,
            series
        ):
            return "datetime"

        if pd.api.types.is_bool_dtype(series):
            return "categorical"

        if pd.api.types.is_numeric_dtype(series):

            unique_count = int(
                series.nunique(
                    dropna=True
                )
            )

            row_count = max(
                len(non_null),
                1
            )

            unique_ratio = (
                unique_count
                / row_count
            )

            if (
                unique_ratio >= 0.95
                and any(
                    word in name
                    for word in [
                        "id",
                        "identifier",
                        "key",
                        "code"
                    ]
                )
            ):
                return "identifier"

            # Low-cardinality integers are treated as
            # discrete only when they repeat often enough.
            if (
                pd.api.types.is_integer_dtype(series)
                and unique_count <= 10
                and unique_ratio <= 0.10
            ):
                return "discrete"

            return "numerical"

        text = (
            non_null
            .astype(str)
            .str.strip()
        )

        if len(text) == 0:
            return "categorical"

        unique_count = int(
            text.nunique()
        )

        unique_ratio = (
            unique_count
            / max(len(text), 1)
        )

        average_length = float(
            text.str.len().mean()
        )

        if (
            any(
                word in name
                for word in [
                    "id",
                    "identifier",
                    "key",
                    "code"
                ]
            )
            and unique_ratio >= 0.80
        ):
            return "identifier"

        if (
            unique_ratio > 0.60
            and average_length > 15
        ):
            return "text"

        return "categorical"

    def profile(self, df):

        rows = []

        for column in df.columns:

            series = df[column]

            rows.append({
                "column": str(column),
                "role": self.classify_column(
                    column,
                    series
                ),
                "dtype": str(series.dtype),
                "missing_count": int(
                    series.isna().sum()
                ),
                "missing_pct": round(
                    float(
                        series.isna().mean() * 100
                    ),
                    2
                ),
                "unique_count": int(
                    series.nunique(
                        dropna=True
                    )
                ),
                "sample_values": ", ".join(
                    series
                    .dropna()
                    .astype(str)
                    .head(3)
                    .tolist()
                )
            })

        return pd.DataFrame(rows)