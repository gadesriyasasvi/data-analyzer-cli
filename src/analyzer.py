import numpy as np
import pandas as pd


class DataAnalyzer:
    def numerical_statistics(
        self,
        df,
        profile
    ):
        rows = []

        columns = (
            profile.loc[
                profile["role"] == "numerical",
                "column"
            ].tolist()
        )

        for column in columns:
            values = pd.to_numeric(
                df[column],
                errors="coerce"
            ).dropna().to_numpy()

            if len(values) == 0:
                continue

            rows.append({
                "column": column,
                "count": int(len(values)),
                "mean": round(
                    float(np.mean(values)),
                    4
                ),
                "median": round(
                    float(np.median(values)),
                    4
                ),
                "std": round(
                    float(
                        np.std(
                            values,
                            ddof=1
                        )
                    ) if len(values) > 1 else 0.0,
                    4
                ),
                "variance": round(
                    float(
                        np.var(
                            values,
                            ddof=1
                        )
                    ) if len(values) > 1 else 0.0,
                    4
                ),
                "min": round(
                    float(np.min(values)),
                    4
                ),
                "q1": round(
                    float(
                        np.percentile(
                            values,
                            25
                        )
                    ),
                    4
                ),
                "q3": round(
                    float(
                        np.percentile(
                            values,
                            75
                        )
                    ),
                    4
                ),
                "max": round(
                    float(np.max(values)),
                    4
                ),
                "range": round(
                    float(
                        np.max(values)
                        - np.min(values)
                    ),
                    4
                )
            })

        return pd.DataFrame(rows)

    def discrete_statistics(
        self,
        df,
        profile
    ):
        rows = []

        columns = (
            profile.loc[
                profile["role"] == "discrete",
                "column"
            ].tolist()
        )

        for column in columns:
            values = pd.to_numeric(
                df[column],
                errors="coerce"
            ).dropna()

            if len(values) == 0:
                continue

            counts = values.value_counts()

            rows.append({
                "column": column,
                "count": int(len(values)),
                "unique_count": int(
                    values.nunique()
                ),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": round(
                    float(values.mean()),
                    4
                ),
                "median": round(
                    float(values.median()),
                    4
                ),
                "mode": float(counts.index[0]),
                "mode_frequency": int(
                    counts.iloc[0]
                ),
                "mode_percentage": round(
                    float(
                        counts.iloc[0]
                        / len(values)
                        * 100
                    ),
                    2
                )
            })

        return pd.DataFrame(rows)

    def categorical_statistics(
        self,
        df,
        profile
    ):
        rows = []

        columns = (
            profile.loc[
                profile["role"] == "categorical",
                "column"
            ].tolist()
        )

        for column in columns:
            values = (
                df[column]
                .dropna()
                .astype(str)
                .str.strip()
            )

            if len(values) == 0:
                continue

            counts = values.value_counts()

            rows.append({
                "column": column,
                "unique_count": int(
                    values.nunique()
                ),
                "top_value": str(
                    counts.index[0]
                ),
                "top_frequency": int(
                    counts.iloc[0]
                ),
                "top_percentage": round(
                    float(
                        counts.iloc[0]
                        / len(values)
                        * 100
                    ),
                    2
                )
            })

        return pd.DataFrame(rows)

    def text_statistics(
        self,
        df,
        profile
    ):
        rows = []

        columns = (
            profile.loc[
                profile["role"] == "text",
                "column"
            ].tolist()
        )

        for column in columns:
            values = (
                df[column]
                .dropna()
                .astype(str)
                .str.strip()
            )

            if len(values) == 0:
                continue

            lengths = values.str.len()

            rows.append({
                "column": column,
                "non_null_count": int(
                    len(values)
                ),
                "unique_count": int(
                    values.nunique()
                ),
                "average_length": round(
                    float(lengths.mean()),
                    2
                ),
                "minimum_length": int(
                    lengths.min()
                ),
                "maximum_length": int(
                    lengths.max()
                )
            })

        return pd.DataFrame(rows)

    def correlation_matrix(
        self,
        df,
        profile
    ):
        columns = (
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

        if len(columns) < 2:
            return pd.DataFrame()

        data = df[columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        return data.corr(
            method="pearson"
        )

    def generate_insights(
        self,
        validation,
        profile,
        numeric_stats,
        discrete_stats,
        categorical_stats,
        text_stats,
        correlations,
        outliers
    ):
        insights = []

        total_cells = (
            validation["rows"]
            * validation["columns"]
        )

        missing_pct = (
            validation["missing_cells"]
            / max(total_cells, 1)
            * 100
        )

        insights.append(
            f"{validation['rows']:,} rows and "
            f"{validation['columns']:,} columns were found."
        )

        if missing_pct > 0:
            insights.append(
                f"{missing_pct:.2f}% of cells were "
                "missing before cleaning."
            )

        high_missing = (
            profile[
                profile["missing_pct"] >= 40
            ]["column"]
            .tolist()
        )

        if high_missing:
            insights.append(
                "High missingness: "
                + ", ".join(
                    high_missing[:4]
                )
                + "."
            )

        if validation["duplicate_rows"] == 0:
            insights.append(
                "No exact duplicate rows were detected."
            )
        else:
            insights.append(
                f"{validation['duplicate_rows']:,} "
                "duplicate rows were found."
            )

        if not numeric_stats.empty:
            row = numeric_stats.loc[
                numeric_stats["variance"].idxmax()
            ]

            insights.append(
                f"{row['column']} has the highest "
                f"variance ({row['variance']})."
            )

        if not correlations.empty:
            pairs = []

            columns = correlations.columns.tolist()

            for i in range(len(columns)):
                for j in range(
                    i + 1,
                    len(columns)
                ):
                    value = correlations.iloc[i, j]

                    if not pd.isna(value):
                        pairs.append(
                            (
                                columns[i],
                                columns[j],
                                float(value)
                            )
                        )

            if pairs:
                positive = max(
                    pairs,
                    key=lambda item: item[2]
                )

                negative = min(
                    pairs,
                    key=lambda item: item[2]
                )

                if positive[2] >= 0.50:
                    insights.append(
                        f"Strongest positive correlation: "
                        f"{positive[0]} and "
                        f"{positive[1]} "
                        f"({positive[2]:.3f})."
                    )

                if negative[2] <= -0.50:
                    insights.append(
                        f"Strongest negative correlation: "
                        f"{negative[0]} and "
                        f"{negative[1]} "
                        f"({negative[2]:.3f})."
                    )

        if not categorical_stats.empty:
            meaningful = (
                categorical_stats[
                    (
                        categorical_stats[
                            "top_percentage"
                        ] >= 40
                    )
                    &
                    (
                        categorical_stats[
                            "top_value"
                        ]
                        .astype(str)
                        .str.casefold()
                        != "unknown"
                    )
                ]
            )

            if not meaningful.empty:
                row = (
                    meaningful
                    .sort_values(
                        "top_percentage",
                        ascending=False
                    )
                    .iloc[0]
                )

                insights.append(
                    f"{row['column']} is concentrated "
                    f"around '{row['top_value']}' "
                    f"({row['top_percentage']}%)."
                )

        if not outliers.empty:
            outlier_rows = (
                outliers[
                    outliers["outlier_count"] > 0
                ]
                .sort_values(
                    "outlier_count",
                    ascending=False
                )
            )

            if not outlier_rows.empty:
                row = outlier_rows.iloc[0]

                insights.append(
                    f"{row['column']} has "
                    f"{int(row['outlier_count']):,} "
                    "potential outliers by the IQR rule."
                )

        return insights[:8]
