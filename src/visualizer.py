import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class DataVisualizer:
    def __init__(self):
        self.generated_plots = []

    def _safe_name(self, text):
        return re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            str(text)
        ).strip("_").lower()

    def _save(self, figure, path):
        figure.savefig(
            path,
            dpi=160,
            bbox_inches="tight"
        )
        plt.close(figure)
        self.generated_plots.append(
            Path(path)
        )

    def create_histograms(
        self,
        df,
        profile,
        output_dir
    ):
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
            ).dropna()

            if len(values) == 0:
                continue

            figure = plt.figure(
                figsize=(9, 5)
            )

            plt.hist(
                values,
                bins=30
            )

            plt.title(
                f"Distribution of {column}"
            )
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.grid(
                axis="y",
                alpha=0.25
            )
            plt.tight_layout()

            self._save(
                figure,
                output_dir
                / (
                    f"hist_"
                    f"{self._safe_name(column)}.png"
                )
            )

    def create_boxplots(
        self,
        df,
        profile,
        output_dir
    ):
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
            ).dropna()

            if len(values) == 0:
                continue

            figure = plt.figure(
                figsize=(9, 4)
            )

            plt.boxplot(
                values,
                vert=False
            )

            plt.title(
                f"Box Plot of {column}"
            )
            plt.xlabel(column)
            plt.grid(
                axis="x",
                alpha=0.25
            )
            plt.tight_layout()

            self._save(
                figure,
                output_dir
                / (
                    f"box_"
                    f"{self._safe_name(column)}.png"
                )
            )

    def create_discrete_charts(
        self,
        df,
        profile,
        output_dir
    ):
        columns = (
            profile.loc[
                profile["role"] == "discrete",
                "column"
            ].tolist()
        )

        for column in columns:
            counts = (
                df[column]
                .value_counts()
                .sort_index()
            )

            if len(counts) == 0:
                continue

            figure = plt.figure(
                figsize=(8, 5)
            )

            plt.bar(
                counts.index.astype(str),
                counts.values
            )

            plt.title(
                f"Distribution of {column}"
            )
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.tight_layout()

            self._save(
                figure,
                output_dir
                / (
                    f"discrete_"
                    f"{self._safe_name(column)}.png"
                )
            )

    def create_categorical_charts(
        self,
        df,
        profile,
        output_dir
    ):
        columns = (
            profile.loc[
                profile["role"] == "categorical",
                "column"
            ].tolist()
        )

        for column in columns:
            counts = (
                df[column]
                .astype(str)
                .value_counts()
                .head(10)
            )

            if len(counts) == 0:
                continue

            figure = plt.figure(
                figsize=(10, 5)
            )

            plt.bar(
                counts.index.astype(str),
                counts.values
            )

            plt.title(
                f"Top Categories in {column}"
            )
            plt.xlabel(column)
            plt.ylabel("Count")

            plt.xticks(
                rotation=45,
                ha="right"
            )

            plt.tight_layout()

            self._save(
                figure,
                output_dir
                / (
                    f"category_"
                    f"{self._safe_name(column)}.png"
                )
            )

    def create_scatter_plot(
        self,
        df,
        correlations,
        output_dir
    ):
        if correlations.empty:
            return

        columns = correlations.columns.tolist()
        pairs = []

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

        if not pairs:
            return

        first, second, value = max(
            pairs,
            key=lambda item: abs(item[2])
        )

        x = pd.to_numeric(
            df[first],
            errors="coerce"
        )

        y = pd.to_numeric(
            df[second],
            errors="coerce"
        )

        valid = x.notna() & y.notna()

        if valid.sum() == 0:
            return

        figure = plt.figure(
            figsize=(8, 5)
        )

        plt.scatter(
            x[valid],
            y[valid],
            alpha=0.45
        )

        plt.title(
            f"{first} vs {second} "
            f"(r = {value:.3f})"
        )

        plt.xlabel(first)
        plt.ylabel(second)
        plt.grid(
            alpha=0.25
        )
        plt.tight_layout()

        self._save(
            figure,
            output_dir
            / (
                f"scatter_"
                f"{self._safe_name(first)}_"
                f"{self._safe_name(second)}.png"
            )
        )

    def create_heatmap(
        self,
        correlations,
        output_dir
    ):
        if correlations.empty:
            return

        figure = plt.figure(
            figsize=(10, 8)
        )

        sns.heatmap(
            correlations,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0
        )

        plt.title(
            "Numerical Correlation Heatmap"
        )

        plt.tight_layout()

        self._save(
            figure,
            output_dir
            / "correlation_heatmap.png"
        )

    def create_plots(
        self,
        df,
        profile,
        correlations,
        output_dir
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.generated_plots = []

        self.create_histograms(
            df,
            profile,
            output_dir
        )

        self.create_boxplots(
            df,
            profile,
            output_dir
        )

        self.create_discrete_charts(
            df,
            profile,
            output_dir
        )

        self.create_categorical_charts(
            df,
            profile,
            output_dir
        )

        self.create_scatter_plot(
            df,
            correlations,
            output_dir
        )

        self.create_heatmap(
            correlations,
            output_dir
        )

        return self.generated_plots
