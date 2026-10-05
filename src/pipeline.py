import json
from pathlib import Path

import pandas as pd

from .loader import DataLoader
from .profiler import (
    DatasetValidator,
    DatasetProfiler
)
from .cleaner import (
    DataCleaner,
    DataTransformer,
    OutlierDetector
)
from .analyzer import DataAnalyzer
from .visualizer import DataVisualizer
from .report import EDAReportGenerator
from .storage import StorageManager


PROJECT_DIR = Path(
    __file__
).resolve().parents[1]

DATA_DIR = (
    PROJECT_DIR / "data"
)

PROCESSED_DIR = (
    DATA_DIR / "processed"
)

OUTPUT_DIR = (
    PROJECT_DIR / "output"
)

DATABASE_DIR = (
    PROJECT_DIR / "database"
)

DB_PATH = (
    DATABASE_DIR
    / "data_intelligence.db"
)


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(item)
            for item in value
        ]

    if pd.isna(value):
        return None

    return value


class DataPipeline:
    def __init__(self):
        PROCESSED_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        DATABASE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.loader = DataLoader()
        self.validator = DatasetValidator()
        self.profiler = DatasetProfiler()
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.outlier_detector = OutlierDetector()
        self.analyzer = DataAnalyzer()
        self.visualizer = DataVisualizer()
        self.report_generator = (
            EDAReportGenerator()
        )
        self.storage = StorageManager(
            DB_PATH
        )

    def run(self, file_path):
        file_path = Path(file_path)
        dataset_name = file_path.stem

        print(
            f"\nDataset: {dataset_name}"
        )
        print(
            f"Source: {file_path}"
        )

        print("\nLoading...")
        df = self.loader.load(
            file_path
        )

        print(
            f"Rows: {len(df):,}"
        )
        print(
            f"Columns: {len(df.columns):,}"
        )

        print("\nValidating...")
        validation = (
            self.validator.validate(df)
        )

        print(
            f"Missing cells: "
            f"{validation['missing_cells']:,}"
        )

        print(
            f"Duplicate rows: "
            f"{validation['duplicate_rows']:,}"
        )

        print("\nProfiling...")
        profile = (
            self.profiler.profile(df)
        )

        print(
            "Column roles:",
            profile["role"]
            .value_counts()
            .to_dict()
        )

        print("\nCleaning...")
        cleaned_df, cleaning_log = (
            self.cleaner.clean(
                df,
                profile
            )
        )

        print(
            f"Rows: "
            f"{cleaning_log['rows_before']:,} "
            f"-> "
            f"{cleaning_log['rows_after']:,}"
        )

        print(
            f"Missing values: "
            f"{cleaning_log['missing_before']:,} "
            f"-> "
            f"{cleaning_log['missing_after']:,}"
        )

        print(
            f"Duplicates removed: "
            f"{cleaning_log['duplicates_removed']:,}"
        )

        print("\nTransforming...")
        normalized_df, normalized_columns = (
            self.transformer.min_max_normalize(
                cleaned_df,
                profile
            )
        )

        print(
            f"Normalized columns: "
            f"{len(normalized_columns)}"
        )

        print("\nAnalyzing...")
        numeric_stats = (
            self.analyzer.numerical_statistics(
                cleaned_df,
                profile
            )
        )

        discrete_stats = (
            self.analyzer.discrete_statistics(
                cleaned_df,
                profile
            )
        )

        categorical_stats = (
            self.analyzer.categorical_statistics(
                cleaned_df,
                profile
            )
        )

        text_stats = (
            self.analyzer.text_statistics(
                cleaned_df,
                profile
            )
        )

        correlations = (
            self.analyzer.correlation_matrix(
                cleaned_df,
                profile
            )
        )

        outliers = (
            self.outlier_detector.detect(
                cleaned_df,
                profile
            )
        )

        insights = (
            self.analyzer.generate_insights(
                validation,
                profile,
                numeric_stats,
                discrete_stats,
                categorical_stats,
                text_stats,
                correlations,
                outliers
            )
        )

        print(
            f"Insights: {len(insights)}"
        )

        dataset_output = (
            OUTPUT_DIR / dataset_name
        )

        plot_dir = (
            dataset_output / "plots"
        )

        report_dir = (
            dataset_output / "reports"
        )

        stats_dir = (
            dataset_output / "statistics"
        )

        for folder in [
            plot_dir,
            report_dir,
            stats_dir
        ]:
            folder.mkdir(
                parents=True,
                exist_ok=True
            )

        print("\nCreating visualizations...")
        plot_paths = (
            self.visualizer.create_plots(
                cleaned_df,
                profile,
                correlations,
                plot_dir
            )
        )

        print(
            f"Plots: {len(plot_paths)}"
        )

        report_path = (
            report_dir
            / "eda_report.html"
        )

        self.report_generator.generate(
            dataset_name=dataset_name,
            source_path=file_path,
            validation=validation,
            profile=profile,
            cleaning_log=cleaning_log,
            numeric_stats=numeric_stats,
            discrete_stats=discrete_stats,
            categorical_stats=categorical_stats,
            text_stats=text_stats,
            correlations=correlations,
            outliers=outliers,
            insights=insights,
            plot_paths=plot_paths,
            output_path=report_path,
            sample_df=cleaned_df.head(10)
        )

        cleaned_path = (
            PROCESSED_DIR
            / f"{dataset_name}_cleaned.csv"
        )

        normalized_path = (
            PROCESSED_DIR
            / f"{dataset_name}_normalized.csv"
        )

        cleaned_df.to_csv(
            cleaned_path,
            index=False
        )

        normalized_df.to_csv(
            normalized_path,
            index=False
        )

        profile.to_csv(
            stats_dir
            / "column_profile.csv",
            index=False
        )

        numeric_stats.to_json(
            stats_dir
            / "numerical_statistics.json",
            orient="records",
            indent=2
        )

        discrete_stats.to_json(
            stats_dir
            / "discrete_statistics.json",
            orient="records",
            indent=2
        )

        categorical_stats.to_json(
            stats_dir
            / "categorical_statistics.json",
            orient="records",
            indent=2
        )

        text_stats.to_json(
            stats_dir
            / "text_statistics.json",
            orient="records",
            indent=2
        )

        (
            stats_dir
            / "correlations.json"
        ).write_text(
            json.dumps(
                json_safe(
                    correlations.round(
                        4
                    ).to_dict()
                    if not correlations.empty
                    else {}
                ),
                indent=2
            ),
            encoding="utf-8"
        )

        (
            stats_dir
            / "outliers.json"
        ).write_text(
            (
                outliers.to_json(
                    orient="records",
                    indent=2
                )
                if not outliers.empty
                else "[]"
            ),
            encoding="utf-8"
        )

        (
            stats_dir
            / "insights.json"
        ).write_text(
            json.dumps(
                json_safe(insights),
                indent=2
            ),
            encoding="utf-8"
        )

        run_id = (
            self.storage.save_run(
                dataset_name,
                file_path,
                validation,
                cleaning_log,
                profile,
                numeric_stats,
                discrete_stats,
                categorical_stats,
                outliers
            )
        )

        for step in [
            "load",
            "validate",
            "profile",
            "clean",
            "transform",
            "analyze",
            "visualize",
            "report",
            "store"
        ]:
            self.storage.log_step(
                run_id,
                step,
                "success"
            )

        latest_result = {
            "info": {
                "dataset_name":
                    dataset_name,
                "source":
                    str(file_path),
                "rows":
                    int(len(cleaned_df)),
                "columns":
                    int(len(cleaned_df.columns)),
                "run_id":
                    int(run_id)
            },
            "preview": json_safe(
                cleaned_df
                .head(10)
                .to_dict(
                    orient="records"
                )
            ),
            "statistics": {
                "continuous_numerical":
                    json_safe(
                        numeric_stats.to_dict(
                            orient="records"
                        )
                    ),
                "discrete_numerical":
                    json_safe(
                        discrete_stats.to_dict(
                            orient="records"
                        )
                    ),
                "categorical":
                    json_safe(
                        categorical_stats.to_dict(
                            orient="records"
                        )
                    ),
                "text":
                    json_safe(
                        text_stats.to_dict(
                            orient="records"
                        )
                    )
            },
            "missing_values":
                json_safe(
                    profile[
                        [
                            "column",
                            "missing_count",
                            "missing_pct"
                        ]
                    ].to_dict(
                        orient="records"
                    )
                ),
            "correlations":
                json_safe(
                    correlations.round(
                        4
                    ).to_dict()
                    if not correlations.empty
                    else {}
                )
        }

        latest_result_path = (
            OUTPUT_DIR
            / "latest_result.json"
        )

        latest_result_path.write_text(
            json.dumps(
                latest_result,
                indent=2,
                default=str
            ),
            encoding="utf-8"
        )

        print("\nFinished.")
        print(
            f"Report: {report_path}"
        )
        print(
            f"Processed data: {cleaned_path}"
        )
        print(
            f"Database run: {run_id}"
        )

        print("\nInsights:")

        for insight in insights:
            print(
                f"- {insight}"
            )

        return latest_result
