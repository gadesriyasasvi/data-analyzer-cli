import sqlite3
from pathlib import Path

import pandas as pd


class StorageManager:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.create_tables()

    def connect(self):
        return sqlite3.connect(
            self.db_path
        )

    def create_tables(self):
        with self.connect() as connection:
            cursor = connection.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    rows_before INTEGER,
                    rows_after INTEGER,
                    columns_count INTEGER,
                    missing_before INTEGER,
                    missing_after INTEGER,
                    duplicates_removed INTEGER,
                    outlier_cells INTEGER
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS column_profiles (
                    run_id INTEGER,
                    column_name TEXT,
                    role TEXT,
                    dtype TEXT,
                    missing_count INTEGER,
                    missing_pct REAL,
                    unique_count INTEGER,
                    FOREIGN KEY(run_id)
                        REFERENCES dataset_runs(run_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS numerical_statistics (
                    run_id INTEGER,
                    column_name TEXT,
                    count INTEGER,
                    mean REAL,
                    median REAL,
                    std REAL,
                    variance REAL,
                    minimum REAL,
                    q1 REAL,
                    q3 REAL,
                    maximum REAL,
                    range_value REAL,
                    FOREIGN KEY(run_id)
                        REFERENCES dataset_runs(run_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discrete_statistics (
                    run_id INTEGER,
                    column_name TEXT,
                    count INTEGER,
                    unique_count INTEGER,
                    minimum REAL,
                    maximum REAL,
                    mean REAL,
                    median REAL,
                    mode REAL,
                    mode_frequency INTEGER,
                    mode_percentage REAL,
                    FOREIGN KEY(run_id)
                        REFERENCES dataset_runs(run_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS categorical_statistics (
                    run_id INTEGER,
                    column_name TEXT,
                    unique_count INTEGER,
                    top_value TEXT,
                    top_frequency INTEGER,
                    top_percentage REAL,
                    FOREIGN KEY(run_id)
                        REFERENCES dataset_runs(run_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    step TEXT,
                    status TEXT,
                    created_at TEXT,
                    FOREIGN KEY(run_id)
                        REFERENCES dataset_runs(run_id)
                )
            """)

    def save_run(
        self,
        dataset_name,
        source_path,
        validation,
        cleaning_log,
        profile,
        numeric_stats,
        discrete_stats,
        categorical_stats,
        outliers
    ):
        with self.connect() as connection:
            cursor = connection.cursor()

            outlier_count = (
                int(
                    outliers["outlier_count"].sum()
                )
                if not outliers.empty
                else 0
            )

            cursor.execute("""
                INSERT INTO dataset_runs (
                    dataset_name,
                    source_path,
                    processed_at,
                    rows_before,
                    rows_after,
                    columns_count,
                    missing_before,
                    missing_after,
                    duplicates_removed,
                    outlier_cells
                )
                VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_name,
                str(source_path),
                cleaning_log["rows_before"],
                cleaning_log["rows_after"],
                validation["columns"],
                cleaning_log["missing_before"],
                cleaning_log["missing_after"],
                cleaning_log["duplicates_removed"],
                outlier_count
            ))

            run_id = int(cursor.lastrowid)

            for _, row in profile.iterrows():
                cursor.execute("""
                    INSERT INTO column_profiles (
                        run_id,
                        column_name,
                        role,
                        dtype,
                        missing_count,
                        missing_pct,
                        unique_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    row["column"],
                    row["role"],
                    row["dtype"],
                    int(row["missing_count"]),
                    float(row["missing_pct"]),
                    int(row["unique_count"])
                ))

            for _, row in numeric_stats.iterrows():
                cursor.execute("""
                    INSERT INTO numerical_statistics (
                        run_id,
                        column_name,
                        count,
                        mean,
                        median,
                        std,
                        variance,
                        minimum,
                        q1,
                        q3,
                        maximum,
                        range_value
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    row["column"],
                    int(row["count"]),
                    float(row["mean"]),
                    float(row["median"]),
                    float(row["std"]),
                    float(row["variance"]),
                    float(row["min"]),
                    float(row["q1"]),
                    float(row["q3"]),
                    float(row["max"]),
                    float(row["range"])
                ))

            for _, row in discrete_stats.iterrows():
                cursor.execute("""
                    INSERT INTO discrete_statistics (
                        run_id,
                        column_name,
                        count,
                        unique_count,
                        minimum,
                        maximum,
                        mean,
                        median,
                        mode,
                        mode_frequency,
                        mode_percentage
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    row["column"],
                    int(row["count"]),
                    int(row["unique_count"]),
                    float(row["min"]),
                    float(row["max"]),
                    float(row["mean"]),
                    float(row["median"]),
                    float(row["mode"]),
                    int(row["mode_frequency"]),
                    float(row["mode_percentage"])
                ))

            for _, row in categorical_stats.iterrows():
                cursor.execute("""
                    INSERT INTO categorical_statistics (
                        run_id,
                        column_name,
                        unique_count,
                        top_value,
                        top_frequency,
                        top_percentage
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    row["column"],
                    int(row["unique_count"]),
                    str(row["top_value"]),
                    int(row["top_frequency"]),
                    float(row["top_percentage"])
                ))

            return run_id

    def log_step(
        self,
        run_id,
        step,
        status
    ):
        with self.connect() as connection:
            connection.execute("""
                INSERT INTO processing_history (
                    run_id,
                    step,
                    status,
                    created_at
                )
                VALUES (?, ?, ?, datetime('now'))
            """, (
                run_id,
                step,
                status
            ))

    def recent_runs(self, limit=10):
        with self.connect() as connection:
            return pd.read_sql_query(
                """
                SELECT
                    run_id,
                    dataset_name,
                    source_path,
                    processed_at,
                    rows_before,
                    rows_after,
                    columns_count,
                    missing_before,
                    missing_after,
                    duplicates_removed,
                    outlier_cells
                FROM dataset_runs
                ORDER BY run_id DESC
                LIMIT ?
                """,
                connection,
                params=(limit,)
            )
