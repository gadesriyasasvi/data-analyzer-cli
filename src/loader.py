import json
from pathlib import Path

import pandas as pd


class DataLoader:
    def __init__(self):
        self.last_encoding = None

    def load(self, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        extension = file_path.suffix.lower()

        if extension == ".csv":
            encodings = [
                "utf-8",
                "utf-8-sig",
                "cp1252",
                "latin1"
            ]

            last_error = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(
                        file_path,
                        encoding=encoding
                    )
                    self.last_encoding = encoding
                    break
                except UnicodeDecodeError as error:
                    last_error = error
            else:
                raise ValueError(
                    f"Could not decode CSV: {file_path}"
                ) from last_error

        elif extension == ".json":
            try:
                df = pd.read_json(file_path)
            except ValueError:
                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as file:
                    data = json.load(file)

                df = pd.DataFrame(data)

            self.last_encoding = "utf-8"

        else:
            raise ValueError(
                "Supported formats are CSV and JSON."
            )

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Loaded data is not a DataFrame."
            )

        if df.empty:
            raise ValueError(
                "The dataset is empty."
            )

        return df
