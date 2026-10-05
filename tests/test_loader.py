import json

import pandas as pd

from src.loader import DataLoader


def test_csv_loader(tmp_path):
    path = tmp_path / "sample.csv"

    pd.DataFrame({
        "age": [20, 30],
        "city": ["A", "B"]
    }).to_csv(
        path,
        index=False
    )

    df = DataLoader().load(path)

    assert df.shape == (2, 2)


def test_json_loader(tmp_path):
    path = tmp_path / "sample.json"

    data = [
        {"value": 1},
        {"value": 2}
    ]

    path.write_text(
        json.dumps(data),
        encoding="utf-8"
    )

    df = DataLoader().load(path)

    assert df["value"].tolist() == [1, 2]
