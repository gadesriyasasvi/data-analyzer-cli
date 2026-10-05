import pandas as pd

from src.cleaner import DataCleaner
from src.profiler import DatasetProfiler


def test_cleaner_removes_missing_values():
    df = pd.DataFrame({
        "age": [20, None, 30],
        "city": ["A", "A", None]
    })

    profile = DatasetProfiler().profile(df)

    cleaned, log = DataCleaner().clean(
        df,
        profile
    )

    assert cleaned.isna().sum().sum() == 0
    assert log["missing_before"] == 2
