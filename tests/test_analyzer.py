import pandas as pd

from src.analyzer import DataAnalyzer
from src.profiler import DatasetProfiler


def test_numerical_statistics():
    df = pd.DataFrame({
        "age": [10, 20, 30, 40]
    })

    profile = DatasetProfiler().profile(df)

    stats = DataAnalyzer().numerical_statistics(
        df,
        profile
    )

    assert stats.iloc[0]["mean"] == 25.0
    assert stats.iloc[0]["median"] == 25.0
