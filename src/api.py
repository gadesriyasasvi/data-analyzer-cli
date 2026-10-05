import json
from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
LATEST_RESULT_PATH = (
    PROJECT_DIR
    / "output"
    / "latest_result.json"
)

app = FastAPI(
    title="Data Intelligence API",
    version="1.0"
)


def load_latest_result():
    if not LATEST_RESULT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No processed dataset is available."
        )

    with open(
        LATEST_RESULT_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


@app.get("/")
def root():
    return {
        "name": "Data Intelligence API",
        "status": "running"
    }


@app.get("/dataset/info")
def dataset_info():
    result = load_latest_result()
    return result["info"]


@app.get("/dataset/preview")
def dataset_preview():
    result = load_latest_result()
    return result["preview"]


@app.get("/dataset/statistics")
def dataset_statistics():
    result = load_latest_result()
    return result["statistics"]


@app.get("/dataset/missing-values")
def dataset_missing_values():
    result = load_latest_result()
    return result["missing_values"]


@app.get("/dataset/correlations")
def dataset_correlations():
    result = load_latest_result()
    return result["correlations"]
