# AI Data Intelligence CLI

A general-purpose Python command-line tool for working with CSV and JSON
datasets.

The tool can:

- validate and profile datasets
- clean missing values and duplicates
- standardize categorical values
- detect continuous numerical outliers
- calculate statistics and correlations
- create automatic visualizations
- generate self-contained HTML EDA reports
- save processed data and JSON results
- store processing information in SQLite
- scrape structured data from a web page
- expose the latest analysis through a REST API

## Project structure

```text
AI_Data_Intelligence/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
├── data/
│   ├── raw/
│   └── processed/
├── output/
├── database/
└── tests/
```

## Setup

Create and activate a virtual environment, then install the dependencies.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Put CSV or JSON files in `data/raw/`.

## Run

```bash
python main.py --file data/raw/titanic.csv
```

Run the web-scraping example:

```bash
python main.py --scrape-quotes
```

Start the API:

```bash
python main.py --api
```

The API runs at:

```text
http://127.0.0.1:8000
```

## Output

Processed data is saved under `data/processed/`.

Dataset reports, plots and statistics are saved under:

```text
output/<dataset_name>/
```

SQLite data is stored in:

```text
database/data_intelligence.db
```

## Git

Initialize the repository:

```bash
git init
git add .
git commit -m "Initial project"
```
