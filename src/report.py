import base64
from pathlib import Path
from datetime import datetime

import pandas as pd


class EDAReportGenerator:
    def _image_to_base64(self, image_path):
        with open(
            image_path,
            "rb"
        ) as file:
            return base64.b64encode(
                file.read()
            ).decode("utf-8")

    def generate(
        self,
        dataset_name,
        source_path,
        validation,
        profile,
        cleaning_log,
        numeric_stats,
        discrete_stats,
        categorical_stats,
        text_stats,
        correlations,
        outliers,
        insights,
        plot_paths,
        output_path,
        sample_df
    ):
        plots = []

        for image_path in plot_paths:
            encoded = (
                self._image_to_base64(
                    image_path
                )
            )

            plots.append(
                f"""
                <div class="plot">
                    <h3>{image_path.stem}</h3>
                    <img
                        src="data:image/png;base64,{encoded}"
                    >
                </div>
                """
            )

        validation_table = (
            pd.DataFrame([validation])
            .T
            .rename(columns={0: "Value"})
            .to_html()
        )

        cleaning_table = (
            pd.DataFrame([cleaning_log])
            .T
            .rename(columns={0: "Value"})
            .to_html()
        )

        sections = [
            (
                "Column Profile",
                profile.to_html(index=False)
            ),
            (
                "Continuous Numerical Statistics",
                (
                    numeric_stats.to_html(index=False)
                    if not numeric_stats.empty
                    else "<p>None.</p>"
                )
            ),
            (
                "Discrete Numerical Statistics",
                (
                    discrete_stats.to_html(index=False)
                    if not discrete_stats.empty
                    else "<p>None.</p>"
                )
            ),
            (
                "Categorical Statistics",
                (
                    categorical_stats.to_html(index=False)
                    if not categorical_stats.empty
                    else "<p>None.</p>"
                )
            ),
            (
                "Text Statistics",
                (
                    text_stats.to_html(index=False)
                    if not text_stats.empty
                    else "<p>None.</p>"
                )
            ),
            (
                "Correlation Analysis",
                (
                    correlations.round(4).to_html()
                    if not correlations.empty
                    else "<p>None.</p>"
                )
            ),
            (
                "Outlier Analysis",
                (
                    outliers.to_html(index=False)
                    if not outliers.empty
                    else "<p>None.</p>"
                )
            ),
            (
                "Cleaned Data Sample",
                sample_df.to_html(index=False)
            )
        ]

        section_html = ""

        for title, content in sections:
            section_html += f"""
            <h2>{title}</h2>
            <div class="card">
                {content}
            </div>
            """

        insight_html = "".join(
            f"<li>{item}</li>"
            for item in insights
        )

        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>EDA Report - {dataset_name}</title>

<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 1200px;
    margin: 30px auto;
    padding: 0 20px;
    background: #f5f5f5;
    color: #222;
}}

.header,
.card,
.insights,
.plot {{
    background: #fff;
    padding: 20px;
    margin: 18px 0;
    border-radius: 8px;
}}

table {{
    border-collapse: collapse;
    width: 100%;
}}

th,
td {{
    border: 1px solid #ddd;
    padding: 7px;
}}

th {{
    background: #eee;
}}

.plot img {{
    max-width: 100%;
}}
</style>
</head>

<body>

<div class="header">
<h1>Data Intelligence Report</h1>

<p><b>Dataset:</b> {dataset_name}</p>

<p><b>Source:</b> {source_path}</p>

<p><b>Generated:</b>
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</p>
</div>

<h2>Dataset Overview</h2>

<div class="card">
{validation_table}
</div>

<h2>Cleaning Summary</h2>

<div class="card">
{cleaning_table}
</div>

{section_html}

<h2>Automatic Insights</h2>

<div class="insights">
<ul>
{insight_html}
</ul>
</div>

<h2>Visualizations</h2>

{''.join(plots)}

</body>
</html>
"""

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path.write_text(
            html,
            encoding="utf-8"
        )

        return output_path
