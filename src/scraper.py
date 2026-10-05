from pathlib import Path

import requests
from bs4 import BeautifulSoup


class WebScraper:
    def __init__(
        self,
        url="https://quotes.toscrape.com/"
    ):
        self.url = url

    def fetch(self):
        response = requests.get(
            self.url,
            headers={
                "User-Agent":
                    "AI-Data-Intelligence/1.0"
            },
            timeout=20
        )

        response.raise_for_status()

        return response.text

    def scrape_quotes(self):
        html = self.fetch()
        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        records = []

        for block in soup.select(
            "div.quote"
        ):
            quote = block.select_one(
                "span.text"
            )

            author = block.select_one(
                "small.author"
            )

            tags = [
                tag.get_text(
                    strip=True
                )
                for tag in block.select(
                    "a.tag"
                )
            ]

            records.append({
                "quote":
                    (
                        quote.get_text(
                            strip=True
                        )
                        if quote
                        else ""
                    ),
                "author":
                    (
                        author.get_text(
                            strip=True
                        )
                        if author
                        else ""
                    ),
                "tags":
                    ", ".join(tags)
            })

        return records

    def save_quotes(
        self,
        output_path="data/processed/scraped_quotes.csv"
    ):
        import pandas as pd

        records = self.scrape_quotes()

        path = Path(output_path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df = pd.DataFrame(records)
        df.to_csv(
            path,
            index=False
        )

        return path
