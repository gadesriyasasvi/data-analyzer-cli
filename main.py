import argparse

import uvicorn

from src.pipeline import DataPipeline
from src.scraper import WebScraper


def build_parser():
    parser = argparse.ArgumentParser(
        description="Command-line data intelligence tool"
    )

    parser.add_argument(
        "--file",
        help="Path to a CSV or JSON dataset"
    )

    parser.add_argument(
        "--scrape-quotes",
        action="store_true",
        help="Scrape quotes and process the result"
    )

    parser.add_argument(
        "--api",
        action="store_true",
        help="Start the REST API"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="API host"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.api:
        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            reload=False
        )
        return

    if args.scrape_quotes:
        path = WebScraper().save_quotes()
    elif args.file:
        path = args.file
    else:
        parser.error(
            "Use --file, --scrape-quotes, or --api."
        )

    DataPipeline().run(path)


if __name__ == "__main__":
    main()
