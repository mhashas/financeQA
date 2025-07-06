import shutil
from pathlib import Path

import click
import fitz
import pymupdf
from fitz import Document, Page
from PIL import Image
from tqdm import tqdm

from financeqa.preprocessing.data_models import DetectionResult
from financeqa.preprocessing.tables.table_detector import TableDetector
from shared.hidden_prints import HiddenPrints


def extract_tables_from_page(page: Page, table_detector: TableDetector) -> list[DetectionResult]:
    """Extract tables from the given PDF page

    Args:
        page: PDF page from which to extract tables
        table_detector: table detector to use

    Returns:
        list of images containing detected tables
    """
    mat = pymupdf.Matrix(3, 3)  # zoom factor 3 in each dimension
    pix = page.get_pixmap(matrix=mat)  # type: ignore
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # type: ignore

    with HiddenPrints():
        tables = table_detector.detect(image)

    return tables


def extract_tables_from_pdf(pdf: Document, table_detector: TableDetector) -> dict[int, list[DetectionResult]]:
    """Extract tables from the given PDF document

    Args:
        pdf: PDF document from which to extract tables
        table_detector: table detector to use

    Returns:
        dictionary mapping page numbers to a list of images containing detected tables
    """
    result = {}

    for page in pdf:
        tables = extract_tables_from_page(page, table_detector)

        if len(tables) > 0:
            result[page.number] = tables

    return result


@click.command()
@click.option("--input_dir", default="./data/docs/pdf/")
@click.option("--output_dir", default="./data/tables/")
def main(input_dir: str, output_dir: str):
    docs_root = Path(input_dir)
    output_root = Path(output_dir)

    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    table_detector = TableDetector()
    docs_paths = list(docs_root.glob("*.pdf"))

    for doc_path in tqdm(docs_paths):
        doc_name = doc_path.stem
        doc_output_dir = output_root / doc_name

        pdf = fitz.open(doc_path)
        tables = extract_tables_from_pdf(pdf, table_detector)
        if len(tables) > 0:
            doc_output_dir.mkdir(parents=True, exist_ok=True)

        for page_num, table_images in tables.items():
            for i, table_image in enumerate(table_images):
                table_image.image.save(doc_output_dir / f"{page_num}_{i}.png")


if __name__ == "__main__":
    main()
