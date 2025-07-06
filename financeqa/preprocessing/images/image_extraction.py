import io
import shutil
from pathlib import Path

import click
import fitz
import pymupdf
from fitz import Document, Page
from imagehash import phash
from PIL import Image
from tqdm import tqdm

from financeqa.preprocessing.data_models import DetectionResult


def extract_image_from_xref(pdf: Document, xref: int) -> Image.Image:
    """Extracts an image from a PDF file given an xref

    Args:
        pdf: the PDF file
        xref: the xref of the image to extract

    Returns:
        The extracted image
    """
    base_image = pdf.extract_image(xref)
    image_bytes = base_image["image"]
    image = Image.open(io.BytesIO(image_bytes))

    return image


def extract_image_header_from_bbox(page: Page, bbox: tuple[int, int, int, int], header_perc: float) -> Image.Image:
    """Extracts an image from a PDF page given an xref

    Args:
        page: the PDF page
        bbox: the bounding box of the image
        header_perc: the percentage of the image to consider as the header

    Returns:
        The extracted image
    """
    bbox_height = bbox[3] - bbox[1]
    extended_y0 = max(0, bbox[1] - header_perc * bbox_height)
    header_bbox = (bbox[0], extended_y0, bbox[2], bbox[1])  # extend header bbox above the image and cut off the bottom

    mat = pymupdf.Matrix(2.5, 2.5)
    pix = page.get_pixmap(clip=header_bbox, matrix=mat)  # type: ignore
    image = Image.open(io.BytesIO(pix.tobytes()))

    return image


def combine_images_horizontally_with_resize(images):
    """
    Combines a list of PIL Image objects horizontally, resizing all images to the smallest height.

    Args:
        images: List of PIL Image objects.

    Returns:
        A single PIL Image object with all input images combined horizontally.
    """
    # Find the smallest height among the images
    min_height = min(img.height for img in images)

    # Resize all images to have the same height (smallest height)
    resized_images = [img.resize((int(img.width * min_height / img.height), min_height)) for img in images]

    # Calculate the total width and use the smallest height
    total_width = sum(img.width for img in resized_images)

    # Create a new blank image with the calculated dimensions
    combined_image = Image.new("RGB", (total_width, min_height))

    # Paste resized images side by side
    x_offset = 0
    for img in resized_images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return combined_image


def combine_image_and_header(image: Image.Image, header: Image.Image) -> Image.Image:
    """Combines an image and a header image

    Args:
        image: the image
        header: the header image

    Returns:
        The combined image
    """
    if header.width != image.width:
        aspect_ratio = header.height / header.width
        new_width = image.width
        new_height = int(new_width * aspect_ratio)
        header = header.resize((new_width, new_height))

    combined_height = header.height + image.height
    combined_image = Image.new("RGB", (image.width, combined_height))

    combined_image.paste(header, (0, 0))
    combined_image.paste(image, (0, header.height))

    return combined_image


def extract_images_from_page(
    page: Page, negative_images: list[Image.Image], header_perc: float = 0.5
) -> list[DetectionResult]:
    """
    Extracts images from a PDF page

    Args:
        page: the PDF page
        negative_images: a list of negative images
        header_perc: the percentage of the image to extend above, considered as the header
    """
    result = []

    for img in page.get_images(full=True):
        xref = img[0]
        image = extract_image_from_xref(page.parent, xref)  # type: ignore

        hamming_distances = [phash(image) - phash(negative_image) for negative_image in negative_images]
        if any(hamming_distance < 10 for hamming_distance in hamming_distances):
            continue

        combined_image = image
        bbox, _ = page.get_image_rects(xref, transform=True)[0]  # type: ignore

        if header_perc > 0:
            image_header = extract_image_header_from_bbox(page, bbox, header_perc)
            combined_image = combine_image_and_header(image, image_header)

        result.append(DetectionResult(image=combined_image, bbox=bbox))

    return result


def extract_images_from_pdf(
    pdf: Document, negative_images: list[Image.Image], header_perc: float = 0.5
) -> dict[int, list[DetectionResult]]:
    """Extracts images from a PDF file

    Args:
        pdf: the PDF file
        negative_images: a list of negative images
        header_perc: the percentage of the image to extend above, considered as the header

    Returns:
        A dictionary mapping page numbers to a list of detection results
    """
    result = {}

    for page in pdf:
        images = extract_images_from_page(page, negative_images, header_perc)

        if len(images) == 0:
            continue

        result[page.number] = images

    return result


@click.command()
@click.option("--input_dir", default="./data/docs/pdf/")
@click.option("--output_dir", default="./data/docs/images/")
@click.option("--negative_images_dir", default="./data/negative_images/")
def main(input_dir: str, output_dir: str, negative_images_dir: str):
    docs_root = Path(input_dir)
    output_root = Path(output_dir)
    negative_images_root = Path(negative_images_dir)

    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    docs_paths = list(docs_root.glob("*.pdf"))
    for doc_path in tqdm(docs_paths):
        doc_name = doc_path.stem
        doc_output_dir = output_root / doc_name

        _, _, company = doc_name.split()

        negative_images = []
        company_negative_images_folder = negative_images_root / company.lower()
        for image in company_negative_images_folder.glob("*"):
            negative_images.append(Image.open(image))

        pdf = fitz.open(doc_path)
        images = extract_images_from_pdf(pdf, negative_images)

        if len(images) > 0:
            doc_output_dir.mkdir(parents=True, exist_ok=True)

        for page_num, page_images in images.items():
            combined_image = page_images[0].image

            if len(page_images) > 1:
                combined_image = combine_images_horizontally_with_resize([image.image for image in page_images])

            combined_image.save(doc_output_dir / f"{page_num}.png")


if __name__ == "__main__":
    main()
