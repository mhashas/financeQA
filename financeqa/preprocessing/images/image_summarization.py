import asyncio
import base64
import io
import os
import re
from pathlib import Path
from typing import Callable

import click
import pandas as pd
from PIL import Image
from tqdm.asyncio import tqdm

from financeqa.constants import MessageType, OpenAIProvider
from financeqa.generate.openai_inference import OpenAIChatCompletion, build_openai_chat_completion


def process_image(
    image_path: Path,
    inference_client: OpenAIChatCompletion,
    message_generator: Callable[[str], list[dict[str, str]]],
    processed_images: set[str],
) -> str | None:
    """Process an image by resizing it and sending it to the OpenAI API for summarization

    Args:
        image_path: path to the image
        inference_client: inference client to use
        message_generator: function that generates the messages to send to the inference client
        processed_images: images that have already been processed

    Returns:
        the summary
    """
    if str(image_path) in processed_images:
        return None

    try:
        with open(str(image_path), "rb") as f:
            img = Image.open(f)
            width, height = img.size

            # Resize
            new_size = (1024, int(1024 * height / width)) if width > height else (int(1024 * width / height), 1024)
            img = img.resize(new_size)

            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        image = f"data:image/png;base64,{image}"
        messages = message_generator(image)

        async def run_get_completions():
            return await inference_client.get_completions(messages, model_name="gpt-4o", max_tokens=4096)

        summary = asyncio.run(run_get_completions())
        return summary
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "Error: Unable to process the image."


async def process_batch(
    batch: list[Path],
    inference_client: OpenAIChatCompletion,
    message_generator: Callable[[str], list[dict[str, str]]],
    processed_images: set[str],
) -> list[str | None]:
    """Process a batch of images

    Args:
        batch: list of images to process
        inference_client: inference client to use
        message_generator: function that generates the messages to send to the inference client
        processed_images: images that have already been processed

    Returns:
        list of the summaries
    """
    tasks = [
        asyncio.to_thread(process_image, image, inference_client, message_generator, processed_images)
        for image in batch
    ]

    results = await asyncio.gather(*tasks)
    return results


def get_message_generator(task_type: str) -> Callable[[str], list[dict[str, str]]]:
    """Return the message generator function for the given task type

    Args:
        task_type: the task type

    Returns:
        the message generator function
    """

    def build_table_messages_with_image(image: str) -> list[dict[str, str]]:
        prompt = """
            You are an expert assistant focused solely on extracting and summarizing tables from images. Identify any table in the image and summarize all the data visible in the table in HTML format. Double-check the data to ensure it is accurate.
            """
        messages = [
            {
                "role": MessageType.SYSTEM.value,
                "content": [{"type": "text", "text": prompt}],
            },
            {
                "role": MessageType.USER.value,
                "content": [
                    {"type": "text", "text": "Extract and summarize the table data from the image below."},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            },
        ]
        return messages

    def build_financial_figure_messages_with_image(image: str) -> list[dict[str, str]]:
        prompt = """
            You are an expert assistant focused on extracting and summarizing all the important data visible in the image. This includes any numbers, statistics, text, or other key information. 
            Ensure that the extracted data is accurate and present it in a clear, structured format. 
            Wrap the extracted data in <data> </data> tags. If there are any patterns or trends that can be identified, highlight them.
        """
        messages = [
            {
                "role": MessageType.SYSTEM.value,
                "content": [{"type": "text", "text": prompt}],
            },
            {
                "role": MessageType.USER.value,
                "content": [
                    {"type": "text", "text": "Please extract and summarize the key data from the image below."},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            },
        ]
        return messages

    message_fn_dict = {
        "table": build_table_messages_with_image,
        "image": build_financial_figure_messages_with_image,
    }

    callable_to_return = message_fn_dict.get(task_type)
    if callable_to_return is None:
        raise ValueError(f"Invalid task type: {task_type}")

    return callable_to_return


def postprocess(summary, tag: str) -> str:
    summary = summary.replace("\n", "")

    re_expression = r"(<" + tag + r".*?>.*?</" + tag + r">)"
    match = re.search(re_expression, summary, re.DOTALL)

    if match:
        # Return the trimmed string containing everything between the first <table> and the last </table>
        return match.group(0)
    else:
        # If no <table> tag is found, return an empty string or the original summary
        return "Error: No info found in the image."


@click.command()
@click.option("--input_dir", default="./data/images/")
@click.option("--csv_path", default="./data/image_summaries.csv")
@click.option("--task_type", type=click.Choice(["table", "image"]), default="image")
def main(input_dir: str, csv_path: str, task_type: str):
    chat_completion_client = build_openai_chat_completion(OpenAIProvider.OPENAI_AZURE)

    message_generator = get_message_generator(task_type)
    tag = "table" if task_type == "table" else "data"

    tables_root = Path(input_dir)
    tables_images = sorted(tables_root.glob("**/*.png"))

    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=["image_name", "summary"])
    processed_images = set(str(tables_root / image_name) for image_name in df["image_name"].tolist())

    batch_size = 2  # otherwise I get 429 rate limit errors
    for i in tqdm(range(0, len(tables_images), batch_size), desc="Processing Batches"):
        batch = tables_images[i : i + batch_size]
        results = asyncio.run(process_batch(batch, chat_completion_client, message_generator, processed_images))

        df_entries = [(b, postprocess(r, tag)) for b, r in zip(batch, results) if r is not None]
        new_entries = pd.DataFrame(df_entries, columns=["image_name", "summary"])
        new_entries["image_name"] = new_entries["image_name"].apply(lambda x: str(Path(x).relative_to(tables_root)))
        df = pd.concat([df, new_entries], ignore_index=True)

        df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
