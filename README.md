# FinanceQA

FinanceQA is an advanced Retrieval-Augmented Generation (RAG) system designed for financial document (i.e. 10Q fillings) question answering and analysis. It leverages state-of-the-art document processing, information extraction, and large language models to enable users to query and compare financial data from complex sources such as reports, tables, and images. The system supports end-to-end workflows including data extraction, summarization, database population, and interactive querying via a REST API.

For a high-level overview of the system architecture and data flow, please refer to the following diagram:

![System Architecture](diagrams/great_diagram.png)

# Installation

## Pre-requisites

* [Docker installation](https://docs.docker.com/get-started/get-docker/)
* [Docker compose installation](https://docs.docker.com/compose/install/)

## Env

1. Copy .env.template to .env and add your own keys API keys. Also change the chroma DB path
2. Create the db

```bash
    docker compose up
```

3. Create a python env

    ```bash
    python3 -m venv env
    ```

4. Activate the env

```bash
    source env/bin/activate
```

5. Install the requirements

```bash
    pip install -r requirements.txt
```

6. Optionally test that the extensive 😊 test suite runs

```bash
    pytest tests
```

# Data Extraction and Preprocessing

To prepare your data for querying, run the following scripts in order. This will extract images and tables from your PDFs, summarize them, and populate the database.

**Note:** All scripts accept optional command-line arguments. If not provided, they will use the default paths shown below.

1. **Extract images from PDFs:**

   ```bash
   bash scripts/extract_pdf_images_to_images.sh [PDF_DIR] [OUTPUT_DIR] [NEGATIVE_IMAGES_DIR]
   ```

   * Default: `PDF_DIR=data/docs/pdf/`, `OUTPUT_DIR=data/images/`, `NEGATIVE_IMAGES_DIR=data/negative_images/`
   * Extracts images from all PDFs in the specified directory and saves them to the output directory.
   * The negative images directory is used to filter out unwanted images.

2. **Extract tables from PDFs as images:**

   ```bash
   bash scripts/extract_pdf_tables_to_images.sh [PDF_DIR] [OUTPUT_DIR]
   ```

   * Default: `PDF_DIR=data/docs/pdf/`, `OUTPUT_DIR=data/tables/`
   * Detects and saves tables from all PDFs in the specified directory as images to the output directory.

3. **Summarize extracted images:**

   ```bash
   bash scripts/summarize_extracted_images_to_df.sh [IMAGES_DIR] [OUTPUT_CSV]
   ```

   * Default: `IMAGES_DIR=data/images/`, `OUTPUT_CSV=data/images_summaries.csv`
   * Summarizes each image in the specified directory and saves results to the CSV file.

4. **Summarize extracted table images:**

   ```bash
   bash scripts/summarize_extracted_tables_to_df.sh [TABLES_DIR] [OUTPUT_CSV]
   ```

   * Default: `TABLES_DIR=data/tables/`, `OUTPUT_CSV=data/tables_summaries.csv`
   * Summarizes each table image in the specified directory and saves results to the CSV file.

5. **Populate the database:**

   ```bash
   bash scripts/populate_db.sh [INPUT_DIR] [IMAGES_CSV_PATH] [TABLES_CSV_PATH]
   ```

   * Default: `INPUT_DIR=data/docs/pdf/`, `IMAGES_CSV_PATH=data/images_summaries.csv`, `TABLES_CSV_PATH=data/tables_summaries.csv`
   * Loads all extracted and summarized data into the vector database.

**Notes:**
* The CSV summary files must have the following columns: `image_name`, `summary`.
* For tables, the `summary` should be an HTML table (`<table>...</table>`). For images, the summary is wrapped in `<data>...</data>` tags.
* The scripts can be run from any location and will automatically navigate to the correct directory.
* All scripts use `python3` for compatibility across different systems.

# Running the App

8. Run the app, from the root of the repo:

```bash
    uvicorn financeqa.app.main:app --host 0.0.0.0 --port 8010
```

9. Test the app

```bash
curl --location 'localhost:8010/query/' \
--header 'Content-Type: application/json' \
--header 'Authorization: secret-key' \
--data '[{
    "message": "compare the revenue of apple with that of microsoft in q3 2022"
}]'
```
