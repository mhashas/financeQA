# allow script to run from any location, cd into scripts folder
cd `dirname "$0"`
cd ../

INPUT_DIR="${1:-data/docs/pdf/}"
IMAGES_CSV_PATH="${2:-data/images_summaries.csv}"
TABLES_CSV_PATH="${3:-data/tables_summaries.csv}"

python3 financeqa/indexing/populate_db.py \
    --input_dir "$INPUT_DIR" \
    --images_csv_path "$IMAGES_CSV_PATH" \
    --tables_csv_path "$TABLES_CSV_PATH" \
