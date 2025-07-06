# allow script to run from any location, cd into scripts folder
cd `dirname "$0"`
cd ../

TABLES_DIR="${1:-data/tables/}"
OUTPUT_CSV="${2:-data/tables_summaries.csv}"

python3 financeqa/preprocessing/images/image_summarization.py \
    --input_dir "$TABLES_DIR" \
    --csv_path "$OUTPUT_CSV" \
    --task_type table
