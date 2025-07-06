# allow script to run from any location, cd into scripts folder
cd `dirname "$0"`
cd ../

PDF_DIR="${1:-data/docs/pdf/}"
OUTPUT_DIR="${2:-data/tables/}"

python3 financeqa/preprocessing/tables/table_extraction_to_image.py \
    --input_dir "$PDF_DIR" \
    --output_dir "$OUTPUT_DIR" \
