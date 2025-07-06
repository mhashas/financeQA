# allow script to run from any location, cd into scripts folder
cd `dirname "$0"`
cd ../

IMAGES_DIR="${1:-data/images/}"
OUTPUT_CSV="${2:-data/images_summaries.csv}"

python3 financeqa/preprocessing/images/image_summarization.py \
    --input_dir "$IMAGES_DIR" \
    --csv_path "$OUTPUT_CSV" \
    --task_type image
