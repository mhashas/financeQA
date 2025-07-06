# allow script to run from any location, cd into scripts folder
cd `dirname "$0"`
cd ../

PDF_DIR="${1:-data/docs/pdf/}"
OUTPUT_DIR="${2:-data/images/}"
NEGATIVE_IMAGES_DIR="${3:-data/negative_images/}"

python3 financeqa/preprocessing/images/image_extraction.py \
    --input_dir "$PDF_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --negative_images_dir "$NEGATIVE_IMAGES_DIR" \
