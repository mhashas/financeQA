# allow script to run from any location, cd into scripts folder
cd `dirname "$0"`
cd ../

python3 financeqa/preprocessing/tables/table_extraction_to_image.py \
    --input_dir data/docs/pdf/ \
    --output_dir data/tables/ \
