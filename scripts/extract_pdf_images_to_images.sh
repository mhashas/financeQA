# allow script to run from any location, cd into scripts folder
cd `dirname "$0"`
cd ../

python3 financeqa/preprocessing/images/image_extraction.py \
    --input_dir data/docs/pdf/ \
    --output_dir data/images/ \
    --negative_images_dir data/negative_images/ \
