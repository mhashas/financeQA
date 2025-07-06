# allow script to run from any location, cd into scripts folder
cd `dirname "$0"`
cd ../

python3 financeqa/preprocessing/images/image_summarization.py \
    --input_dir data/images/ \
    --csv_path data/images_summaries.csv \
    --task_type image
