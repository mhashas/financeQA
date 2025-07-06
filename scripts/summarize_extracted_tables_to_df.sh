# allow script to run from any location, cd into scripts folder
cd `dirname "$0"`
cd ../

python3 financeqa/preprocessing/images/image_summarization.py \
    --input_dir data/tables/ \
    --csv_path data/tables_summaries.csv \
    --task_type table
