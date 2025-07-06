# allow script to run from any location, cd into scripts folder
cd `dirname "$0"`
cd ../

python3 financeqa/indexing/populate_db.py \
    --input_dir data/docs/pdf/ \
    --images_csv_path data/images_summaries.csv \
    --tables_csv_path data/tables_summaries.csv \
