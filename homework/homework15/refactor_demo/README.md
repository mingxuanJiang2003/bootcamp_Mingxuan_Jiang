Usage
python refactor_demo/run_step.py ingest --input data/raw_prices.csv --out data/ingested.parquet
python refactor_demo/run_step.py clean --input data/ingested.parquet --out data/clean_prices.csv
python refactor_demo/run_step.py feature --input data/clean_prices.csv --out data/features.csv
Logs are written to refactor_demo/logs
Outputs are idempotent via checkpoints/*.hash
