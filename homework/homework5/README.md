# Stage 05 — Data Storage

## Folder structure
```
data/
  raw/         # immutable raw inputs
  processed/   # clean/validated data ready for modeling
notebooks/
```
## Formats
- CSV for human-readable raw drops
- Parquet for efficient, typed, columnar storage in `data/processed/`

## Environment-driven paths
Set in `.env`:
```
DATA_DIR_RAW=data/raw
DATA_DIR_PROCESSED=data/processed
```
The code reads these vars and writes/reads accordingly.

## How to run
1. Create/activate a virtual environment.
2. `pip install -r requirements.txt`
3. Open the notebook in `notebooks/`, run all cells.
4. Two files will be created:
   - `data/raw/sample_20250816-1806.csv`
   - `data/processed/sample_20250816-1806.parquet`
