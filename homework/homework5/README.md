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
