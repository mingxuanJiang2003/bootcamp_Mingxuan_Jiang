@echo off
SETLOCAL
IF NOT EXIST .venv (
  py -m venv .venv
)
CALL .\.venv\Scripts\activate
pip install -r requirements.txt
python src\forecast_baseline.py --csv data\synthetic_timeseries.csv --out outputs
python src\classify_baseline.py --csv data\synthetic_timeseries.csv --out outputs
ENDLOCAL
