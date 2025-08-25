@echo off
REM One-click setup and run (Windows)
SETLOCAL
IF NOT EXIST .venv (
  py -m venv .venv
)
CALL .\.venv\Scripts\activate
pip install -r requirements.txt

echo Running baseline training...
python src\train_linear.py --csv data\synthetic_regression.csv --target y --out outputs

echo Running diagnostics...
python src\diagnostics.py --csv data\synthetic_regression.csv --target y --out outputs

echo Done. Open notebooks\modeling_regression_team.ipynb in VSCode to explore interactively.
ENDLOCAL
