# Homework 6 — Data Preprocessing

This folder contains my submission for **Stage 6: Data Preprocessing**.

## What’s included
- `src/cleaning.py`: modular cleaning functions with docstrings
- `notebooks/preprocessing.ipynb`: end-to-end preprocessing demo
- `data/raw/`: raw data directory (a small sample dataset is included for demo)
- `data/processed/`: cleaned output directory (the notebook writes here)

## Cleaning Strategy
1. **Drop high-missing features**: Remove columns whose missing ratio exceeds **0.6**. This preserves signal while avoiding columns dominated by gaps.
2. **Fill numeric gaps with median**: Median is robust to outliers and keeps distribution shape better than mean for skewed data.
3. **Min–Max normalization**: Scale numeric features to **[0,1]** for downstream models that are sensitive to feature scales.

