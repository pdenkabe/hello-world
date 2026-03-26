# Project: Wine Quality Analysis

## Overview
Data analysis and regression modeling project using the UCI Wine Quality dataset (id=186).
Red and white Portuguese "Vinho Verde" wines — 4,898 samples, 11 chemical features, quality score (0–10) as target.

## Environment
- Python 3.11 (Windows)
- Key packages: `ucimlrepo`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

## Data Access
```python
from ucimlrepo import fetch_ucirepo
wine = fetch_ucirepo(id=186)
X = wine.data.features   # 11 chemical features
y = wine.data.targets.squeeze()  # quality score
```

## Project Files
- `wine_quality_analysis.py` — EDA, distributions, correlation heatmap, 3 baseline models
- `wine_model_selection.py` — 42-combo model search (7 feature sets × 6 models)

## Key Findings
- Best model: **Random Forest, all 11 features** — CV R²=0.506, RMSE=0.609
- Linear models plateau at R²≈0.26 (relationships are nonlinear)
- Top features by importance: `alcohol`, `volatile_acidity`, `density`, `sulphates`
- Dropping collinear `density` made no meaningful difference

## Conventions
- Save all plots as `.png` in the project root
- Use `random_state=42` and `test_size=0.2` for reproducibility
- Rank models by 5-fold CV R², not test-set R² alone
- Scale features for linear models; tree models use raw values
