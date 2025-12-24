# Titanic Survival Analysis (EDA)

## Goal
Explore which factors are associated with passenger survival on the Titanic using basic EDA, aggregation, and simple visualizations.

## Dataset
Titanic dataset (CSV): loaded directly from a public GitHub raw link.

## Tools
- Python
- Pandas
- Matplotlib

## What is done
1. Quick EDA: dataset shape, columns, missing values
2. Key questions:
   - Survival rate by sex
   - Survival rate by ticket class (Pclass)
   - Age comparison (means + distributions)
   - Fare comparison (median)
3. Visualizations:
   - Bar charts for survival rates
   - Histograms for age distributions

## Key findings (short)
- Women survived much more often than men.
- 1st class passengers had significantly higher survival rates than 3rd class.
- Mean age differs slightly, but distributions may differ (children effect is visible on histograms).
- Survivors tend to have higher ticket fares (median fare is higher).

## How to run
```bash
pip install -r requirements.txt
python analysis.py
