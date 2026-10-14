# AutoML Project - Automated Machine Learning Pipeline

A comprehensive machine learning automation tool that handles preprocessing, feature selection, model training, and report generation.

## Features

✨ **Complete ML Pipeline**
- Automated data preprocessing (missing values, duplicates, encoding)
- Two-step feature selection:
  - **Step 1:** Mutual Information (MI) for feature importance
  - **Step 2:** Correlation analysis to drop redundant features
- Automatic problem type detection (Classification/Regression)
- Multiple model training and evaluation
- Comprehensive reporting (HTML, PDF, JSON, CSV)

🔍 **Smart Feature Selection**
- Identifies and removes low-information features
- Detects and removes text/ID features automatically
- Drops highly correlated features (default threshold: 0.90)
- Justification for each decision provided

📊 **Multiple Output Formats**
- Interactive HTML reports
- PDF reports with full metrics
- JSON metrics for programmatic access
- CSV results for further analysis

## Installation

```bash
cd AutoML_Project
pip install -r requirements.txt
```

## Quick Start

```bash
# Basic usage
python src/main.py --input data.csv --target target_column

# With custom thresholds
python src/main.py -i data.csv -t price --mi-threshold 0.05 --correlation 0.85

# Specify output formats
python src/main.py -i data.csv -t target --formats json csv html pdf
```

## Command Line Arguments

```
-i, --input          Path to input CSV file (required)
-t, --target         Target variable column name (required)
--mi-threshold       MI threshold for feature selection (default: 0.01)
--correlation        Correlation threshold for dropping features (default: 0.90)
--output            Output directory for reports (default: output)
--formats           Output formats: json, csv, html, pdf (default: json csv html)
```

## Project Structure

```
AutoML_Project/
├── src/
│   ├── modules/
│   │   ├── preprocessing.py       # Data cleaning and preprocessing
│   │   ├── feature_selection.py   # MI + correlation feature selection
│   │   ├── model_training.py      # Model training and evaluation
│   │   ├── reporting.py           # Report generation
│   │   ├── utils.py               # Utility functions
│   │   └── __init__.py
│   ├── config.py                  # Configuration settings
│   └── main.py                    # CLI interface
├── data/                          # Input data directory
├── output/                        # Generated reports and metrics
│   ├── reports/                   # HTML/PDF reports
│   └── metrics/                   # JSON/CSV metrics
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Pipeline Phases

### Phase 1: Data Loading & Preprocessing
- Loads CSV data
- Handles missing values (mean for numeric, 'UNKNOWN' for categorical)
- Removes duplicates
- One-hot encodes low-cardinality categorical features
- Label encodes high-cardinality categorical features
- Standardizes numeric features

### Phase 2: Feature Selection
- **Step 1 (MI):** Calculates mutual information scores
  - Removes features below MI threshold
  - Removes text features (name, title, description, etc.)
  - Removes ID features
- **Step 2 (Correlation):** Drops highly correlated features
  - Identifies feature pairs with correlation > 0.90
  - Keeps feature with higher MI score

### Phase 3: Model Training & Evaluation

**Classification Models:**
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

**Regression Models:**
- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting

**Metrics:**
- Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Regression: R² Score, MAE, MSE, RMSE
- Cross-validation: 5-fold CV for model robustness

### Phase 4: Report Generation
- Model performance comparison
- Best model identification
- Detailed metrics breakdown
- Multiple output formats

## Example Output

```
================================================================================
                 AutoML Project - Automated Machine Learning Pipeline
================================================================================

...

================================================================================
                              COMPLETION SUMMARY
================================================================================

╔═══════════════════════════════════════════════════════════╗
║                   AutoML PROJECT SUMMARY                  ║
╚═══════════════════════════════════════════════════════════╝

BEST MODEL: Gradient Boosting
─────────────────────────────────────────────────────────────
  F1-Score: 0.9245
  Accuracy: 0.9312
  Precision: 0.9267
  Recall: 0.9234
─────────────────────────────────────────────────────────────
```

## Configuration

Edit `src/config.py` to adjust:
- Feature selection thresholds
- Test/train split ratio
- Number of cross-validation folds
- Model hyperparameters
- Text/ID feature keywords

## Advanced Usage

### Custom Feature Thresholds

```bash
# Lower MI threshold to keep more features
python src/main.py -i data.csv -t target --mi-threshold 0.001

# Higher correlation threshold to keep more correlated features
python src/main.py -i data.csv -t target --correlation 0.95
```

### Output Directory

```bash
# Specify custom output directory
python src/main.py -i data.csv -t target --output my_results/
```

### Selective Report Formats

```bash
# Only JSON and CSV
python src/main.py -i data.csv -t target --formats json csv

# With PDF
python src/main.py -i data.csv -t target --formats html pdf
```

## Output Files

### Reports Directory (`output/reports/`)
- `report_YYYYMMDD_HHMMSS.html` - Interactive HTML report
- `report_YYYYMMDD_HHMMSS.pdf` - PDF report

### Metrics Directory (`output/metrics/`)
- `report_YYYYMMDD_HHMMSS.json` - Machine-readable metrics
- `results_YYYYMMDD_HHMMSS.csv` - CSV formatted results

## Troubleshooting

**Missing values warning**
- The tool automatically handles missing values. Features with >50% missing are dropped.

**All features removed**
- Increase MI threshold or run with lower threshold: `--mi-threshold 0.001`

**PDF generation fails**
- Install weasyprint: `pip install weasyprint`
- On Windows, may require additional dependencies

**Memory issues with large datasets**
- Reduce dataset size or run in chunks
- Check available RAM

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn
- plotly
- weasyprint (for PDF)

## Contributing

To extend or modify the pipeline:

1. **Add preprocessing steps**: Edit `src/modules/preprocessing.py`
2. **Modify feature selection**: Edit `src/modules/feature_selection.py`
3. **Add models**: Edit `src/modules/model_training.py`
4. **Customize reports**: Edit `src/modules/reporting.py`

## License

Open source project for educational purposes.

## Support

For issues or questions, check the output reports in `output/` directory for detailed metrics and logs.
