# Breast Cancer Data Analysis

An automated Python-based analytics pipeline for breast cancer datasets. The project performs data preprocessing, exploratory data analysis, statistical analysis, machine learning, clustering, anomaly detection, and generates a comprehensive PDF report with visualizations.

## Features

- Automatic dataset discovery
- Supports CSV, Excel, SQLite, DB, and ZIP files
- Data cleaning and preprocessing
- Missing value analysis
- Statistical summary generation
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Machine learning classification
- Random Forest and Gradient Boosting models
- Feature importance analysis
- Risk score regression (if available)
- K-Means clustering
- Isolation Forest anomaly detection
- Automatic chart generation
- Comprehensive PDF analytics report
- Processed dataset export

## Project Structure

```
Breast-Cancer-Data-Analysis/
│
├── Output/
│   ├── charts/
│   ├── Breast_Cancer_Enhanced_Dataset_Analytics_Report.pdf
│   └── processed_breast_cancer_dataset.csv
│
├── breast_cancer_analysis.py
├── dataset.csv
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- ReportLab
- OpenPyXL

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

Place your dataset in the project directory.

Run:

```bash
python breast_cancer_analysis.py
```

The script automatically detects supported datasets and generates all outputs inside the `Output` directory.

## Supported Input Formats

- CSV
- XLSX
- XLS
- SQLite Database
- DB
- ZIP archives containing supported datasets

## Generated Outputs

- PDF analytics report
- Processed dataset
- Charts and visualizations
- Machine learning evaluation metrics
- Clustering analysis
- Anomaly detection summary

## Machine Learning Models

Classification

- Random Forest Classifier
- Gradient Boosting Classifier

Regression (if risk score exists)

- Random Forest Regressor
- Gradient Boosting Regressor

Additional Analytics

- K-Means Clustering
- Isolation Forest Anomaly Detection

## License

This project is licensed under the Apache License 2.0.

See the LICENSE file for details.

## Author

Developed for automated breast cancer data analysis and machine learning-based diagnostic analytics.
