# ITM-390 AI-Based Threat Detection Tool

This project contains:
- A machine learning pipeline for threat detection (normal vs malicious traffic)
- A Streamlit web app for real-time predictions using saved model artifacts

## Reorganized Project Structure

```text
Project/
├── app.py
├── README.md
├── artifacts/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── notebooks/
│   └── threat_detection.ipynb
├── output/
│   ├── class_distribution.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── confusion_matrix_best.png
│   └── (additional confusion matrices)
└── scripts/
    ├── qa_crosscheck.py
    └── threat_detection.py
```

## Environment Setup

These steps are portable and work for anyone cloning this repository.

1. Clone the repo and enter the project folder.
2. (Recommended) Create and activate a virtual environment.
3. Install dependencies.

### 1) Clone and enter project

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2) Optional virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install kagglehub scikit-learn pandas numpy matplotlib seaborn imbalanced-learn joblib streamlit
```

If `python` is not recognized on your machine, use `python3` instead.

## Run the Training Pipeline

This will download data, train/tune models, regenerate plots, and save artifacts to `artifacts/`.

```bash
python scripts/threat_detection.py
```

## Run the Streamlit App

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

## Run the QA Crosscheck

Use this after training or app changes to verify the project end-to-end:

```bash
python scripts/qa_crosscheck.py
```

## Notes

- `app.py` loads model files from `artifacts/`.
- `scripts/threat_detection.py` writes output plots to `output/` and artifacts to `artifacts/`.
- `scripts/qa_crosscheck.py` validates file integrity, model/scaler consistency, image outputs, app syntax, dependencies, and end-to-end predictions.

## Troubleshooting

- If kaggle download fails, make sure your Kaggle credentials are configured on your machine.
- If dependency install fails, confirm your virtual environment is activated before running commands.
- If Streamlit does not open automatically, copy the local URL from terminal output into your browser.
