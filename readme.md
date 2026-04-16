# Customer Churn Prediction (ANN)

This project implements a Customer Churn Prediction application using an Artificial Neural Network (ANN) and Streamlit. It predicts whether a customer is likely to stay or leave based on their demographics and financial data.

## Features
- **Modern Python**: Implemented with Python 3.9+ type hints and `python-pro` patterns.
- **Robust Loading**: Graceful error handling for missing or corrupted model artifacts.
- **Interactive UI**: Sleek Streamlit interface with real-time predictions and probability metrics.

## Project Structure
- `app.py`: Main Streamlit application.
- `artifacts/`: (Required) Directory containing trained models and scalers.
    - `churn_ann_model.keras`: Trained TensorFlow model.
    - `scaler.pkl`: Pickled scikit-learn scaler.
    - `feature_columns.json`: List of feature names in correct order.
- `requirements.txt`: Project dependencies.
- `pyproject.toml`: Tooling configuration (Ruff, Pyright).

## Setup instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Artifacts**:
   Ensure your trained model artifacts are placed in the `artifacts/` directory.

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Development
This project uses **Ruff** for linting and formatting, and **Pyright** for type checking. Configuration can be found in `pyproject.toml`.
