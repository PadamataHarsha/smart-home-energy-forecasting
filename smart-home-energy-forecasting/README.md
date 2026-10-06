# Smart Home Energy Forecasting

Streamlit application for forecasting household active power using a small LSTM model.

## Run

```bash
streamlit run practice/app.py --server.address 0.0.0.0 --server.port 8501
```

The dataset is downloaded automatically from Kaggle with an official UCI fallback. The downloaded data and model artifacts are stored outside the repository when needed.

## Project structure

```text
practice/
├── __init__.py
├── app.py
└── energy_forecast.py
models/
requirements.txt
.env.example
.gitignore
README.md
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir -r requirements.txt
```

Dataset sources can be configured with `KAGGLE_DATASET_HANDLE`, `ENERGY_DATA_DIR`, and `ENERGY_MODEL_DIR`.
