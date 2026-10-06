import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

WINDOW_SIZE = 96
FORECAST_STEPS = 4
KAGGLE_DATASET_HANDLE = os.getenv(
    "KAGGLE_DATASET_HANDLE", "uciml/electric-power-consumption-data-set"
)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
DATA_DIR = Path(os.getenv("ENERGY_DATA_DIR", "/tmp/smart-home-energy-data"))
MODEL_DIR = Path(os.getenv("ENERGY_MODEL_DIR", "models"))


def _find_data_file(folder: Path) -> Path:
    candidates = list(folder.rglob("household_power_consumption.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"No household power consumption TXT file found under {folder}"
        )
    return max(candidates, key=lambda path: path.stat().st_size)


def _download_kaggle() -> Path:
    os.environ.setdefault("KAGGLEHUB_CACHE", str(DATA_DIR / "kaggle-cache"))
    import kagglehub

    downloaded = Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))
    return _find_data_file(downloaded)


def _download_uci() -> Path:
    target = DATA_DIR / "household_power_consumption.txt"
    if target.exists() and target.stat().st_size > 0:
        return target

    zip_path = Path(tempfile.gettempdir()) / "household_power_consumption.zip"
    try:
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            members = [
                name
                for name in archive.namelist()
                if Path(name).name == "household_power_consumption.txt"
            ]
            if not members:
                raise FileNotFoundError("Expected TXT file is absent from the UCI archive")
            with archive.open(members[0]) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    finally:
        zip_path.unlink(missing_ok=True)
    return target


def get_dataset() -> tuple[Path, str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return _download_kaggle(), "Kaggle"
    except Exception as kaggle_error:
        try:
            return _download_uci(), "UCI fallback"
        except Exception as uci_error:
            raise RuntimeError(
                "Kaggle failed and the UCI fallback also failed. "
                "Check internet access and Kaggle credentials."
            ) from uci_error


def preprocess_data(path: str, max_points: int = 15_000) -> pd.Series:
    path_obj = Path(path)
    df = pd.read_csv(
        path_obj,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        na_values=["?", ""],
    )
    timestamps = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )
    power = pd.to_numeric(df["Global_active_power"], errors="coerce")
    series = pd.Series(
        power.to_numpy(dtype="float32"),
        index=timestamps,
        name="Global_active_power",
    )
    series = series[~series.index.isna()].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series.resample("15min").mean().interpolate(limit=8).dropna().astype("float32")
    if max_points:
        series = series.tail(max_points)
    minimum_points = WINDOW_SIZE + FORECAST_STEPS + 100
    if len(series) < minimum_points:
        raise ValueError(
            f"Only {len(series)} usable readings; at least {minimum_points} are required"
        )
    return series


def _sequences(values: np.ndarray):
    x_values, y_values = [], []
    for index in range(WINDOW_SIZE, len(values) - FORECAST_STEPS + 1):
        x_values.append(values[index - WINDOW_SIZE : index])
        y_values.append(values[index : index + FORECAST_STEPS])
    return (
        np.asarray(x_values, dtype="float32")[..., None],
        np.asarray(y_values, dtype="float32"),
    )


def _artifact_paths():
    return (
        MODEL_DIR / "energy_lstm.keras",
        MODEL_DIR / "x_scaler.joblib",
        MODEL_DIR / "y_scaler.joblib",
        MODEL_DIR / "metadata.json",
    )


def load_or_train(series: pd.Series, epochs: int = 3, force_retrain: bool = False):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path, x_scaler_path, y_scaler_path, metadata_path = _artifact_paths()
    if (
        not force_retrain
        and all(path.exists() for path in (model_path, x_scaler_path, y_scaler_path, metadata_path))
    ):
        try:
            metadata = json.loads(metadata_path.read_text())
            if (
                metadata.get("window_size") == WINDOW_SIZE
                and metadata.get("forecast_steps") == FORECAST_STEPS
            ):
                return (
                    keras.models.load_model(model_path),
                    joblib.load(x_scaler_path),
                    joblib.load(y_scaler_path),
                    False,
                )
        except Exception:
            pass

    raw = series.to_numpy(dtype="float32").reshape(-1, 1)
    x_scaler = MinMaxScaler()
    scaled = x_scaler.fit_transform(raw).astype("float32").ravel()
    x_data, y_data = _sequences(scaled)
    split_index = int(len(x_data) * 0.8)
    x_train, y_train = x_data[:split_index], y_data[:split_index]
    x_val, y_val = x_data[split_index:], y_data[split_index:]

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(WINDOW_SIZE, 1)),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(FORECAST_STEPS),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    callbacks = [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=64,
        verbose=0,
        callbacks=callbacks,
    )

    y_scaler = MinMaxScaler().fit(raw)
    model.save(model_path)
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)
    metadata_path.write_text(
        json.dumps({"window_size": WINDOW_SIZE, "forecast_steps": FORECAST_STEPS})
    )
    return model, x_scaler, y_scaler, True


def forecast_next(model, x_scaler, y_scaler, series: pd.Series):
    recent = series.iloc[-WINDOW_SIZE:].to_numpy(dtype="float32").reshape(-1, 1)
    x_input = x_scaler.transform(recent).reshape(1, WINDOW_SIZE, 1)
    scaled_prediction = model.predict(x_input, verbose=0).reshape(-1, 1)
    return y_scaler.inverse_transform(scaled_prediction).ravel()


def evaluate_model(model, x_scaler, y_scaler, series: pd.Series):
    values = series.tail(max(500, WINDOW_SIZE + 100)).to_numpy(dtype="float32").reshape(-1, 1)
    scaled = x_scaler.transform(values).ravel()
    x_data, y_scaled = _sequences(scaled)
    if len(x_data) > 300:
        x_data, y_scaled = x_data[-300:], y_scaled[-300:]
    predicted_scaled = model.predict(x_data, verbose=0)
    actual = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    predicted = y_scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).ravel()
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "evaluation_points": int(len(actual)),
    }
