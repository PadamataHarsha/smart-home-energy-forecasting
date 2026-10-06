import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from energy_forecast import (
    FORECAST_STEPS,
    WINDOW_SIZE,
    evaluate_model,
    forecast_next,
    get_dataset,
    load_or_train,
    preprocess_data,
)

st.set_page_config(page_title="Smart Home Energy Forecasting", page_icon="🏠", layout="wide")
st.title("🏠 Smart Home Energy Forecasting")
st.caption("Automatic dataset download with Kaggle-first fallback.")

with st.sidebar:
    st.header("Model settings")
    max_points = st.slider("Recent 15-minute readings", 2_000, 30_000, 15_000, 1_000)
    epochs = st.slider("Training epochs", 1, 10, 3, 1)
    retrain = st.button("Retrain Model", type="secondary")

try:
    with st.status("Preparing dataset and model...", expanded=True) as status:
        dataset_path, source = get_dataset()
        history = preprocess_data(str(dataset_path), max_points=max_points)
        model, x_scaler, y_scaler, trained = load_or_train(
            history, epochs=epochs, force_retrain=retrain
        )
        status.update(label="Forecast ready", state="complete", expanded=False)
except Exception as exc:
    st.error(f"Application could not start: {exc}")
    st.stop()

predictions = forecast_next(model, x_scaler, y_scaler, history)
metrics = evaluate_model(model, x_scaler, y_scaler, history)
current = float(history.iloc[-1])

latest_timestamp = history.index.max()
day_start = latest_timestamp - pd.Timedelta(days=1)
daily_history = history.loc[history.index >= day_start]

daily_avg = float(daily_history.mean())
daily_peak = float(daily_history.max())

threshold = max(daily_avg * 1.5, 2.5)

with st.sidebar:
    st.success("Dataset ready")
    st.write(f"**Source:** {source}")
    st.write(f"**Records:** {len(history):,}")
    st.write("**Sampling:** 15 minutes")
    st.write(f"**Input window:** Previous {WINDOW_SIZE} readings")
    st.write(f"**Forecast horizon:** Next {FORECAST_STEPS * 15} minutes")
    st.write(f"**Model:** {'trained this run' if trained else 'loaded from cache'}")

st.subheader("Current and predicted consumption")
forecast_cols = st.columns(5)
forecast_cols[0].metric("Current consumption", f"{current:.2f} kW")
for index, value in enumerate(predictions, start=1):
    forecast_cols[index].metric(
        f"Next {index * 15} min",
        f"{value:.2f} kW",
        f"{value - current:+.2f} kW",
    )

summary_cols = st.columns(4)
summary_cols[0].metric("Daily average", f"{daily_avg:.2f} kW")
summary_cols[1].metric("Peak consumption", f"{daily_peak:.2f} kW")
summary_cols[2].metric("MAE", f"{metrics['mae']:.3f} kW")
summary_cols[3].metric("RMSE", f"{metrics['rmse']:.3f} kW")

future_index = pd.date_range(
    history.index[-1] + pd.Timedelta(minutes=15),
    periods=FORECAST_STEPS,
    freq="15min",
)
fig = go.Figure()
fig.add_trace(go.Scatter(x=history.tail(WINDOW_SIZE).index, y=history.tail(WINDOW_SIZE).to_numpy(), name="Historical", mode="lines"))
fig.add_trace(go.Scatter(x=future_index, y=predictions, name="LSTM forecast", mode="lines+markers"))
fig.update_layout(
    title="Historical and forecast consumption",
    xaxis_title="Time",
    yaxis_title="Global active power (kW)",
    height=430,
)
st.plotly_chart(fig, use_container_width=True)

if max(predictions) >= threshold:
    st.warning(f"High-consumption warning: forecast reaches {max(predictions):.2f} kW.")
else:
    st.success("Forecast consumption is within the normal recent range.")

st.subheader("Energy-saving recommendations")
recommendations = []
if max(predictions) >= threshold:
    recommendations.extend([
        "Delay washing machines, water heaters, or other heavy appliances until the forecast drops.",
        "Avoid running several high-power appliances at the same time.",
    ])
if predictions[-1] > current:
    recommendations.append("Consumption is trending upward. Check heating, cooling, and standby devices.")
recommendations.append("Compare predictions with actual readings regularly and retrain when usage patterns change.")
for item in recommendations:
    st.write(f"• {item}")

with st.expander("Model evaluation"):
    st.json(metrics)
