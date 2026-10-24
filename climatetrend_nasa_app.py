import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from pathlib import Path

st.set_page_config(page_title="ClimateTrend AI", page_icon="🌍", layout="wide")
st.title("🌍 ClimateTrend AI")
st.caption("Explore historical global temperature anomalies and a simple baseline trend projection.")
st.info("This is an educational baseline model, not an official NASA forecast or a physical climate model. Results depend on the data file supplied.")

DATA_FILE = Path(__file__).with_name("climatetrend_nasa_gistemp_v4_1880_2025.csv")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    required = {"year", "temp_anomaly_c"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain columns named year and temp_anomaly_c.")
    df = df[["year", "temp_anomaly_c"]].dropna().copy()
    df["year"] = df["year"].astype(int)
    df["temp_anomaly_c"] = pd.to_numeric(df["temp_anomaly_c"], errors="raise")
    return df.sort_values("year")

if not DATA_FILE.exists():
    st.error("Dataset file not found. Upload climatetrend_nasa_gistemp_v4_1880_2025.csv to the same GitHub repository as this Python file.")
    st.stop()

try:
    df = load_data(DATA_FILE)
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

first_year, last_year = int(df.year.min()), int(df.year.max())
c1, c2, c3 = st.columns(3)
c1.metric("First year in file", first_year)
c2.metric("Last year in file", last_year)
c3.metric("Annual records", len(df))

st.subheader("Historical temperature anomaly")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.year, y=df.temp_anomaly_c, mode="lines", name="Annual anomaly"))
fig.update_layout(xaxis_title="Year", yaxis_title="Temperature anomaly (°C)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Holdout validation")
st.write("The model is trained using records through 2020, then compared with available observations from 2021 onward. This is a simple test of trend extrapolation, not proof of predictive skill for future climate conditions.")
train = df[df.year <= 2020]
test = df[df.year > 2020]
if len(train) >= 2 and len(test) > 0:
    model = LinearRegression().fit(train[["year"]], train["temp_anomaly_c"])
    test_pred = model.predict(test[["year"]])
    mae = mean_absolute_error(test["temp_anomaly_c"], test_pred)
    rmse = np.sqrt(mean_squared_error(test["temp_anomaly_c"], test_pred))
    m1, m2 = st.columns(2)
    m1.metric("Holdout MAE (°C)", f"{mae:.3f}")
    m2.metric("Holdout RMSE (°C)", f"{rmse:.3f}")
    validation = pd.DataFrame({"Year": test.year, "Observed (°C)": test.temp_anomaly_c, "Model estimate (°C)": test_pred})
    st.dataframe(validation, use_container_width=True, hide_index=True)
else:
    st.warning("The dataset needs at least two training records through 2020 and at least one record after 2020 to calculate holdout validation.")

st.subheader("Baseline trend projection")
horizon = st.slider("How many years ahead?", min_value=1, max_value=30, value=10)
fit_model = LinearRegression().fit(df[["year"]], df["temp_anomaly_c"])
future_years = np.arange(last_year + 1, last_year + horizon + 1)
future_pred = fit_model.predict(pd.DataFrame({"year": future_years}))
forecast = pd.DataFrame({"Year": future_years, "Projected anomaly (°C)": future_pred})
forecast_fig = go.Figure()
forecast_fig.add_trace(go.Scatter(x=df.year, y=df.temp_anomaly_c, mode="lines", name="Historical data"))
forecast_fig.add_trace(go.Scatter(x=future_years, y=future_pred, mode="lines+markers", name="Linear trend projection", line=dict(dash="dash")))
forecast_fig.update_layout(xaxis_title="Year", yaxis_title="Temperature anomaly (°C)", hovermode="x unified")
st.plotly_chart(forecast_fig, use_container_width=True)
st.dataframe(forecast, use_container_width=True, hide_index=True)
st.caption("Source label in the project: NASA GISTEMP v4. Before public presentation, verify the CSV against the latest official NASA download and document the anomaly baseline/units. A straight-line regression does not account for emissions pathways, natural variability, or climate-system physics.")
with st.expander("View dataset"):
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="climatetrend_data.csv", mime="text/csv")
