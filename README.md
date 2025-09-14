# Time Series Forecasting of Air Pollutants (NO₂ & CO) Using GRU and ARIMA

## Project Overview
This project focuses on forecasting daily concentrations of air pollutants, specifically **NO₂** and **CO**, using historical air quality sensor data.  
The workflow demonstrates handling noisy and non-stationary time series data, feature engineering, deep learning with **GRU**, and classical time series forecasting with **ARIMA**.  

- NO₂ was **differenced** to stabilize trends before GRU modeling.  
- CO was **stationary**, so no differencing was applied.  
- Daily resampling was applied to ensure consistent time steps.

---

## Dataset
- Sensor readings include: `CO(GT)`, `NO₂(GT)`, `NOx(GT)`, `O₃`, `T`, `RH`, `AH`, and others.  
- Engineered features: lag variables (`CO_lag1`, `NO2_lag1`, etc.), temporal variables (`day`, `month`, `day_of_week`, `week_of_year`).  
- Target columns:
  - NO₂: `NO2(GT)_diff` for GRU (differenced series)
  - CO: `CO(GT)` (stationary)

---

## Workflow / Steps

1. **Data Preprocessing**
   - Resampled data to daily frequency.
   - Handled missing values and created lag features.
   - Created temporal features: `day`, `month`, `day_of_week`, `week_of_year`.
   - Differenced NO₂ for stationarity: `NO2(GT)_diff = NO2(GT).diff()`.
   - Scaled features using `MinMaxScaler`.

2. **GRU Modeling**
   - **NO₂ Prediction**
     - Trained GRU to predict **differenced NO₂ values** (`NO2(GT)_diff`).
     - Inverted differencing after prediction to recover actual NO₂ values.
   - **CO Prediction**
     - Trained GRU on raw CO values (`CO(GT)`) as the series was stationary.

3. **ARIMA Modeling**
   - Built ARIMA models on resampled daily NO₂ and CO series for comparison.

4. **Prediction & Evaluation**
   - GRU predictions aligned with test set.
   - Inverted differencing for NO₂ predictions.
   - Evaluated using **MSE, RMSE, MAE, R², MAPE**.

---

## Model Performance

### GRU Model (NO₂ with differencing)
| Metric | Value |
|--------|-------|
| MAE    | 11.27 |
| RMSE   | 14.07 |
| R²     | 0.62  |
> GRU predicts differenced NO₂ (`NO2(GT)_diff`) which is inverted to obtain actual NO₂ values.

### GRU Model (CO, stationary)
| Metric | Value |
|--------|-------|
| MSE    | 0.2131 |
| RMSE   | 0.4616 |
| MAE    | 0.3587 |
| R²     | 0.5419 |

### ARIMA Model
| Target | MSE   | RMSE  | MAE   | R²     |
|--------|-------|-------|-------|--------|
| NO₂    | 598.24| 24.46 | 19.35 | -0.02  |
| CO     | ...   | ...   | ...   | ...    |

---

## Visualizations
- **Actual vs Predicted NO₂ (GRU)**
- **Actual vs Predicted CO (GRU)**
- **Actual vs Predicted NO₂ & CO (ARIMA)**

```markdown
![NO2 Prediction Plot](./plots/no2_prediction_plot.png)
![CO Prediction Plot](./plots/co_prediction_plot.png)
