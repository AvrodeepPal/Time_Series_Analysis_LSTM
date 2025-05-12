# 📈 Stock Price Prediction using LSTM: ReLU vs GELU

This project offers a full **stock price forecasting pipeline** powered by **deep learning** using **LSTM (Long Short-Term Memory)** networks. It features **two variants**—one using traditional **ReLU** activation and another using modern **GELU**—with a side-by-side comparison of their prediction performance.

The goal is to explore whether newer activations like **GELU** can enhance time series predictions in financial datasets. The project is designed for **researchers, learners, and professionals** looking to experiment with deep learning in time series forecasting.

---

## 📌 Key Features

* ✅ **ReLU-based LSTM Model**: A conventional LSTM architecture for time series regression
* ✨ **GELU-based LSTM Model**: A modern activation with smoother gradient flow
* 📊 **Visual + Statistical Comparison**: Compare models in terms of accuracy, error metrics, and visual predictions
* 📅 **30-day Future Forecasting**: Predict upcoming stock prices beyond the historical data
* 🔍 **Metrics Report**: RMSE, MAE, MAPE, R² Score, and custom accuracy %
* 📈 **Visualization Suite**:

  * Actual vs Predicted prices
  * Loss curves
  * Prediction error histograms
  * Future forecast plots
  * Scatter plots for actual vs predicted
* ⚙️ **Flexible Configuration**: Change indicators, model hyperparameters, window sizes, and date ranges easily
* 🔬 **Side-by-Side Model Analysis**: Integrated cell for direct performance comparison of ReLU vs GELU

---

## 🧪 Notebook Structure

The notebook is structured into three logical, runnable sections (cells):

### 🔹 Cell 1: **ReLU LSTM Model**

* Standard LSTM architecture with ReLU activations
* Trained on historical stock data (customizable)
* Includes early stopping and learning rate scheduler
* Generates predictions, evaluates metrics, and visualizes results

### 🔹 Cell 2: **GELU LSTM Model**

* Identical architecture but with **GELU** activations
* Trains using the same dataset, split, and hyperparameters
* Outputs predictions, metrics, and visualizations for fair comparison

### 🔹 Cell 3: **Performance Comparison**

* Loads both trained models
* Compares predictions using:

  * 📉 Loss curves
  * 📈 Actual vs Predicted overlay plots
  * 📊 Metrics table (RMSE, MAE, MAPE, R²)
  * 🔮 30-day future forecasts
  * 🔴 Error histograms
  * 🔵 Scatter plots of predicted vs actual

---

## 📂 Data Requirements

Your dataset must be a **CSV file** with at least the following columns:

* `Date` (format: YYYY-MM-DD)
* `Close` (daily closing price)

Recommended additional columns:

* `Volume`, `Open`, `High`, `Low`

**Sample format**:

```
Date,Open,High,Low,Close,Adj Close,Volume
2010-01-04,213.43,214.50,212.38,214.01,214.01,123432400
```

---

## ⚙️ How It Works (Pipeline)

1. **Data Preprocessing**

   * Load CSV, parse dates
   * Interpolate missing values
   * Set up time-indexed DataFrame

2. **Feature Engineering**

   * Add technical indicators (MA7, MA20, MA50, RSI, ROC, Volatility)
   * Scale features using MinMaxScaler

3. **Sequence Generation**

   * Use a 60-day lookback window to form training sequences
   * Split into train/test datasets by date

4. **Model Training**

   * Define LSTM architecture with either ReLU or GELU
   * Train using Adam optimizer with EarlyStopping and ReduceLROnPlateau

5. **Evaluation & Visualization**

   * Inverse transform predictions
   * Compute RMSE, MAE, MAPE, R²
   * Plot predictions, losses, and error trends

6. **Future Forecasting**

   * Use latest known data to forecast next 30 days
   * Visualize projected future stock movement

7. **Model Comparison**

   * Evaluate both models side-by-side
   * Quantify GELU's impact over ReLU with stats and visuals

---

## 🛠 Setup Instructions

1. Clone the repository or open the notebook in **Google Colab**
2. Upload your stock data CSV
3. Install dependencies (Colab usually has most pre-installed):

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

4. Run each cell step by step (ReLU → GELU → Compare)

---

## 📊 Example Outputs

* 📉 **Loss Curves**
  Training vs Validation loss for both ReLU and GELU models

* 📈 **Actual vs Predicted Prices**
  Visual overlays showing prediction accuracy

* 🔮 **30-Day Future Forecast**
  Predicted future prices from the latest available data

* 📋 **Metrics Comparison Table**
  RMSE, MAE, MAPE, R² for ReLU vs GELU

* 🟥 **Error Distribution**
  Histogram of prediction errors for both models

---

## 💡 Customization Options

| Component                 | How to Customize                                     |
| ------------------------- | ---------------------------------------------------- |
| Lookback Window           | Modify `lookback=60` in sequence creation            |
| Technical Indicators      | Edit `add_technical_indicators()`                    |
| Model Architecture        | Change `build_model_relu()` / `build_model_gelu()`   |
| Train/Test Dates          | Set in `split_data_by_date()`                        |
| Epochs & Batch Size       | Change in `train_model()` function                   |
| Learning Rate & Callbacks | Adjust `Adam(learning_rate=...)` and callbacks logic |
| Visual Style              | Modify `matplotlib` / `seaborn` theme settings       |

---

## 📌 Use Cases

* 🔬 **Educational Projects**: Learn time series forecasting with deep learning
* 🧪 **Experimental Research**: Test GELU vs ReLU performance on financial data
* 💼 **ML Prototyping**: Rapidly prototype stock prediction ideas
* 🧠 **Model Tuning Practice**: Practice optimization, hyperparameter tuning, and architecture design

---

## ⚠️ Disclaimer

> This notebook is intended for **educational and research purposes only**.
> It does **not constitute financial advice** and should not be used for real-world trading decisions.
> Stock markets are inherently noisy and affected by many unpredictable external factors.

---

## 📎 References & Inspirations

* Time Series Forecasting with Deep Learning (TensorFlow/Keras)
* GELU Activation: Hendrycks & Gimpel (2016)
* Technical Indicators (Investopedia, TA-Lib concepts)
* LSTM Neural Networks for Sequence Modeling

---

## ⭐️ Like it? Star it!

If you find this project helpful, consider giving it a ⭐️ to support its development.
