import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from io import BytesIO
import base64


def ar(df):
    # Get the mean temperature data
    mean_temp = df['Mean Temperature (deg C)']

    # Make the data stationary by differencing
    mean_temp_diff = mean_temp.diff().dropna()

    # Split the data into training and test sets
    train_size = len(mean_temp_diff) - 7
    train, test = mean_temp_diff[:train_size], mean_temp_diff[train_size:]

    # Function to find the optimal lag using AIC
    def find_optimal_lag(train_data, max_lag):
        best_aic = np.inf
        best_lag = 1
        for lag in range(1, max_lag + 1):
            model = AutoReg(train_data, lags=lag, old_names=False)
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_aic:
                best_aic = aic
                best_lag = lag
        return best_lag

    # Function to perform one-step-ahead AR forecast
    def one_step_ar_forecast(train_data, steps, max_lag):
        history = list(train_data)
        predictions = []
        for t in range(steps):
            optimal_lag = find_optimal_lag(history, max_lag)
            model = AutoReg(history, lags=optimal_lag, old_names=False)
            model_fit = model.fit()
            yhat = model_fit.predict(start=len(history), end=len(history))[0]
            predictions.append(yhat)
            history.append(yhat)
        return predictions
    
    def save_plot(fig):
        figfile = BytesIO()
        fig.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file
        figdata_png = base64.b64encode(figfile.getvalue()).decode('utf-8')
        return f'<img src="data:image/png;base64,{figdata_png}" />'

    # Perform one-step-ahead forecast for 7 steps
    steps = 7
    predictions_diff = one_step_ar_forecast(train, steps, max_lag=8)

    # Reverse the differencing to get the actual forecasted values
    last_value = mean_temp.iloc[train_size - 1]  # The last actual value before the test period
    predictions = [last_value + sum(predictions_diff[:i+1]) for i in range(len(predictions_diff))]

    # Create a DataFrame for the predictions
    predicted_dates = pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    predictions_df = pd.DataFrame({'date': predicted_dates, 'predicted_mean_temperature': predictions})

    # Calculate RMSFE
    actual_values = mean_temp[train_size:train_size + steps]
    rmsfe = np.sqrt(mean_squared_error(actual_values, predictions))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(mean_temp.index[:train_size], mean_temp[:train_size], label='Train')
    ax.plot(mean_temp.index[train_size:], mean_temp[train_size:], label='Test')
    ax.plot(predicted_dates, predictions, label='Predicted', color='red')
    ax.axvline(x=mean_temp.index[train_size-1], color='black', linestyle='--', label='Train/Test Split')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Temperature (°C)')
    ax.set_title('One-step-ahead AR Prediction')
    ax.legend()
    
    plot_html = save_plot(fig)
    plt.close(fig)
    
    return rmsfe, plot_html
