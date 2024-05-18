import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import base64
from io import BytesIO

def sarima(df):
    # Get the mean temperature data
    mean_temp = df['Mean Temperature (deg C)']

    # Split the data into training and test sets
    train_size = len(mean_temp) - 7
    train, test = mean_temp[:train_size], mean_temp[train_size:]

    # Fit the SARIMA model
    sarima_model = SARIMAX(train, order=(1, 1, 0), seasonal_order=(1, 1, 0, 12))
    sarima_fit = sarima_model.fit(disp=False)

    # Forecast the next 7 steps
    forecast = sarima_fit.get_forecast(steps=7)
    predictions = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Calculate RMSFE
    rmsfe = np.sqrt(mean_squared_error(test, predictions))

    def save_plot(fig):
        figfile = BytesIO()
        fig.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file
        figdata_png = base64.b64encode(figfile.getvalue()).decode('utf-8')
        return f'<img src="data:image/png;base64,{figdata_png}" />'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train, label='Train')
    ax.plot(test.index, test, label='Test')
    ax.plot(test.index, predictions, label='Predicted', color='red')
    ax.axvline(x=train.index[-1], color='black', linestyle='--', label='Train/Test Split')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Temperature (°C)')
    ax.set_title('One-step-ahead SARIMA Prediction')
    ax.legend()
    
    plot_html = save_plot(fig)
    plt.close(fig)
    
    return rmsfe, plot_html
