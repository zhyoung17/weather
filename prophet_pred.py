import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import base64
from io import BytesIO

def prophet(df):
    # Prepare the data for Prophet
    df_prophet = df.reset_index()[['date', 'Mean Temperature (deg C)']]
    df_prophet.columns = ['ds', 'y']

    # Split the data into training and test sets
    train_size = len(df_prophet) - 7
    train, test = df_prophet[:train_size], df_prophet[train_size:]

    # Initialize and fit the Prophet model with additional seasonalities
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(train)

    # Make predictions
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_prophet['ds'], df_prophet['y'], 'k.', label='Observed data points')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2, label='Uncertainty interval')
    ax.axvline(x=train['ds'].iloc[-1], color='black', linestyle='--', label='Train/Test Split')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Temperature (°C)')
    ax.set_title('Prophet Forecast')
    ax.legend()

    def save_plot(fig):
        figfile = BytesIO()
        fig.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file
        figdata_png = base64.b64encode(figfile.getvalue()).decode('utf-8')
        return f'<img src="data:image/png;base64,{figdata_png}" />'
    
    plot_html = save_plot(fig)
    plt.close(fig)

    # Calculate RMSFE
    predicted = forecast['yhat'].iloc[-7:].values
    actual = test['y'].values
    rmsfe = np.sqrt(mean_squared_error(actual, predicted))

    return rmsfe, plot_html
