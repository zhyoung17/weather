import pandas as pd
from statsmodels.tsa.stattools import adfuller

def preprocess_data(file_path):
    df = pd.read_excel(file_path)
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    # Set the 'date' as the index
    df.set_index('date', inplace=True)

    numeric_columns = [
        'Daily Rainfall Total (mm)', 'Highest 30 min Rainfall (mm)', 'Highest 60 min Rainfall (mm)',
        'Highest 120 min Rainfall (mm)', 'Mean Temperature (deg C)', 'Maximum Temperature (deg C)',
        'Minimum Temperature (deg C)', 'Mean Wind Speed (km/h)', 'Max Wind Speed (km/h)'
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df.dropna(inplace=True)

    return df

def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    return result[1]

def make_stationary(df, target_column):
    mean_temp = df[target_column]
    p_value = check_stationarity(mean_temp)

    if p_value > 0.05:
        # Apply non-seasonal differencing if not stationary
        mean_temp_diff = mean_temp.diff().dropna()
        p_value_diff = check_stationarity(mean_temp_diff)

        if p_value_diff > 0.05:
            # Apply seasonal differencing if still not stationary
            mean_temp_diff = mean_temp.diff().diff(365).dropna()
            p_value_seasonal_diff = check_stationarity(mean_temp_diff)

            if p_value_seasonal_diff > 0.05:
                return None  # Data is still not stationary after differencing
            else:
                df[target_column] = mean_temp_diff
        else:
            df[target_column] = mean_temp_diff
    else:
        df[target_column] = mean_temp

    return df
