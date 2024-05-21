import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def RF(df):
    mean_temp = df['Mean Temperature (deg C)']
    df = pd.get_dummies(df, columns=['Station'], drop_first=True)
    lags = 7
    lagged_features = pd.DataFrame()
    for i in range(1, lags + 1):
        lagged_features[f'lag_{i}'] = mean_temp.shift(i)
    lagged_features = lagged_features.dropna()
    mean_temp = mean_temp[lagged_features.index]
    predictors = df.loc[lagged_features.index].drop(columns=['Mean Temperature (deg C)'])
    predictors = pd.concat([predictors, lagged_features], axis=1)

    # Split the data into training and test sets
    train_size = len(predictors) - 7
    train_X, test_X = predictors[:train_size], predictors[train_size:]
    train_y, test_y = mean_temp[:train_size], mean_temp[train_size:]

    # Train the Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train_X, train_y)

    predictions = rf.predict(test_X)
    rmsfe = np.sqrt(mean_squared_error(test_y, predictions))

    feature_importances = rf.feature_importances_
    features_df = pd.DataFrame({
        'Feature': train_X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(mean_temp.index[:train_size], mean_temp[:train_size], label='Train')
    ax.plot(mean_temp.index[train_size:], mean_temp[train_size:], label='Test')
    ax.plot(test_y.index, predictions, label='Predicted', color='red')
    ax.axvline(x=mean_temp.index[train_size-1], color='black', linestyle='--', label='Train/Test Split')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Temperature (°C)')
    ax.set_title('Random Forest Prediction')
    ax.legend()

    def save_plot(fig, filename):
        figfile = BytesIO()
        fig.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue()).decode('utf-8')
        html_str = f'<img src="data:image/png;base64,{figdata_png}" />'
        return html_str
    
    plot_html = save_plot(fig, 'rf_plot.png')
    plt.close(fig)
    
    return rmsfe, plot_html
