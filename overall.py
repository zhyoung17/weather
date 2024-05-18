import pandas as pd
import os
import base64
from io import BytesIO

from RF import RF
from ar import ar
from prophet_pred import prophet
from sarima import sarima
from data_cleaning import preprocess_data, make_stationary

file_path = 'Combined.xlsx'
df = preprocess_data(file_path)
df_stationary = make_stationary(df, 'Mean Temperature (deg C)')

if df_stationary is None:
    print("Data could not be made stationary.")
else:
    temp_file_path = 'preprocessed_combined.xlsx'
    df_stationary.to_excel(temp_file_path)
    rf_rmsfe, rf_plot_html = RF(df_stationary)
    ar_rmsfe, ar_plot_html = ar(df_stationary)
    prophet_rmsfe, prophet_plot_html = prophet(df_stationary)
    sarima_rmsfe, sarima_plot_html = sarima(df_stationary)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Weather Prediction Model Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h1 {{ text-align: center; }}
            .grid-container {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                padding: 20px;
            }}
            .model-section {{
                border: 1px solid #ccc;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }}
            .model-section img {{ width: 100%; height: auto; }}
            .metrics {{ font-size: 18px; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h1>Weather Prediction Model Comparison</h1>
        <div class="grid-container">
            <div class="model-section">
                <h2>Random Forest Model</h2>
                <div class="metrics">RMSFE: {rf_rmsfe:.4f}</div>
                {rf_plot_html}
            </div>
            <div class="model-section">
                <h2>AR Model</h2>
                <div class="metrics">RMSFE: {ar_rmsfe:.4f}</div>
                {ar_plot_html}
            </div>
            <div class="model-section">
                <h2>Prophet Model</h2>
                <div class="metrics">RMSFE: {prophet_rmsfe:.4f}</div>
                {prophet_plot_html}
            </div>
            <div class="model-section">
                <h2>SARIMA Model</h2>
                <div class="metrics">RMSFE: {sarima_rmsfe:.4f}</div>
                {sarima_plot_html}
            </div>
        </div>
    </body>
    </html>
    """

    report_path = "model_comparison_report.html"
    with open(report_path, "w") as file:
        file.write(html_content)

    print(f"Report generated: {report_path}")
