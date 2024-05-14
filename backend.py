import requests
import pandas as pd
from datetime import datetime, timedelta

# Define the base URL for the API
base_url = "https://api.data.gov.sg/v1/environment/air-temperature"

# Function to get air temperature readings for a specific date
def get_air_temperature(date):
    # Create the query parameters
    params = {'date': date}
    
    # Make the GET request
    response = requests.get(base_url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Request failed for date: {date} with status code: {response.status_code}")
        return None

# Define the date range for historical data
start_date = datetime.strptime("2024-05-13", "%Y-%m-%d")
end_date = datetime.now() - timedelta(days=1)
all_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# Initialize an empty list to store the readings
all_readings = []

# Loop through each date and get the readings
for date in all_dates:
    formatted_date = date.strftime("%Y-%m-%d")
    print(f"Fetching data for date: {formatted_date}")
    daily_data = get_air_temperature(formatted_date)
    
    # Check if daily_data and daily_data['items'] are not None and have data
    if daily_data and 'items' in daily_data and len(daily_data['items']) > 0:
        readings = daily_data['items'][0]['readings']
        timestamp = daily_data['items'][0]['timestamp']
        
        for reading in readings:
            reading['timestamp'] = timestamp
            all_readings.append(reading)
    else:
        print(f"Date: {formatted_date} has no data")

# Convert the list of readings to a DataFrame
all_readings_df = pd.DataFrame(all_readings)

# Save the DataFrame to a CSV file
all_readings_df.to_csv("historical_daily_air_temperature_readings.csv", index=False)

# Print a summary of the data
print("Historical daily air temperature readings:")
print(all_readings_df.head())
