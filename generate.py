import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Constants
num_records = 1000  # Total records to generate
places = ['Peelamedu', 'Singanallur']
vehicle_types = ['Two Wheeler', 'Auto Rickshaw', 'Car/Utility', 'Buses', 'Trucks', 'Total_Vehicles']
traffic_situations = ['Low', 'Moderate', 'High']

# Generate random timestamps for the last two months
current_time = datetime.now()
timestamps = [current_time - timedelta(days=np.random.randint(0, 60), hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60)) for _ in range(num_records)]

# Generate random data
data = {
    'Time': timestamps,
    'Place': np.random.choice(places, num_records),
    'Two Wheeler': np.random.randint(0, 50, num_records),
    'Auto Rickshaw': np.random.randint(0, 30, num_records),
    'Car/Utility': np.random.randint(0, 100, num_records),
    'Buses': np.random.randint(0, 20, num_records),
    'Trucks': np.random.randint(0, 15, num_records),
    'Total_Vehicles': np.random.randint(10, 200, num_records),
    'Traffic Situation': np.random.choice(traffic_situations, num_records)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('traffic_data.csv', index=False)

print("Dataset generated and saved as 'traffic_data.csv'")
