import pandas as pd
import random

# Sample data generation
date_range = pd.date_range(start='2024-04-01', end='2024-09-21', freq='h')  # Use 'h' for hourly frequency
places = ['Hopes', 'Gandhipuram', 'Peelamedu']
data = []

# Define thresholds for traffic situation
low_threshold = 1000
medium_threshold = 2000

for time in date_range:
    place = random.choice(places)

    # Randomly generate a total vehicle count to ensure all traffic situations are represented
    total_vehicles = random.choice([
        random.randint(0, low_threshold - 1),  # Low
        random.randint(low_threshold, medium_threshold - 1),  # Medium
        random.randint(medium_threshold, 3000)  # High
    ])

    # Initialize vehicle counts
    two_wheeler = min(random.randint(0, 20), total_vehicles)
    remaining_vehicles = total_vehicles - two_wheeler

    auto_rickshaw = min(random.randint(600, 800), remaining_vehicles)
    remaining_vehicles -= auto_rickshaw

    car_utility = min(random.randint(150, 250), remaining_vehicles)
    remaining_vehicles -= car_utility

    buses = min(random.randint(400, 500), remaining_vehicles)
    remaining_vehicles -= buses

    # Assign the remaining vehicles to trucks
    trucks = remaining_vehicles

    # Ensure no negative counts
    if trucks < 0:
        trucks = 0

    # Determine traffic situation
    if total_vehicles < low_threshold:
        traffic_situation = 'Low'
    elif low_threshold <= total_vehicles < medium_threshold:
        traffic_situation = 'Medium'
    else:
        traffic_situation = 'High'

    data.append([time, place, two_wheeler, auto_rickshaw, car_utility, buses, trucks, total_vehicles, traffic_situation])

# Create DataFrame
columns = ['Time', 'Place', 'Two Wheeler', 'Auto Rickshaw', 'Car/Utility', 'Buses', 'Trucks', 'Total_Vehicles', 'Traffic Situation']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('traffic_data.csv', index=False)
