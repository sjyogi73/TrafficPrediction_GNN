import pandas as pd
from datetime import datetime, timedelta
import random

# Function to generate sample traffic data for the last two months
def generate_sample_data(num_records=1000):
    data = []
    current_time = datetime.now()
    
    # Calculate the starting date for the last two months
    start_time = current_time - timedelta(days=60)  # Last two months

    for _ in range(num_records):
        # Generate random vehicle counts
        two_wheeler = random.randint(0, 100)
        auto_rickshaw = random.randint(0, 50)
        car_utility = random.randint(0, 150)
        buses = random.randint(0, 20)
        trucks = random.randint(0, 30)
        
        total_vehicles = two_wheeler + auto_rickshaw + car_utility + buses + trucks
        
        # Generate a random traffic situation
        traffic_situation = random.choice(['Low', 'Moderate', 'High'])

        # Create a record with a timestamp within the last two months
        record_time = start_time + timedelta(days=random.randint(0, 60), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        
        # Append the record to the data list
        data.append([record_time, "Peelamedu", two_wheeler, auto_rickshaw, car_utility, buses, trucks, total_vehicles, traffic_situation])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=["Time", "Place", "Two Wheeler", "Auto Rickshaw", "Car/Utility", "Buses", "Trucks", "Total_Vehicles", "Traffic Situation"])
    
    return df

# Generate the sample dataset with 1000 records for the last two months
sample_data = generate_sample_data(1000)

# Display the sample dataset
print(sample_data)

# Save to CSV if needed
sample_data.to_csv("traffic_data.csv", index=False)
