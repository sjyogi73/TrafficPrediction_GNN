from tensorflow.keras.models import load_model
import pandas as pd

# Load the trained model
model = load_model('traffic_condition_model.h5')

# Input data
input_data = pd.DataFrame({'date': ['2024-12-12 01:00']})
input_data['date'] = pd.to_datetime(input_data['date'])
input_data['hour'] = input_data['date'].dt.hour
input_data['day_of_week'] = input_data['date'].dt.dayofweek
input_data = input_data.drop('date', axis=1)

# Add other features (assuming they are available)
input_data['weather'] = ['sunny']  # replace with actual weather data
input_data['temperature'] = [25]  # replace with actual temperature data
input_data['humidity'] = [60]  # replace with actual humidity data
input_data['traffic_volume'] = [100]  # replace with actual traffic volume data
input_data['traffic_speed'] = [30]  # replace with actual traffic speed data
input_data['traffic_occupancy'] = [20]  # replace with actual traffic occupancy data

# Scale the data (assuming you used MinMaxScaler during training)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
input_data[['traffic_volume', 'traffic_speed', 'traffic_occupancy']] = scaler.fit_transform(input_data[['traffic_volume', 'traffic_speed', 'traffic_occupancy']])

# Convert categorical features to numerical features
input_data['weather'] = pd.Categorical(input_data['weather']).codes

# Make predictions
predictions = model.predict(input_data)

# Convert predictions to traffic conditions
traffic_conditions = []
for prediction in predictions:
    if prediction < 0.3:
        traffic_conditions.append('Low')
    elif prediction < 0.6:
        traffic_conditions.append('Moderate')
    else:
        traffic_conditions.append('High')

# Output the results
print('Traffic Conditions:')
print('Two Wheeler:', traffic_conditions[0])
print('Auto Rickshaw:', traffic_conditions[1])
print('Car/Utility:', traffic_conditions[2])
print('Buses:', traffic_conditions[3])
print('Trucks:', traffic_conditions[4])
print('Total Vehicles:', traffic_conditions[5])