# Load the saved model
from tensorflow.keras.models import load_model
model = load_model('traffic_condition_model.h5')

# Create a new dataframe with the input features
new_input = pd.DataFrame({'Two Wheeler': [10], 'Auto Rickshaw': [20], 'Car/Utility': [30], 'Buses': [40], 'Trucks': [50], 'Total_Vehicles': [60], 'hour': [12], 'day_of_week': [3]})

# Define the scaler
scaler = MinMaxScaler()

# Fit the scaler to the new input data and transform it
new_input[['Two Wheeler', 'Auto Rickshaw', 'Car/Utility', 'Buses', 'Trucks', 'Total_Vehicles']] = scaler.fit_transform(new_input[['Two Wheeler', 'Auto Rickshaw', 'Car/Utility', 'Buses', 'Trucks', 'Total_Vehicles']])

# Make predictions on the new input
predictions = model.predict(new_input)

# Convert predictions to traffic conditions
traffic_condition = []
for prediction in predictions:
    if prediction.argmax() == 0:
        traffic_condition.append('Low')
    elif prediction.argmax() == 1:
        traffic_condition.append('Moderate')
    else:
        traffic_condition.append('High')

print('Predicted Traffic Condition:', traffic_condition[0])