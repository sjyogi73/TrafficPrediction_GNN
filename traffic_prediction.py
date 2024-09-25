import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data from CSV file
df = pd.read_csv('traffic_data.csv')  # Load your dataset

# Function to filter data for the last two months
def filter_last_two_months(df):
    """
    Filters the dataset to include only the last two months from the current date.
    """
    current_date = pd.Timestamp.now()
    two_months_ago = current_date - timedelta(days=60)  # Approximate 2 months
    df['Time'] = pd.to_datetime(df['Time'])  # Ensure 'Time' column is in datetime format
    filtered_df = df[df['Time'] >= two_months_ago]
    return filtered_df

# Filter the dataset for the last 2 months
df = filter_last_two_months(df)

# Preprocessing functions
def encode_categorical(df):
    """
    Encode the 'Traffic Situation' categorical column into numeric values using LabelEncoder.
    """
    label_encoder = LabelEncoder()
    df['Traffic Situation'] = label_encoder.fit_transform(df['Traffic Situation'])
    return df, label_encoder

def extract_time_features(df):
    """
    Extracts time-related features such as 'Hour', 'Day', and 'Month' from the 'Time' column.
    """
    df['Time'] = pd.to_datetime(df['Time'])
    df['Hour'] = df['Time'].dt.hour
    df['Day'] = df['Time'].dt.day
    df['Month'] = df['Time'].dt.month
    return df

def prepare_features_target(df):
    """
    Prepares the features and target variables for the model by dropping unnecessary columns and
    splitting the data into X (features) and y (target).
    """
    X = df.drop(columns=['Traffic Situation', 'Place', 'Time']).values  # Ensure these columns are dropped
    y = df['Traffic Situation'].values
    return X, y

# Preprocess data
df, label_encoder = encode_categorical(df)
df = extract_time_features(df)
X, y = prepare_features_target(df)

# Convert data to the correct numeric types
X = X.astype(float)
y = y.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple graph structure for GNN training
num_nodes_train = X_train.shape[0]
# Define edges for graph where every node is connected to every other node (fully connected graph)
edge_index_train = torch.tensor([[i, j] for i in range(num_nodes_train) for j in range(num_nodes_train) if i != j], dtype=torch.long).t().contiguous()

# Create graph data object for PyTorch Geometric
train_data = Data(x=torch.tensor(X_train, dtype=torch.float32), edge_index=edge_index_train, y=torch.tensor(y_train, dtype=torch.long))

# Create DataLoader for batch processing
train_loader = DataLoader([train_data], batch_size=32)

# Define a Graph Neural Network (GNN) model using GCNConv layers
class GNNModel(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_classes):
        """
        Initialize the GNN model with two Graph Convolutional layers (GCNConv).
        """
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_features)  # First GCN layer
        self.conv2 = GCNConv(hidden_features, output_classes)  # Second GCN layer

    def forward(self, data):
        """
        Forward pass of the GNN model.
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # Apply ReLU activation
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # Use log softmax for classification

# Model hyperparameters
input_features = X_train.shape[1]  # Number of input features
hidden_features = 16  # Number of hidden units in GCN layer
output_classes = len(label_encoder.classes_)  # Number of output classes (Traffic situations)

# Initialize model, optimizer, and loss function
model = GNNModel(input_features, hidden_features, output_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
criterion = torch.nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification

# Training loop
for epoch in range(200):
    model.train()  # Set model to training mode
    for data in train_loader:
        optimizer.zero_grad()  # Zero gradients
        out = model(data)  # Forward pass
        loss = criterion(out, data.y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

# Train a Linear Regression model for predicting vehicle counts based on time features
def train_linear_regression_model(df):
    """
    Trains a linear regression model to predict vehicle counts based on time features.
    Returns the model and feature column names.
    """
    features = ['Hour', 'Day', 'Month']
    vehicle_types = ['Two Wheeler', 'Auto Rickshaw', 'Car/Utility', 'Buses', 'Trucks', 'Total_Vehicles']

    X = df[features]
    y = df[vehicle_types]

    # Train a linear regression model for each vehicle type
    models = {}
    for vehicle in vehicle_types:
        model = LinearRegression()
        model.fit(X, y[vehicle])
        models[vehicle] = model
    
    return models

# Train the regression models
linear_models = train_linear_regression_model(df)

# Modify the function to extract features for a specific date using linear regression models
def extract_full_features_for_specific_date(target_date, selected_place):
    """
    Extracts the relevant features for a given date for prediction using linear regression models.
    """
    # Filter data for the selected place
    df_place = df[df['Place'] == selected_place]

    # Prepare the input for prediction using time features
    input_features = np.array([[target_date.hour, target_date.day, target_date.month]])

    # Predict vehicle counts using the linear regression models
    predicted_counts = []
    for vehicle in ['Two Wheeler', 'Auto Rickshaw', 'Car/Utility', 'Buses', 'Trucks', 'Total_Vehicles']:
        predicted_count = linear_models[vehicle].predict(input_features)[0]
        predicted_counts.append(predicted_count)

    return predicted_counts  # Return only the predicted counts

def predict_traffic():
    """
    Use the GNN model to predict traffic conditions for a given date input by the user.
    """
    user_input = entry.get()
    selected_place = place_var.get()
    
    try:
        user_datetime = pd.to_datetime(user_input)  # Convert user input to datetime
        predicted_counts = extract_full_features_for_specific_date(user_datetime, selected_place)

        # Ensure predicted_counts is a list and get the counts
        if isinstance(predicted_counts, list) and len(predicted_counts) > 0:
            # Calculate total vehicle count from extracted features
            total_vehicle_count = sum(predicted_counts)  # Sum all predicted vehicle counts
            total_vehicle_count = round(total_vehicle_count)  # Round the total vehicle count

            # Make prediction using the GNN model
            feature_vector = torch.tensor([user_datetime.hour, user_datetime.day, user_datetime.month] + predicted_counts, dtype=torch.float32).unsqueeze(0)  # Reshape for single sample input
            edge_index_pred = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop for a single node
            pred_data = Data(x=feature_vector, edge_index=edge_index_pred)

            with torch.no_grad():
                model.eval()
                out = model(pred_data)  # Predict traffic situation
                pred = out.argmax(dim=1).item()
                traffic_situation = label_encoder.inverse_transform([pred])[0]

                # Display results including total vehicle count and traffic situation
                messagebox.showinfo("Prediction Result", f'Total Vehicle Count: {total_vehicle_count}, Traffic Situation: {traffic_situation} for {selected_place}')
        else:
            messagebox.showerror("Error", "No predictions available.")
    
    except Exception as e:
        messagebox.showerror("Error", f"Error processing input: {e}")

# Function to plot a feature chart based on traffic situation and time of day
def plot_feature_chart():
    """
    Plot a bar chart showing traffic situation versus hour for the selected place.
    """
    selected_place = place_var.get()
    filtered_df = df[df['Place'] == selected_place]
    
    plt.figure(figsize=(10, 5))
    plt.bar(filtered_df['Hour'], filtered_df['Traffic Situation'], color='skyblue')
    plt.title(f'Feature Chart: Hour vs Traffic Situation for {selected_place}')
    plt.xlabel('Hour')
    plt.ylabel('Traffic Situation')
    plt.xticks(rotation=45)
    plt.show()

# Function to plot a feature chart based on traffic situation and time of day
def plot_feature_chartt():
    """
    Plot a bar chart showing traffic situation versus hour for the selected place.
    """
    selected_place = place_var.get()
    filtered_df = df[df['Place'] == selected_place]
    
    plt.figure(figsize=(10, 5))
    plt.bar(filtered_df['Hour'], filtered_df['Traffic Situation'], color='blue', alpha=0.7)
    plt.title(f'Traffic Situation for {selected_place} by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Traffic Situation')
    plt.xticks(range(0, 24))  # Set x-ticks for hours
    plt.grid(axis='y')
    plt.show()

# Function to plot the actual chart of traffic situation over time
def plot_actual_chart():
    """
    Plot a line chart showing the actual traffic situation over hours for the selected place.
    """
    selected_place = place_var.get()
    filtered_df = df[df['Place'] == selected_place]
    
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_df['Time'], filtered_df['Traffic Situation'], marker='o')
    plt.title(f'Actual Traffic Situation over Time for {selected_place}')
    plt.xlabel('Time')
    plt.ylabel('Traffic Situation')
    plt.xticks(rotation=45)
    plt.show()

# Function to plot the actual chart of vehicle counts over time
def plot_actual_chartt():
    """
    Plot a line chart showing the actual vehicle counts over time for the selected place.
    """
    selected_place = place_var.get()
    filtered_df = df[df['Place'] == selected_place]
    
    plt.figure(figsize=(12, 6))
    
    # Plotting each vehicle type
    plt.plot(filtered_df['Time'], filtered_df['Two Wheeler'], marker='o', label='Two Wheeler')
    plt.plot(filtered_df['Time'], filtered_df['Auto Rickshaw'], marker='o', label='Auto Rickshaw')
    plt.plot(filtered_df['Time'], filtered_df['Car/Utility'], marker='o', label='Car/Utility')
    plt.plot(filtered_df['Time'], filtered_df['Buses'], marker='o', label='Buses')
    plt.plot(filtered_df['Time'], filtered_df['Trucks'], marker='o', label='Trucks')
    plt.plot(filtered_df['Time'], filtered_df['Total_Vehicles'], marker='o', label='Total Vehicles')

    plt.title(f'Actual Vehicle Counts over Time for {selected_place}')
    plt.xlabel('Time')
    plt.ylabel('Number of Vehicles')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()


# Function to extract features for prediction
def extract_full_features(user_datetime, selected_place):
    """
    Extract the relevant features for a given datetime and place for prediction.
    Matches based on the closest hour available in the last two months dataset.
    """
    # Filter data to the last two months and selected place
    df_place = df[df['Place'] == selected_place]

    # Filter by day and month
    df_filtered = df_place[
        (df_place['Day'] == user_datetime.day) & 
        (df_place['Month'] == user_datetime.month)
    ]

    if not df_filtered.empty:
        # Find the row with the closest hour (ignoring minutes)
        closest_row = df_filtered.iloc[(df_filtered['Hour'] - user_datetime.hour).abs().argmin()]

        # Extract vehicle counts from the closest row
        two_wheeler = closest_row['Two Wheeler']
        auto_rickshaw = closest_row['Auto Rickshaw']
        car_utility = closest_row['Car/Utility']
        buses = closest_row['Buses']
        trucks = closest_row['Trucks']
        total_vehicles = closest_row['Total_Vehicles']

        return [
            closest_row['Hour'],  # Use the closest hour
            closest_row['Day'],
            closest_row['Month'],
            two_wheeler,
            auto_rickshaw,
            car_utility,
            buses,
            trucks,
            total_vehicles
        ]
    else:
        print(f"No data found for {selected_place} on {user_datetime.date()}")
        return [user_datetime.hour, user_datetime.day, user_datetime.month, 0, 0, 0, 0, 0, 0]


def plot_predicted_chart():
    """
    Plot a line chart showing the predicted traffic situation over hours for the selected place.
    """
    selected_place = place_var.get()
    filtered_df = df[df['Place'] == selected_place]

    predicted_situations = []
    hours = sorted(filtered_df['Hour'].unique())

    for hour in hours:
        # Ensure correct formatting with year, month, day, and time
        date_time_str = f'{filtered_df["Time"].iloc[0].year}-{filtered_df["Month"].iloc[0]:02d}-{filtered_df["Day"].iloc[0]:02d} {hour:02d}:00'
        datetime_obj = pd.to_datetime(date_time_str)
        
        feature_vector = extract_full_features(datetime_obj, selected_place)
        feature_vector = torch.tensor([feature_vector], dtype=torch.float32)
        edge_index_pred = torch.tensor([[0], [0]], dtype=torch.long)
        pred_data = Data(x=feature_vector, edge_index=edge_index_pred)
        
        with torch.no_grad():
            model.eval()
            out = model(pred_data)
            pred = out.argmax(dim=1).item()
            predicted_situations.append(label_encoder.inverse_transform([pred])[0])

    # Plot the predicted traffic situation
    plt.figure(figsize=(10, 5))
    plt.plot(hours, predicted_situations, marker='x', color='red', label='Predicted Traffic Situation')
    plt.title(f'Predicted Chart: Traffic Situation Over Hours for {selected_place}')
    plt.xlabel('Hour')
    plt.ylabel('Predicted Traffic Situation')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# Function to plot the predicted chart of vehicle counts over time
# Function to plot the predicted chart of vehicle counts over time
def plot_original():
    """
    Plot a line chart showing the predicted vehicle counts over time for the selected place.
    """
    selected_place = place_var.get()
    
    # Filter the original DataFrame for the selected place to get the predicted data
    filtered_df = df[df['Place'] == selected_place]

    plt.figure(figsize=(12, 6))
    
    # Preparing the hours for the x-axis (using unique hours from the data)
    unique_hours = sorted(filtered_df['Hour'].unique())
    
    # Initialize lists for predicted counts
    predicted_two_wheeler = []
    predicted_auto_rickshaw = []
    predicted_car_utility = []
    predicted_buses = []
    predicted_trucks = []
    predicted_total_vehicles = []
    
    for hour in unique_hours:
        # Ensure correct formatting with year, month, day, and time
        date_time_str = f'{filtered_df["Time"].iloc[0].year}-{filtered_df["Month"].iloc[0]:02d}-{filtered_df["Day"].iloc[0]:02d} {hour:02d}:00'
        datetime_obj = pd.to_datetime(date_time_str)
        
        # Extract features for prediction
        feature_vector = extract_full_features(datetime_obj, selected_place)
        feature_vector = torch.tensor([feature_vector], dtype=torch.float32)
        edge_index_pred = torch.tensor([[0], [0]], dtype=torch.long)
        pred_data = Data(x=feature_vector, edge_index=edge_index_pred)
        
        with torch.no_grad():
            model.eval()
            out = model(pred_data)
            predicted_counts = out.argmax(dim=1).item()
            
            # Store predicted counts for each vehicle type
            predicted_two_wheeler.append(predicted_counts[0])  # Assuming index 0 corresponds to Two Wheeler
            predicted_auto_rickshaw.append(predicted_counts[1])  # Adjust indices based on your output structure
            predicted_car_utility.append(predicted_counts[2])  # And so on...
            predicted_buses.append(predicted_counts[3])
            predicted_trucks.append(predicted_counts[4])
            predicted_total_vehicles.append(predicted_counts[5])

    # Plot the predicted vehicle counts
    plt.plot(unique_hours, predicted_two_wheeler, marker='o', label='Predicted Two Wheeler')
    plt.plot(unique_hours, predicted_auto_rickshaw, marker='o', label='Predicted Auto Rickshaw')
    plt.plot(unique_hours, predicted_car_utility, marker='o', label='Predicted Car/Utility')
    plt.plot(unique_hours, predicted_buses, marker='o', label='Predicted Buses')
    plt.plot(unique_hours, predicted_trucks, marker='o', label='Predicted Trucks')
    plt.plot(unique_hours, predicted_total_vehicles, marker='o', label='Predicted Total Vehicles')

    plt.title(f'Predicted Vehicle Counts Over Time for {selected_place}')
    plt.xlabel('Hour')
    plt.ylabel('Number of Vehicles')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()


# Create GUI
root = tk.Tk()
root.title("Traffic Prediction Tool")

# GUI elements
tk.Label(root, text="Enter Date (YYYY-MM-DD HH:MM:SS):").grid(row=0, column=0)
entry = tk.Entry(root)
entry.grid(row=0, column=1)

place_var = tk.StringVar(root)
place_var.set(df['Place'].unique()[0])  # Default to first place
tk.OptionMenu(root, place_var, *df['Place'].unique()).grid(row=1, column=0,columnspan=2)

tk.Button(root, text="Predict Traffic", command=predict_traffic).grid(row=2, column=0, columnspan=2)
tk.Button(root, text="Plot Actual Chart", command=plot_actual_chartt).grid(row=3, column=0, columnspan=2)
tk.Button(root, text="Feature Chart", command=plot_feature_chartt).grid(row=4, column=0, columnspan=2)
tk.Button(root, text="Show Predicted Traffic (GNN)", command=plot_predicted_chart).grid(row=5, column=0, columnspan=2) 
# tk.Button(root, text="Original Chart", command=plot_original).grid(row=6, column=0, columnspan=2)

# Start GUI loop
root.mainloop()
