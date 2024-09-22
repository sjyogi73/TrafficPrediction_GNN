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

# Load data from CSV file
df = pd.read_csv('traffic_data.csv')

# Preprocessing functions
def encode_categorical(df):
    label_encoder = LabelEncoder()
    df['Traffic Situation'] = label_encoder.fit_transform(df['Traffic Situation'])
    return df, label_encoder

def extract_time_features(df):
    df['Time'] = pd.to_datetime(df['Time'])
    df['Hour'] = df['Time'].dt.hour
    df['Day'] = df['Time'].dt.day
    df['Month'] = df['Time'].dt.month
    return df

def prepare_features_target(df):
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

# Create a simple graph structure
num_nodes_train = X_train.shape[0]
edge_index_train = torch.tensor([[i, j] for i in range(num_nodes_train) for j in range(num_nodes_train) if i != j], dtype=torch.long).t().contiguous()

train_data = Data(x=torch.tensor(X_train, dtype=torch.float32), edge_index=edge_index_train, y=torch.tensor(y_train, dtype=torch.long))

# Create DataLoader
train_loader = DataLoader([train_data], batch_size=32)

class GNNModel(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, output_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Model hyperparameters
input_features = X_train.shape[1]
hidden_features = 16
output_classes = len(label_encoder.classes_)

# Initialize model, optimizer, and loss function
model = GNNModel(input_features, hidden_features, output_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(200):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

# Function to extract full features for prediction
def extract_full_features(user_datetime):
    return [
        user_datetime.hour,
        user_datetime.day,
        user_datetime.month,
        0,  # Placeholder for Two Wheeler
        0,  # Placeholder for Auto Rickshaw
        0,  # Placeholder for Car/Utility
        0,  # Placeholder for Buses
        0,  # Placeholder for Trucks
        0,  # Placeholder for Total_Vehicles
    ]

# Function to make prediction
def predict_traffic():
    user_input = entry.get()
    selected_place = place_var.get()
    
    try:
        user_datetime = pd.to_datetime(user_input)
        feature_vector = extract_full_features(user_datetime)
        feature_vector = torch.tensor([feature_vector], dtype=torch.float32)
        edge_index_pred = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop for a single node
        pred_data = Data(x=feature_vector, edge_index=edge_index_pred)

        with torch.no_grad():
            out = model(pred_data)
            pred = out.argmax(dim=1).item()
            traffic_situation = label_encoder.inverse_transform([pred])[0]
            total_vehicle_count = feature_vector.sum().item()  # Example calculation

            messagebox.showinfo("Prediction Result", f'Total Vehicle Count: {total_vehicle_count}, Traffic Situation: {traffic_situation} for {selected_place}')
    except Exception as e:
        messagebox.showerror("Error", f"Error processing input: {e}")

# Function to plot feature chart
def plot_feature_chart():
    selected_place = place_var.get()
    filtered_df = df[df['Place'] == selected_place]
    
    plt.figure(figsize=(10, 5))
    plt.bar(filtered_df['Hour'], filtered_df['Traffic Situation'], color='skyblue')
    plt.title(f'Feature Chart: Hour vs Traffic Situation for {selected_place}')
    plt.xlabel('Hour')
    plt.ylabel('Traffic Situation')
    plt.xticks(rotation=45)
    plt.show()

# Function to plot actual chart (assuming you have actual data to compare)
def plot_actual_chart():
    selected_place = place_var.get()
    filtered_df = df[df['Place'] == selected_place]
    
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_df['Hour'], filtered_df['Traffic Situation'], marker='o', label='Traffic Situation')
    plt.title(f'Actual Chart: Traffic Situation Over Hours for {selected_place}')
    plt.xlabel('Hour')
    plt.ylabel('Traffic Situation')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# GUI Setup
root = tk.Tk()
root.title("Traffic Prediction")

label = tk.Label(root, text="Enter a date and time (YYYY-MM-DD HH:MM):")
label.pack(pady=10)

entry = tk.Entry(root)
entry.pack(pady=10)

# Dropdown for selecting place
place_var = tk.StringVar(root)
place_var.set(df['Place'].unique()[0])  # Default value

place_dropdown = tk.OptionMenu(root, place_var, *df['Place'].unique())
place_dropdown.pack(pady=10)

predict_button = tk.Button(root, text="Predict", command=predict_traffic)
predict_button.pack(pady=10)

feature_chart_button = tk.Button(root, text="Feature Chart", command=plot_feature_chart)
feature_chart_button.pack(pady=10)

actual_chart_button = tk.Button(root, text="Actual Chart", command=plot_actual_chart)
actual_chart_button.pack(pady=10)

root.mainloop()
