import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

# --------------------- Data Loading and Preprocessing --------------------- #

# Load data from CSV file
df = pd.read_csv('traffic_data.csv')  # Ensure 'traffic_data.csv' is in the same directory

# Function to filter data for the last two months
def filter_last_two_months(df):
    current_date = pd.Timestamp.now()
    two_months_ago = current_date - timedelta(days=60)  # Approximate 2 months
    df['Time'] = pd.to_datetime(df['Time'])  # Ensure 'Time' column is in datetime format
    filtered_df = df[df['Time'] >= two_months_ago].reset_index(drop=True)
    return filtered_df

# Apply the filter
df = filter_last_two_months(df)

# Preprocessing functions
def encode_categorical(df):
    label_encoder = LabelEncoder()
    df['Traffic Situation'] = label_encoder.fit_transform(df['Traffic Situation'])
    return df, label_encoder

def extract_time_features(df):
    df['Hour'] = df['Time'].dt.hour
    df['Day'] = df['Time'].dt.day
    df['Month'] = df['Time'].dt.month
    return df

def prepare_features_target(df):
    X = df.drop(columns=['Traffic Situation', 'Place', 'Time']).values  # Features
    y = df['Traffic Situation'].values  # Target
    return X, y

# Apply preprocessing
df, label_encoder = encode_categorical(df)
df = extract_time_features(df)
X, y = prepare_features_target(df)

# Convert data to correct numeric types
X = X.astype(float)
y = y.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------- Preparing Data for GNN --------------------- #

def create_graph_data(X, y):
    data_list = []
    for i in range(X.shape[0]):
        x = torch.tensor(X[i], dtype=torch.float).unsqueeze(0)  # Single node feature
        edge_index = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)  # Self-loop
        y_label = torch.tensor([y[i]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y_label)
        data_list.append(data)
    return data_list

# Create Data objects
train_data_list = create_graph_data(X_train, y_train)
test_data_list = create_graph_data(X_test, y_test)

# Create DataLoader for batch processing
train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

# --------------------- Defining the GNN Model --------------------- #

class GNNModel(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_features)  # First GCN layer
        self.conv2 = GCNConv(hidden_features, output_classes)  # Second GCN layer

    def forward(self, data):
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

# --------------------- Training the GNN Model --------------------- #

def train(model, loader, optimizer, criterion, epochs=200):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()  # Zero gradients
            out = model(data)  # Forward pass
            loss = criterion(out, data.y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            total_loss += loss.item() * data.num_graphs
        avg_loss = total_loss / len(loader.dataset)
        if epoch % 20 == 0 or epoch == 1:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

# Train the model
print("Starting GNN training...")
train(model, train_loader, optimizer, criterion, epochs=200)
print("Training completed.")

# --------------------- Evaluating the GNN Model --------------------- #

def evaluate(model, loader):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            preds = out.argmax(dim=1)
            correct += (preds == data.y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    accuracy = correct / len(loader.dataset)
    return accuracy, all_preds, all_labels

# Evaluate on training data
train_acc, _, _ = evaluate(model, train_loader)
print(f'Training Accuracy: {train_acc:.4f}')

# Evaluate on test data
test_acc, test_preds, test_labels = evaluate(model, test_loader)
print(f'Test Accuracy: {test_acc:.4f}')

# Detailed classification report
print("Classification Report:")
print(classification_report(test_labels, test_preds, target_names=label_encoder.classes_))

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(label_encoder.classes_))
plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
plt.yticks(tick_marks, label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Annotate the confusion matrix
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# --------------------- Linear Regression for Vehicle Counts --------------------- #

def train_linear_regression_model(df):
    features = ['Hour', 'Day', 'Month']
    vehicle_types = ['Two Wheeler', 'Auto Rickshaw', 'Car/Utility', 'Buses', 'Trucks', 'Total_Vehicles']

    X = df[features]
    y = df[vehicle_types]

    models = {}
    for vehicle in vehicle_types:
        model = LinearRegression()
        model.fit(X, y[vehicle])
        models[vehicle] = model

    return models

# Train the regression models
linear_models = train_linear_regression_model(df)

# --------------------- Prediction Function --------------------- #

def extract_full_features_for_specific_date(target_date, selected_place):
    df_place = df[df['Place'] == selected_place]
    input_features = np.array([[target_date.hour, target_date.day, target_date.month]])

    predicted_counts = []
    for vehicle in ['Two Wheeler', 'Auto Rickshaw', 'Car/Utility', 'Buses', 'Trucks', 'Total_Vehicles']:
        predicted_count = linear_models[vehicle].predict(input_features)[0]
        predicted_counts.append(predicted_count)

    return predicted_counts  # Return only the predicted counts
# Global variables to store predictions
predicted_counts = []
user_datetime = None

# Modify the predict_traffic function
def predict_traffic():
    global predicted_counts, user_datetime  # Declare as global
    user_input = entry.get()
    selected_place = place_var.get()

    try:
        user_datetime = pd.to_datetime(user_input)  # Convert user input to datetime
        predicted_counts = extract_full_features_for_specific_date(user_datetime, selected_place)

        if isinstance(predicted_counts, list) and len(predicted_counts) > 0:
            total_vehicle_count = sum(predicted_counts)  # Sum all predicted vehicle counts
            total_vehicle_count = round(total_vehicle_count)  # Round the total vehicle count

            feature_vector = [user_datetime.hour, user_datetime.day, user_datetime.month] + predicted_counts
            feature_vector = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)  # Shape: [1, num_features]
            edge_index_pred = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)  # Self-loop

            # Make a prediction with GNN model
            with torch.no_grad():
                gnn_output = model(Data(x=feature_vector, edge_index=edge_index_pred))
                predicted_class = gnn_output.argmax(dim=1).item()
                predicted_traffic_condition = label_encoder.inverse_transform([predicted_class])[0]  # Get the traffic situation label

            # Show prediction result in message box
            messagebox.showinfo("Prediction Result",
                                f"Predicted Traffic Condition: {predicted_traffic_condition}\n"
                                f"Predicted Vehicle Counts:\n"
                                f"Two Wheeler: {int(predicted_counts[0])}\n"
                                f"Auto Rickshaw: {int(predicted_counts[1])}\n"
                                f"Car/Utility: {int(predicted_counts[2])}\n"
                                f"Buses: {int(predicted_counts[3])}\n"
                                f"Trucks: {int(predicted_counts[4])}\n"
                                f"Total Vehicles: {int(total_vehicle_count)}")

            # Generate bar chart for predicted vehicle counts
            plot_predicted_traffic(predicted_counts, user_datetime)  # Call the new plot function

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        
# --------------------- Plotting Functions --------------------- #

def plot_actual_traffic():
    plt.figure(figsize=(10, 5))
    actual_counts = df.groupby('Traffic Situation').size()
    actual_counts.plot(kind='bar', color='orange')
    plt.title('Actual Traffic Conditions')
    plt.xlabel('Traffic Situation')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_predicted_traffic(predicted_counts, user_datetime):
    plt.figure(figsize=(10, 5))
    vehicle_types = ['Two Wheeler', 'Auto Rickshaw', 'Car/Utility', 'Buses', 'Trucks']
    plt.bar(vehicle_types, predicted_counts[:5], color='blue')
    plt.title(f'Predicted Vehicle Counts for {user_datetime.strftime("%Y-%m-%d %H:%M")}')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --------------------- GUI for User Input --------------------- #

# Create the main window
root = tk.Tk()
root.title("Traffic Prediction Tool")

# Input for date and time
tk.Label(root, text="Enter date and time (YYYY-MM-DD HH:MM):").pack()
entry = tk.Entry(root)
entry.pack()

# Dropdown for selecting place
place_var = tk.StringVar(root)
place_var.set(df['Place'].unique()[0])  # Set default value
tk.OptionMenu(root, place_var, *df['Place'].unique()).pack()

# Button to make prediction
tk.Button(root, text="Predict Traffic", command=predict_traffic).pack()

# Button to plot actual traffic
tk.Button(root, text="Plot Actual Traffic", command=plot_actual_traffic).pack()

# Button to plot predicted traffic
tk.Button(root, text="Plot Predicted Traffic", command=lambda: plot_predicted_traffic(predicted_counts, user_datetime)).pack()

# Run the GUI event loop
root.mainloop()
