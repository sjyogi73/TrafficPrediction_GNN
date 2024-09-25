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
    """
    Filters the dataset to include only the last two months from the current date.
    """
    current_date = pd.Timestamp.now()
    two_months_ago = current_date - timedelta(days=60)  # Approximate 2 months
    df['Time'] = pd.to_datetime(df['Time'])  # Ensure 'Time' column is in datetime format
    filtered_df = df[df['Time'] >= two_months_ago].reset_index(drop=True)
    return filtered_df

# Apply the filter
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
    df['Hour'] = df['Time'].dt.hour
    df['Day'] = df['Time'].dt.day
    df['Month'] = df['Time'].dt.month
    return df

def prepare_features_target(df):
    """
    Prepares the features and target variables for the model by dropping unnecessary columns and
    splitting the data into X (features) and y (target).
    """
    X = df.drop(columns=['Traffic Situation', 'Place', 'Time']).values  # Features
    y = df['Traffic Situation'].values  # Target
    return X, y

# Apply preprocessing
df, label_encoder = encode_categorical(df)
df = extract_time_features(df)
X, y = prepare_features_target(df)

# Convert data to the correct numeric types
X = X.astype(float)
y = y.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------- Preparing Data for GNN --------------------- #

def create_graph_data(X, y):
    """
    Creates a list of Data objects for PyTorch Geometric, each representing a single data point
    with a self-loop.
    """
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

# --------------------- Training the GNN Model --------------------- #

def train(model, loader, optimizer, criterion, epochs=200):
    """
    Trains the GNN model.
    """
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
    """
    Evaluates the GNN model on the provided DataLoader.
    """
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

# --------------------- Feature Extraction for Prediction --------------------- #

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

# --------------------- Prediction Function --------------------- #

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

            # Prepare feature vector for GNN
            feature_vector = [user_datetime.hour, user_datetime.day, user_datetime.month] + predicted_counts
            feature_vector = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)  # Shape: [1, num_features]
            edge_index_pred = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)  # Self-loop

            # Create Data object for prediction
            pred_data = Data(x=feature_vector, edge_index=edge_index_pred)

            with torch.no_grad():
                model.eval()
                out = model(pred_data)  # Predict traffic situation
                pred = out.argmax(dim=1).item()
                traffic_situation = label_encoder.inverse_transform([pred])[0]

                # Display results including total vehicle count and traffic situation
                messagebox.showinfo("Prediction Result", 
                                    f'Total Vehicle Count: {total_vehicle_count}\nTraffic Situation: {traffic_situation}\nPlace: {selected_place}')
        else:
            messagebox.showerror("Error", "No predictions available.")

    except Exception as e:
        messagebox.showerror("Error", f"Error processing input: {e}")

# --------------------- Enhanced Plotting Function --------------------- #

def plot_feature_chart():
    """
    Plot a dual-axis chart showing predicted total vehicle counts and traffic situations across hours for the selected place.
    """
    selected_place = place_var.get()

    # Generate predictions for each hour of the day for the selected place
    hours = list(range(24))
    predicted_vehicles = []
    predicted_situations = []

    for hour in hours:
        # Assume the prediction is for today
        target_date = pd.Timestamp.now().replace(hour=hour, minute=0, second=0, microsecond=0)
        predicted_counts = extract_full_features_for_specific_date(target_date, selected_place)

        if predicted_counts:
            total_vehicle_count = round(sum(predicted_counts))

            # Prepare feature vector for GNN
            feature_vector = [target_date.hour, target_date.day, target_date.month] + predicted_counts
            feature_vector = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)  # Shape: [1, num_features]
            edge_index_pred = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)  # Self-loop

            # Create Data object for prediction
            pred_data = Data(x=feature_vector, edge_index=edge_index_pred)

            with torch.no_grad():
                model.eval()
                out = model(pred_data)
                pred = out.argmax(dim=1).item()
                traffic_situation = label_encoder.inverse_transform([pred])[0]

            # Collect predicted values
            predicted_vehicles.append(total_vehicle_count)
            # For plotting, convert traffic situation to numerical for visualization
            traffic_numerical = label_encoder.transform([traffic_situation])[0]
            predicted_situations.append(traffic_numerical)

    # Plot predicted total vehicle counts and traffic situation across hours
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot predicted total vehicles as a bar plot
    color = 'tab:blue'
    ax1.set_xlabel('Hour of the Day')
    ax1.set_ylabel('Predicted Total Vehicles', color=color)
    ax1.bar(hours, predicted_vehicles, color=color, alpha=0.6, label='Predicted Total Vehicles')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(hours)

    # Create a second y-axis for traffic situation
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Predicted Traffic Situation', color=color)
    ax2.plot(hours, predicted_situations, color=color, marker='o', label='Predicted Traffic Situation')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xticks(hours)

    # Add title and legends
    plt.title(f'Predicted Traffic and Vehicle Counts for {selected_place}')
    fig.tight_layout()  # Adjust layout to prevent clipping

    # Create custom legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Show traffic situation labels on secondary y-axis
    ax2.set_yticks(range(len(label_encoder.classes_)))
    ax2.set_yticklabels(label_encoder.classes_)

    plt.show()

# --------------------- Additional Plotting Functions --------------------- #

def plot_actual_chart():
    """
    Plot a line chart showing the actual traffic situation over time for the selected place.
    """
    selected_place = place_var.get()
    filtered_df = df[df['Place'] == selected_place]

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df['Time'], filtered_df['Traffic Situation'], marker='o', linestyle='-', color='green')
    plt.title(f'Actual Traffic Situation over Time for {selected_place}')
    plt.xlabel('Time')
    plt.ylabel('Traffic Situation')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_actual_vehicle_counts():
    """
    Plot a line chart showing the actual vehicle counts over time for the selected place.
    """
    selected_place = place_var.get()
    filtered_df = df[df['Place'] == selected_place]

    plt.figure(figsize=(14, 7))

    vehicle_types = ['Two Wheeler', 'Auto Rickshaw', 'Car/Utility', 'Buses', 'Trucks', 'Total_Vehicles']
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']

    for vehicle, color in zip(vehicle_types, colors):
        plt.plot(filtered_df['Time'], filtered_df[vehicle], marker='o', label=vehicle, color=color)

    plt.title(f'Actual Vehicle Counts over Time for {selected_place}')
    plt.xlabel('Time')
    plt.ylabel('Number of Vehicles')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------- Tkinter GUI Setup --------------------- #

# Initialize Tkinter root
root = tk.Tk()
root.title("Traffic Prediction Tool")

# Configure grid layout
root.columnconfigure(0, weight=1, pad=10)
root.columnconfigure(1, weight=1, pad=10)
root.rowconfigure([0,1,2,3,4], weight=1, pad=10)

# Label and entry for date and time input
tk.Label(root, text="Enter Date & Time (YYYY-MM-DD HH:MM):").grid(row=0, column=0, sticky=tk.E)
entry = tk.Entry(root, width=30)
entry.grid(row=0, column=1, sticky=tk.W)

# Dropdown for place selection
tk.Label(root, text="Select Place:").grid(row=1, column=0, sticky=tk.E)
place_var = tk.StringVar(root)
place_var.set(df['Place'].unique()[0])  # Default to first place
place_menu = tk.OptionMenu(root, place_var, *df['Place'].unique())
place_menu.grid(row=1, column=1, sticky=tk.W)

# Button to predict traffic
predict_button = tk.Button(root, text="Predict Traffic", command=predict_traffic, bg='lightblue')
predict_button.grid(row=2, column=0, columnspan=2, pady=5)

# Button to plot feature chart based on predictions
plot_feature_button = tk.Button(root, text="Plot Predicted Feature Chart", command=plot_feature_chart, bg='lightgreen')
plot_feature_button.grid(row=3, column=0, columnspan=2, pady=5)

# Button to plot actual traffic situation
plot_actual_traffic_button = tk.Button(root, text="Plot Actual Traffic Situation", command=plot_actual_chart, bg='lightyellow')
plot_actual_traffic_button.grid(row=4, column=0, columnspan=2, pady=5)

# Button to plot actual vehicle counts
plot_actual_vehicle_button = tk.Button(root, text="Plot Actual Vehicle Counts", command=plot_actual_vehicle_counts, bg='lightpink')
plot_actual_vehicle_button.grid(row=5, column=0, columnspan=2, pady=5)

# Start the Tkinter main loop
root.mainloop()
