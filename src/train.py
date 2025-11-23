import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


from .model import LSTMModel

def create_sequences(data, seq_length):
    """
    Creates sequences for LSTM training.
    """
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

def train_lstm(model, train_loader, num_epochs=50, lr=0.001):
    """
    Handles the training loop for the LSTM model.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        loss = 0 # Initialize loss for scope safety
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
    
    print("Training completed.")
    return model

def predict_lstm(model, X_test, scaled_data, scaler, seq_length, forecast_steps=100):
    """
    Handles model evaluation, test set prediction, and future forecasting.
    """
    model.eval()
    with torch.no_grad():
        # Predict on Test Data
        test_predictions = model(X_test).cpu().numpy()
        
        # Forecast Future Values
        # For forecasting future values, use the last sequence
        last_sequence_data = scaled_data[-seq_length:]
        curr_sequence = torch.tensor(last_sequence_data, dtype=torch.float32).unsqueeze(0) # Shape: [1, seq_len, 1]
        
        future_predictions = []
        for _ in range(forecast_steps):
            pred = model(curr_sequence).item()
            future_predictions.append(pred)
            # Update the sequence
            pred_tensor = torch.tensor([[[pred]]], dtype=torch.float32)
            curr_sequence = torch.cat((curr_sequence[:, 1:, :], pred_tensor), dim=1)

    # Inverse transform predictions
    test_predictions = scaler.inverse_transform(test_predictions)
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return test_predictions, future_predictions