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
    
    print("Starting LSTM training...")
    for epoch in range(num_epochs):
        model.train()
        loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'LSTM Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
    
    print("LSTM Training completed.")
    return model

def train_mlp(model, train_loader, num_epochs=50, lr=0.001):
    """
    Handles the training loop for the MLP model.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting MLP training...")
    for epoch in range(num_epochs):
        model.train()
        loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f'MLP Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
            
    print("MLP Training completed.")
    return model

def predict_lstm(model, X_test, scaled_data, scaler, seq_length, forecast_steps=100):
    """
    Original LSTM-only prediction and forecasting function.
    """
    model.eval()
    with torch.no_grad():
        # Predict on Test Data
        test_predictions = model(X_test).cpu().numpy()
        
        # Forecast Future Values
        last_sequence_data = scaled_data[-seq_length:]
        curr_sequence = torch.tensor(last_sequence_data, dtype=torch.float32).unsqueeze(0) 
        
        future_predictions = []
        for _ in range(forecast_steps):
            pred = model(curr_sequence).item()
            future_predictions.append(pred)
            pred_tensor = torch.tensor([[[pred]]], dtype=torch.float32)
            curr_sequence = torch.cat((curr_sequence[:, 1:, :], pred_tensor), dim=1)

    test_predictions = scaler.inverse_transform(test_predictions)
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return test_predictions, future_predictions

def run_hybrid_forecast(lstm_model, mlp_model, scaled_data, exog_data, scaler_price, scaler_exog, seq_length, forecast_steps=100, noise_level=0.02):
    """
    Generates future forecasts using the hybrid LSTM + MLP approach.
    
    Args:
        noise_level (float): Standard deviation of Gaussian noise added to the feedback loop 
                             to simulate volatility and prevent 'straight line' predictions.
    """
    lstm_model.eval()
    mlp_model.eval()
    
    # 1. Initialize the sequence with the last available data
    last_sequence_data = scaled_data[-seq_length:]
    curr_sequence = torch.tensor(last_sequence_data, dtype=torch.float32).unsqueeze(0) # Shape: [1, seq_len, 1]
    
    # 2. Prepare future exogenous variables (Use the LATEST available data)
    last_exog = exog_data[-1:] # Shape: [1, num_exog_features]
    last_exog_tensor = torch.tensor(last_exog, dtype=torch.float32)
    
    future_predictions = []
    
    with torch.no_grad():
        for _ in range(forecast_steps):
            # A. Get LSTM prediction (Base feature)
            lstm_pred = lstm_model(curr_sequence) # Shape: [1, 1]
            
            # B. Combine LSTM prediction with Exogenous features
            mlp_input = torch.cat((lstm_pred, last_exog_tensor), dim=1) # Shape: [1, 1 + n_exog]
            
            # C. Get Final MLP prediction
            final_pred = mlp_model(mlp_input).item()
            future_predictions.append(final_pred)
            
            # D. Update LSTM Sequence for next step
            # CRITICAL FIX: Feed the FINAL MLP prediction back to LSTM, not the intermediate LSTM prediction.
            # This ensures LSTM reacts to the exogenous correction.
            
            # Add some noise to the feedback to simulate market volatility (prevent straight lines)
            noise = np.random.normal(0, noise_level)
            feedback_value = final_pred + noise
            
            feedback_tensor = torch.tensor([[[feedback_value]]], dtype=torch.float32).to(curr_sequence.device)
            
            # Update sequence: remove oldest, add new feedback
            curr_sequence = torch.cat((curr_sequence[:, 1:, :], feedback_tensor), dim=1)

    # Inverse transform predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_actual = scaler_price.inverse_transform(future_predictions)
    
    return future_predictions_actual