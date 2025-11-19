import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


from .model import LSTMModel, create_sequences

# ----------------------------
# Prepare data for LSTM
# ----------------------------
def prepare_lstm_data(close_prices, seq_length=60):

    # ---- Fix: ensure numeric dtype and drop NaN ----
    close_prices = close_prices.astype("float32")
    close_prices = close_prices.dropna()

    # ---- Fix: MinMaxScaler cannot handle constant series ----
    if close_prices.max() == close_prices.min():
        raise ValueError("close_prices has no variation (max == min). LSTM cannot train.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices.values.reshape(-1, 1))

    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)

    # Split
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Torch conversion
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # Dataloader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    return train_loader, test_loader, X_test, scaler, scaled_data, train_size


# ----------------------------
# Train LSTM
# ----------------------------
def train_lstm(model, train_loader, num_epochs=50, lr=0.001, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if device.type != "cpu":
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            
            if epoch == 0 and batch_idx == 0:
                print("X_batch shape:", X_batch.shape)
                print("y_batch shape:", y_batch.shape)
                print("outputs shape:", outputs.shape)
                print("sample y:", y_batch[:5].view(-1).tolist())
                print("sample outputs:", outputs[:5].view(-1).tolist())

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Loss (last batch): {loss.item():.8f}, "
                  f"Loss (avg): {avg_loss:.8f}")

    print("Training completed.")
    return model



# ----------------------------
# Prediction
# ----------------------------
def predict_lstm(model, X_test, scaler, scaled_data, seq_length=60, forecast_steps=20):

    model.eval()
    with torch.no_grad():

        test_predictions = model(X_test).cpu().numpy()

        # future forecast
        last_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
        last_seq = torch.tensor(last_seq, dtype=torch.float32)

        future = []
        for _ in range(forecast_steps):
            pred = model(last_seq).item()
            future.append(pred)

            last_seq = torch.cat(
                (last_seq[:, 1:, :], torch.tensor([[[pred]]], dtype=torch.float32)),
                dim=1
            )

    # inverse scaling
    min_v = scaler.data_min_[0]
    scale = scaler.data_max_[0] - scaler.data_min_[0]

    test_predictions = test_predictions * scale + min_v
    future = np.array(future) * scale + min_v

    return test_predictions, future