import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.dates as mdates
import sys

class Config:
    RANDOM_SEED = 42
    SEQUENCE_LENGTH = 6  # 6 weeks of history to predict next week
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 20

# === Load CSV and process data ===
try:
    df = pd.read_csv("total_data.csv")
except FileNotFoundError:
    print("Error: Data file not found")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data: {str(e)}")
    sys.exit(1)

# Convert FL_DATE to datetime
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

# Handle missing values
df['ARR_DELAY'] = df['ARR_DELAY'].fillna(0)

# Remove cancelled and diverted
df_clean = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()

# === Weekly Aggregation ===
df_weekly = df_clean.set_index('FL_DATE').resample('W')['ARR_DELAY'].mean().dropna()
print(f"Weekly data from {df_weekly.index.min()} to {df_weekly.index.max()}")
print(f"Total weeks: {len(df_weekly)}")

# Normalize
scaler = MinMaxScaler()
data = df_weekly.values.reshape(-1, 1)
data_scaled = scaler.fit_transform(data)
data_scaled = torch.FloatTensor(data_scaled)

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

seq_length = Config.SEQUENCE_LENGTH
X, y = create_sequences(data_scaled, seq_length)

# Split
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

# === LSTM Model ===
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.linear(out)
        return out

# Instantiate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(hidden_size=Config.HIDDEN_SIZE, num_layers=Config.NUM_LAYERS, dropout=Config.DROPOUT).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# Move data to device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# === Training ===
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

print("Training started...")
for epoch in range(Config.EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    train_loss = criterion(output, y_train)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train Loss = {train_loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

    if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# === Evaluation ===
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

y_test_np = y_test.cpu().numpy()
y_pred_np = y_pred.cpu().numpy()
y_test_inv = scaler.inverse_transform(y_test_np)
y_pred_inv = scaler.inverse_transform(y_pred_np)

rmse = np.sqrt(np.mean((y_test_inv - y_pred_inv)**2))
mae = np.mean(np.abs(y_test_inv - y_pred_inv))
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / (y_test_inv + 1e-8))) * 100

print("\nTest Set Results:")
print(f"RMSE: {rmse:.2f} minutes")
print(f"MAE: {mae:.2f} minutes")
print(f"MAPE: {mape:.2f}%")

# === Forecast 52 weeks ===
print("\nForecasting 52 weeks...")
predictions = []
last_seq = data_scaled[-seq_length:].unsqueeze(0).to(device)

with torch.no_grad():
    for _ in range(52):
        pred = model(last_seq)
        predictions.append(pred.item())
        last_seq = torch.cat((last_seq[:, 1:, :], pred.unsqueeze(2)), dim=1)

predictions = np.array(predictions).reshape(-1, 1)
predictions_inv = scaler.inverse_transform(predictions)

# === Plotting ===
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Losses
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Plot 2: Actual vs Predicted
test_start_idx = len(df_weekly) - len(y_test)
test_dates = df_weekly.index[test_start_idx:]
ax2.plot(test_dates, y_test_inv, label='Actual')
ax2.plot(test_dates, y_pred_inv, label='Predicted')
ax2.set_title('Test Set: Actual vs Predicted')
ax2.set_xlabel('Date')
ax2.set_ylabel('Average Arrival Delay (minutes)')
ax2.legend()
ax2.grid(True)

# Plot 3: Historical + Forecast
ax3.plot(df_weekly.index, df_weekly.values, label='Historical', color='blue')
last_date = df_weekly.index[-1]
future_dates = [last_date + timedelta(weeks=i) for i in range(1, 53)]
ax3.plot(future_dates, predictions_inv, label='52-Week Forecast', color='red')
ax3.set_title('Weekly Historical Data and Forecast')
ax3.set_xlabel('Date')
ax3.set_ylabel('Avg Arrival Delay (minutes)')
ax3.legend()
ax3.grid(True)

# Plot 4: Confidence Interval
forecast_std = np.std(y_test_inv - y_pred_inv)
upper_bound = predictions_inv + 1.96 * forecast_std
lower_bound = predictions_inv - 1.96 * forecast_std
ax4.plot(future_dates, predictions_inv, label='Forecast', color='red')
ax4.fill_between(future_dates, lower_bound.flatten(), upper_bound.flatten(), 
                 alpha=0.3, color='red', label='95% CI')
ax4.set_title('Forecast with Confidence Interval')
ax4.set_xlabel('Date')
ax4.set_ylabel('Delay (minutes)')
ax4.legend()
ax4.grid(True)

# Improve x-axis formatting
for ax in [ax3, ax4]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

plt.tight_layout()
plt.show()

# Save forecast
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Delay': predictions_inv.flatten(),
    'Upper_Bound': upper_bound.flatten(),
    'Lower_Bound': lower_bound.flatten()
})
forecast_df.to_csv('weekly_flight_delay_forecast.csv', index=False)
print("\nForecast saved to 'weekly_flight_delay_forecast.csv'")
