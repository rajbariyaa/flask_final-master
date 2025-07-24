import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import plotly.graph_objects as go
import plotly.io as pio

# === Load the dataset ===
df = pd.read_csv("flights_sample_3m.csv", parse_dates=["FL_DATE"])  

# === Clean and filter ===
df['ARR_DELAY'] = df['ARR_DELAY'].fillna(0)

# Filter only up to August 2023
df = df[df['FL_DATE'] <= "2023-08-31"]

# Aggregate daily average ARR_DELAY
df_daily = df.groupby('FL_DATE')['ARR_DELAY'].mean().reset_index()
df_daily = df_daily.set_index('FL_DATE').sort_index()

# === Normalize the target ===
scaler = MinMaxScaler(feature_range=(-1, 1))
data = df_daily['ARR_DELAY'].values.reshape(-1, 1)
data_scaled = scaler.fit_transform(data)
data_scaled = torch.FloatTensor(data_scaled)

# === Sequence generation ===
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

seq_length = 30
X, y = create_sequences(data_scaled, seq_length)

# === Train/Test Split ===
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# === Define LSTM Model ===
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.linear(out[:, -1, :])

# === Train the model ===
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# === Evaluate ===
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

y_test_np = y_test.numpy()
y_pred_np = y_pred.numpy()
y_test_inv = scaler.inverse_transform(y_test_np)
y_pred_inv = scaler.inverse_transform(y_pred_np)
rmse = np.sqrt(np.mean((y_test_inv - y_pred_inv) ** 2))
print(f"Test RMSE: {rmse:.2f}")

# === Forecast next 365 days ===
predictions = []
last_seq = data_scaled[-seq_length:].unsqueeze(0)

with torch.no_grad():
    for _ in range(365):
        pred = model(last_seq)
        predictions.append(pred.item())
        last_seq = torch.cat((last_seq[:, 1:, :], pred.unsqueeze(2)), dim=1)

predictions = np.array(predictions).reshape(-1, 1)
predictions_inv = scaler.inverse_transform(predictions)

# === Plot and Save ===
last_date = df_daily.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 366)]

fig = go.Figure()

# Apply a 7-day rolling average for smoothness
smoothed_delay = df_daily['ARR_DELAY'].rolling(window=7, center=True).mean()

fig.add_trace(go.Scatter(
    x=df_daily.index, y=smoothed_delay,
    mode='lines', name='Smoothed Historical (7-day Avg)', line=dict(color='blue')
))

fig.add_trace(go.Scatter(x=future_dates, y=predictions_inv.flatten(),
                         mode='lines', name='Forecast (365 Days)', line=dict(color='red')))

fig.update_layout(
    title="Arrival Delay Forecast (PyTorch LSTM)",
    xaxis_title="Date",
    yaxis_title="Average Arrival Delay (minutes)",
    legend=dict(x=0.01, y=0.99),
    template="plotly_white"
)

pio.write_html(fig, file="arrival_delay_forecast.html", auto_open=True)
print("✅ Forecast plot saved to 'arrival_delay_forecast.html'")
