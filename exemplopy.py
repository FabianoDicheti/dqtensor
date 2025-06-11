import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Parâmetros
WINDOW_SIZE = 72
FORECAST_SIZE = 5
STRIDE = 2
BATCH_SIZE = 64
EPOCHS = 10

# 1. Carregamento do dataset

df = pd.read_csv('./jena_climate.csv')
temperatura = df['T (degC)'].values.astype('float32')

# 2. Normalização

mean = temperatura[:200000].mean()
std = temperatura[:200000].std()
temperatura = (temperatura - mean) / std

# 3. Função para criar janelas
def create_windows(series, window_size, forecast_size, stride):
    X, y = [], []
    for i in range(0, len(series) - window_size - forecast_size, stride):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size:i + window_size + forecast_size])
    return np.array(X), np.array(y)

# 4. Criar janelas com passo 2
X, y = create_windows(temperatura, WINDOW_SIZE, FORECAST_SIZE, STRIDE)

# 5. Dividir treino, validação e teste
train_size = int(0.7 * len(X))
val_size = int(0.2 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# 6. Expandir dimensão para [samples, timesteps, features]
X_train = np.expand_dims(X_train, -1)
X_val = np.expand_dims(X_val, -1)
X_test = np.expand_dims(X_test, -1)

# 7. Modelo LSTM

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(WINDOW_SIZE, 1)),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(FORECAST_SIZE)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 8. Treinamento
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# 9. Avaliação
loss, mae = model.evaluate(X_test, y_test)
print(f"Teste - Loss: {loss:.4f} | MAE: {mae:.4f}")

