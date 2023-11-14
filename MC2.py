import os
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

symbols = ['^GSPC', 'GOOG', 'MSFT']
symbols = ['^AEX', '^BFX', '^GDAXI']
start_date = '2010-01-01'
start_date = '2000-01-01'
end_date = '2022-04-08'
vPrices = yf.download(symbols, start=start_date, end=end_date)['Close'].dropna()
vReturns = np.log(vPrices / vPrices.shift(1)).dropna() *100
vNpReturns = np.array(vReturns.transpose())
vTrain = vNpReturns[:,:int(len(vNpReturns[0,:]) * 0.8)]
vTest = vNpReturns[:,int(len(vNpReturns[0,:]) * 0.8):]



vNpY = np.zeros((6, vNpReturns.shape[1]))
vNpY[:3,:] = vNpReturns**2
vNpY[3,:] = vNpReturns[0,:] * vNpReturns[1,:]
vNpY[4,:] = vNpReturns[0,:] * vNpReturns[2,:]
vNpY[5,:] = vNpReturns[1,:] * vNpReturns[2,:]

# Define the window size
window_size = 12

# Create the input and output data
X = []
y = []
for i in range(window_size, len(vNpReturns[0])):
    X.append(vNpY[:, i-window_size:i] )
    y.append(vNpY[:, i:i+1]  )

# Convert the data to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
split = int(0.80*len(X))-2
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]


# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
y_train = scaler.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
y_test = scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)

# Define the number of features
n_features = X_train.shape[1]
dDropoutProbablity = 0.5
# Define the model
model = Sequential()

# Encoder
model.add(layers.LSTM(512, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(layers.LeakyReLU())
model.add(Dropout(dDropoutProbablity))
# model.add(layers.LSTM(512, return_sequences=True))
# model.add(layers.LeakyReLU())
# model.add(Dropout(dDropoutProbablity))
model.add(layers.LSTM(512))
model.add(layers.LeakyReLU())
model.add(Dropout(dDropoutProbablity))
model.add(layers.Dense(6))
model.add(layers.LeakyReLU())

# Repeat vector for the decoder
# model.add(RepeatVector(1))  # Set to 1 to match y_train shape

# # Decoder
# model.add(LSTM(256, return_sequences=True))
# model.add(Dropout(dDropoutProbablity))  # MC Dropout
# model.add(LSTM(256, return_sequences=True))
# model.add(Dropout(dDropoutProbablity))  # MC Dropout
# model.add(LSTM(256, return_sequences=True))
# model.add(Dropout(dDropoutProbablity))  # MC Dropout
# model.add(TimeDistributed(Dense(6)))  # Adjust to match y_train features
# model.add(Dense(6, activation='relu'))
# Compile the model
#opt = Adam(learning_rate=0.001, clipvalue=0.5)
model.compile(optimizer=Adam(), loss='mse')


# early_stopping = EarlyStopping(monitor='val_loss', patience=1500, restore_best_weights=True)

# Fit the model
#model.fit(X_train, y_train, epochs=100, validation_split=0.25)
history = model.fit(X_train, y_train, epochs=80, batch_size = split, validation_split = 0.2, shuffle = False)

# Number of Monte Carlo samples
n_samples = 300

# Predictions placeholder
predictions = []

# Running the model with dropout turned on
for i in range(n_samples):
    y_pred = model(X_test, training=True)  # dropout is turned on by setting training=True
    predictions.append(y_pred) 

# Average over the MC samples
predictions_mean = np.mean(predictions, axis=0)

# Reshape predictions for inverse transform
predictions_reshaped = predictions_mean.reshape(-1, predictions_mean.shape[-1])

# Inverse transform the predictions
predictions_descaled = scaler.inverse_transform(predictions_reshaped)

# Reshape back to original
predictions_descaled = predictions_descaled.reshape(predictions_mean.shape)


#X_test = scaler.inverse_transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
y_test = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
y_pred_final = predictions_descaled

# Calculate the prediction MSE
for i in range(6):
    plt.plot(y_test[:,i], label="Actual")
    plt.plot(vTest[i,:]**2)
    plt.plot(y_pred_final[:,i], label="Predicted", alpha = 0.7)
    plt.legend()
    plt.show()
    print(mean_squared_error(y_test[:,i], y_pred_final[:,i]))


# Plot training & validation loss values
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()