import os
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from itertools import product
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

symbols = ['^GSPC', 'GOOG', 'MSFT']
symbols = ['^AEX', '^BFX', '^GDAXI']
start_date = '2010-01-01'
start_date = '2000-01-01'
end_date = '2022-04-08'
vPrices = yf.download(symbols, start=start_date, end=end_date)['Close'].dropna()
vReturns = vPrices.pct_change().dropna()
vReturns = np.log(vPrices / vPrices.shift(1)).dropna() *100
vNpReturns = np.array(vReturns.transpose())
vTrain = vNpReturns[:,:int(len(vNpReturns[0,:]) * 0.8)]
vTest = vNpReturns[:,int(len(vNpReturns[0,:]) * 0.8):]



vNpY = np.zeros((6, vNpReturns.shape[1]))
vNpY[:3, :] = vNpReturns**2
vNpY[3, :] = vNpReturns[0, :] * vNpReturns[1, :]
vNpY[4, :] = vNpReturns[0, :] * vNpReturns[2, :]
vNpY[5, :] = vNpReturns[1, :] * vNpReturns[2, :]

# Define the hyperparameter search space
window_sizes = list(range(1, 20))
layer_sizes = [16,32, 64, 128, 256, 512]
num_layers = [1, 2, 3, 4, 5]
param_space = list(product(window_sizes, num_layers, layer_sizes))


best_loss = float('inf')
best_model = None
vVal_loss = [] 
iIteration = 0
for params in param_space:
    iIteration += 1 
    window_size, num_layers, layer_size = params

    # Create the input and output data
    X = []
    y = []
    for i in range(window_size, len(vNpReturns[0])):
        X.append(vNpY[:, i-window_size:i])
        y.append(vNpY[:, i:i+1])

    # Convert the data to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split the data into training, validation, and test sets
    split_train = int(0.55 * len(X))
    split_val = int(0.8 * len(X))
    X_train, y_train = X[:split_train], y[:split_train]
    X_val, y_val = X[split_train:split_val], y[split_train:split_val]
    X_test, y_test = X[split_val:], y[split_val:]

    # Scale the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_train = scaler.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
    y_val = scaler.transform(y_val.reshape(-1, y_val.shape[-1])).reshape(y_val.shape)
    y_test = scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)

    def build_model(hidden_sizes):
        model = keras.Sequential()
        model.add(layers.LSTM(hidden_sizes[0], return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(layers.LeakyReLU())
        for size in hidden_sizes[1:]:
            model.add(layers.LSTM(size, return_sequences=True))
            model.add(layers.LeakyReLU())
        model.add(layers.LSTM(hidden_sizes[-1]))
        model.add(layers.Dense(6))
        model.add(layers.LeakyReLU())
        model.compile(loss='mse', optimizer='adam')
        return model
    
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "CV/cross_validation_" + str(iIteration) + "_model_weights.h5",
    monitor="val_loss",
    save_best_only=True,
    mode="min",)

    early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

    model = build_model([layer_size] * num_layers)
    history = model.fit(X_train, y_train, epochs=2000, batch_size=split_train, validation_data=(X_val, y_val), shuffle=False, verbose=0, callbacks=[checkpoint_callback, early_stopping])
    model.load_weights("CV/cross_validation_" + str(iIteration) + "_model_weights.h5")
    # val_loss = model.evaluate(X_val, y_val)
    val_loss = model.evaluate(scaler.inverse_transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape), scaler.inverse_transform(y_val.reshape(-1, y_val.shape[-1])).reshape(y_val.shape))
    print('Param Space : ', params)
    print('Iteration : ', iIteration)
    print('Validation Loss : ', val_loss)
    vVal_loss.append(val_loss)
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = model
    

# Evaluate the best model on the test set
test_loss = best_model.evaluate(X_test, y_test)
print("Best model test loss:", test_loss)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

X_test = scaler.inverse_transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
y_test = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[-1])).reshape(y_pred.shape)



df = pd.DataFrame({'Param Space': param_space, 'Val loss': vVal_loss})

# Plot the predicted and actual values
for i in range(6):
    plt.plot(y_test[:, i], label="Actual")
    plt.plot(y_pred[:, i], label="Predicted", alpha=0.7)
    plt.legend()
    plt.show()
    print(mean_squared_error(y_test[:, i], y_pred[:, i]))

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# array1 = vVal_loss
# array2 = param_space

# # #Save Arrays to a file
# # with open('arrays.pkl', 'wb') as f:
# #     pickle.dump((array1, array2), f)

# # Load arrays from the file
# with open('arrays.pkl', 'rb') as f:
#     loaded_array1, loaded_array2 = pickle.load(f)