import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import math

# Your API key
key="7354f94be486111c741e613aec63c109a6b301e6"

# Specify the stock symbol and the start and end dates
symbol = 'FTNT'
start_date = '2002-01-01'
end_date = '2023-12-31'

# Fetching the data
df = pdr.get_data_tiingo(symbol, start=start_date, end=end_date, api_key=key)
df = df.reset_index()

# Feature Engineering
# Here you can add more features or technical indicators
df['average_price'] = (df['high'] + df['low']) / 2
selected_columns = df[['close', 'average_price']]

# Scaling
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(selected_columns)

# Splitting the dataset
training_size = int(len(scaled_data) * 0.80)
train_data, test_data = scaled_data[0:training_size, :], scaled_data[training_size:len(scaled_data), :]

# Creating the dataset
def create_dataset(dataset, time_step=100):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), :] 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 2)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 2)

# Create the Stacked LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 2)))
model.add(Dropout(0.2)) # Dropout to prevent overfitting
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Model summary
model.summary()

# Training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=2)

# Prediction
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to original scale
train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((len(train_predict), 1))), axis=1))[:,0]
test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((len(test_predict), 1))), axis=1))[:,0]

# Calculate RMSE
train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))

print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")

# Plotting
plt.plot(scaler.inverse_transform(scaled_data)[:,0]) # Actual
# len(test_predict)=603,train_predict.shape(2714,)



# Create the arrays
array1 = np.nan*np.zeros((100,len(train_predict)))
array2 = train_predict.reshape(-1, 1)
array3 = np.nan*np.zeros((len(test_predict),))

# Print the shapes of each array
print("Shape of array1:", array1.shape)
print("Shape of array2:", array2.shape)
print("Shape of array3:", array3.shape)

np.concatenate((np.nan*np.zeros((100,len(train_predict))), train_predict.reshape(-1, 1), np.nan*np.zeros((len(test_predict),))))

# plt.plot(np.concatenate((np.nan*np.zeros((100,len(train_predict))), train_predict, np.nan*np.zeros((len(test_predict),)))), color='red') # Train
plt.plot(np.concatenate((np.nan*np.zeros((len(train_predict)+(100*2)+1,)), test_predict.reshape(-1,1))), color='green') # Test
plt.show()
