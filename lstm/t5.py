import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import mean_squared_error
import math
"""
2023-11-12: 
1) adding more features 12_EMA
2) add LSTM+Cov1D ( CNN of 1D, although CNN for spaital , in this case 1D for serial)
3) in real mode, change the epochs=10 to 100 or bigger ..

@todo
1) plt not work
2) run time error
 2714 603
Traceback (most recent call last):
  File "/home/test/testscript/./aa/ml/t5.py", line 108, in <module>
    train_predict = scaler.inverse_transform(train_predict).reshape(-1,1)
  File "/home/test/.local/lib/python3.9/site-packages/sklearn/preprocessing/_data.py", line 548, in inverse_transform
    X -= self.min_
ValueError: non-broadcastable output operand with shape (2714,1) doesn't match the broadcast shape (2714,4)
"""
# Your API key
key = "7354f94be486111c741e613aec63c109a6b301e6"

# Specify the stock symbol and the start and end dates
symbol = 'FTNT'
start_date = '2002-01-01'
end_date = '2023-12-31'

# Fetching the data
df = pdr.get_data_tiingo(symbol, start=start_date, end=end_date, api_key=key)
df = df.reset_index()

# Feature Engineering
# Adding more features or technical indicators

# Calculate MACD
df['12_EMA'] = df['close'].ewm(span=12, adjust=False).mean()
df['26_EMA'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['12_EMA'] - df['26_EMA']
df['average_price'] = (df['high'] + df['low']) / 2

# Calculate RSI
delta = df['close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14, min_periods=1).mean()
avg_loss = loss.rolling(window=14, min_periods=1).mean()

rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))
print (df.head())

# Select relevant columns for scaling
selected_columns = df[['close', 'average_price', 'MACD', 'RSI']]

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
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
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)  # 4 features now
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 4)

# Create the Stacked LSTM with Conv1D model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_step, 4)))  # 4 features now
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Model summary
model.summary()

# Training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=64, verbose=2)
# Prediction
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

print (len(train_predict),len(test_predict))
# Inverse transform to original scale
train_predict = scaler.inverse_transform(train_predict).reshape(-1,1)
test_predict = scaler.inverse_transform(test_predict).reshape(-1,1)

# Calculate RMSE
train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))

print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")

# Plotting
plt.plot(scaler.inverse_transform(scaled_data)[:, 0])  # Actual
plt.plot(np.concatenate((np.nan * np.zeros((100,)), train_predict, np.nan * np.zeros((len(test_predict),)))))
plt.plot(np.concatenate((np.nan * np.zeros((len(train_predict) + (100 * 2) + 1,)), test_predict)))
plt.show(block=True)
