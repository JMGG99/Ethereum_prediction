import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping



data = pd.read_csv("coin_Ethereum.csv") #we read our dataset
data["Date"] = pd.to_datetime(data["Date"]) #We convert date strinngs to datetime object to work better
data.drop(columns=["SNo", "Name", "Symbol", "Marketcap"], inplace=True) #We drop columns that we don't consider necessary to predict the target value "close"

scaler = StandardScaler()
numerical_cols = ["High", "Low", "Open", "Close", "Volume"]
data[numerical_cols] = scaler.fit_transform(data[numerical_cols]) #As we have a big difference between final values and initial values it would be better if we apply standar scaler
#print(data.head())

#Lets generate our time series
data_values = data[numerical_cols].values
target_values = data["Close"].values 

window_size = 7
batch_size = 64

train_data, val_data, train_target, val_target = train_test_split(
    data_values, target_values, test_size=0.1, shuffle=False
)

train_generator = TimeseriesGenerator(train_data, train_target, length=window_size, batch_size=batch_size)
val_generator = TimeseriesGenerator(val_data, val_target, length=window_size, batch_size=batch_size )

#Lets create our model
model = Sequential([
    GRU(128, return_sequences=True, recurrent_dropout=0.1, input_shape=(window_size, len(numerical_cols))),
    Dropout(0.2),

    GRU(64, recurrent_dropout=0.1, return_sequences=True),
    Dropout(0.1),


    GRU(64, recurrent_dropout=0.1, return_sequences=False),
    Dropout(0.1),


    Dense(1)
])

#model.summary()

model.compile(optimizer='adam', loss='mse')

#Lets train our model
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    train_generator, 
    validation_data = val_generator,
    epochs=80, 
    callbacks=[early_stopping]
)
"""
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()"
"""

