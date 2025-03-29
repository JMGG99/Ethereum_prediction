# Ethereum_prediction
# Ethereum Price Prediction using GRU

## Overview
This project trains a deep learning model using Gated Recurrent Units (GRU) to predict Ethereum's closing price based on historical data. The model is built with TensorFlow and Keras, employing multiple GRU layers, dropout regularization, and early stopping for optimal performance.

## Features
- **Data Preprocessing**: Normalization and windowed time-series transformation.
- **Deep Learning Model**: Three GRU layers with different configurations.
- **Training and Validation**: Split dataset with early stopping to prevent overfitting.
- **Evaluation and Visualization**: Training loss and validation loss graphs to analyze performance.

## Data
The dataset includes historical Ethereum price data with features:
- Open, High, Low, Close
- Volume
- Date (processed into time-series windows)

## Model Architecture
```plaintext
    GRU(128, return_sequences=True, recurrent_dropout=0.1, input_shape=(window_size, len(numerical_cols))),
    Dropout(0.2),

    GRU(64, recurrent_dropout=0.1, return_sequences=True),
    Dropout(0.1),


    GRU(64, recurrent_dropout=0.1, return_sequences=False),
    Dropout(0.1),


    Dense(1)
```
Optimizer: Adam
Loss Function: Mean Squared Error (MSE)


## Usage
1. **Prepare Data**: Ensure dataset is formatted correctly.
2. **Train Model**: Run the script to train the GRU network.
3. **Evaluate & Predict**: Use trained model for price forecasting.

```bash
python Coin_Ethe_Time_Series.py
```

## Results
- The training and validation loss graphs indicate the model's learning progress.
- Model performance can be further optimized with more data and hyperparameter tuning.

## Future Work
- Experiment with LSTM and Transformer models.
- Improve feature engineering.
- Deploy the model as an API for real-time predictions.

## Contributions
Feel free to fork this repository and improve the model! Open a pull request with your contributions.



