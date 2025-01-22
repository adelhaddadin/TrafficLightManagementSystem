#!/usr/bin/env python
# model_training.py
# coding: utf-8
#Converted to .py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import json


def pad_lanes(X, max_lanes):
    """Pad lane dimension if different groups have different # of lanes."""
    num_samples, num_lanes, num_features = X.shape
    padded_X = np.zeros((num_samples, max_lanes, num_features))
    padded_X[:, :num_lanes, :] = X
    return padded_X


def generate_input(dataset, group_col, lane_col, feature_cols, target_col):
    """
    Create X, y for each group, shaped (time_steps, #lanes, #features).
    Return dict {group_name: (X, y)}.
    """
    group_sequences = {}

    max_lanes = dataset[lane_col].nunique()

    for group_name, group_data in dataset.groupby(group_col):
        if not group_data.index.is_monotonic_increasing:
            raise ValueError(
                f"Group '{group_name}' is not sorted by time index."
            )
        
        unique_lanes = group_data[lane_col].unique()

        hours = group_data.index.unique()  # if hour is the index
        hourly_data = []
        targets = []

        for hour in hours:
            hour_data = group_data.loc[hour]
            lane_features = hour_data.set_index(lane_col)[feature_cols].reindex(unique_lanes).values
            hourly_data.append(lane_features)
            target_val = hour_data[target_col].iloc[0] if isinstance(hour_data, pd.DataFrame) else hour_data[target_col]
            targets.append(target_val)

        hourly_data = np.array(hourly_data)  # shape: (num_hours, num_lanes, num_features)
        targets = np.array(targets)          # shape: (num_hours,)

        X = hourly_data[:-1]
        y = targets[1:]

        X_padded = pad_lanes(X, max_lanes)

        group_sequences[group_name] = (X_padded, y)

    return group_sequences


def split_data_chronologically(X, y, train_ratio=0.6, val_ratio=0.2):
    """Chronological split: train, val, test."""
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape):
    """
    Example LSTM model. 
    input_shape: (time_steps, max_lanes, num_features)
    """
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(32, activation='relu', return_sequences=True)(inputs)
    lstm_2   = LSTM(16, activation='relu')(lstm_out)
    dense_out = Dense(64, activation='relu')(lstm_2)
    dropout_1 = Dropout(0.2)(dense_out)
    outputs = Dense(1)(dropout_1)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def plot_training_metrics(group_name, history, test_loss, test_mae):
    """
    Plot and save training/validation loss and MAE for a specific group.
    """
    # Extract metrics
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']

    # Create the plots
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
    plt.title(f'{group_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(mae, label='Training MAE')
    plt.plot(val_mae, label='Validation MAE')
    plt.axhline(y=test_mae, color='r', linestyle='--', label='Test MAE')
    plt.title(f'{group_name} - MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()

    # Save the plot
    plot_file = f'{group_name}_training_plot.png'
    plt.savefig(plot_file)
    print(f"Training plot saved for {group_name} as '{plot_file}'")

    # Close the plot to free memory
    plt.close()


def train_group_model(group_name, X, y, epochs=10, batch_size=32):
    """
    Train a model for a single group. Return the trained model and test metrics.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_chronologically(X, y)

    model = build_model(input_shape=X_train.shape[1:])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Group {group_name} --> Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # Call the plotting function
    plot_training_metrics(group_name, history, test_loss, test_mae)
    return model, (test_loss, test_mae)


if __name__ == "__main__":
    # 1. Read final dataset
    data_path = r'final_data.csv'
    data = pd.read_csv(data_path)

    # Suppose 'hour' is the time index
    data.set_index('hour', inplace=True)

    # 2. Identify feature columns and target
    exclude_cols = ['id', 'group', 'group_label', 'target']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    target_col = 'target'
    group_col = 'group_label'
    lane_col = 'id'  # or 'lane_label' if needed

    # 3. Generate (X, y) sequences per group
    group_sequences = generate_input(
        dataset=data,
        group_col=group_col,
        lane_col=lane_col,
        feature_cols=feature_cols,
        target_col=target_col
    )

    # Save the sequences to a JSON file
    try:
        sequences_json = {group: {'X': X.tolist(), 'y': y.tolist()} for group, (X, y) in group_sequences.items()}
        output_file = r'C:\Users\POWER TECH\Downloads\group_sequences.json'
        with open(output_file, 'w') as file:
            json.dump(sequences_json, file)
        print(f"Sequences saved as '{output_file}'")
    except Exception as e:
        print(f"Error saving sequences: {e}")

    # 4. Train a model per group 
    trained_models = {}
    for group_name, (X, y) in group_sequences.items():
        model, metrics = train_group_model(group_name, X, y, epochs=10, batch_size=16)
        trained_models[group_name] = model
        model.save(f"model_{group_name}.h5")

