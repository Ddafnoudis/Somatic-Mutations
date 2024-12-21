"""
Optimize the learning for the MLP model
"""
import os
import sys
import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.utils import shuffle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, classification_report


if not sys.stdout.encoding == 'UTF-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)


def learning_rate_optimization(feat_dl, tar_dl, target_classes_dl, seed):
    """
    Classification process using Multilayer Perceptron
    """
    feat_dl = feat_dl.astype(str)

    # One hot encoding
    feat_dl = pd.get_dummies(feat_dl, drop_first=True, dtype=int)
    print(f"Feature shape for MLP training: {feat_dl.shape}")
    # Label Encoding
    lb = LabelEncoder()
    tar_dl_enc = lb.fit_transform(tar_dl)

    # Shuffle the data
    feat_dl, tar_dl_enc = shuffle(feat_dl, tar_dl_enc, random_state=seed)

    # Five fold stratification
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, test_index in skf.split(feat_dl, tar_dl_enc):
        X_train_dl, X_test_dl = feat_dl.iloc[train_index], feat_dl.iloc[test_index]
        y_train_dl, y_test_dl = tar_dl_enc[train_index], tar_dl_enc[test_index]
    # Define the validation set
    X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(X_train_dl, y_train_dl, test_size=0.25,  random_state=seed)

    # Define the number of classes
    num_classes = len(np.unique(target_classes_dl))
    # Reshape the target values
    y_train_dl_reshaped = np.eye(num_classes)[y_train_dl]
    # Reshape Test set
    y_test_dl_reshaped = np.eye(num_classes)[y_test_dl]
    # Reshape the validation target
    y_val_dl_reshaped = np.eye(num_classes)[y_val_dl]
    
    # Define the size of the features
    feature_size = len(feat_dl.columns)

    # Depine the dropout rate
    dropout_rate = 0.4
    # Define the epochs
    epochs = 100
    # Define the batch_size
    batch_size = 100
    
    
    # Define the Neural Network structure using layers and dropout rate
    sequential_model = keras.Sequential(
        [
        layers.Input(shape=(feature_size, )),
        layers.Dense(16, activation="relu"),
        layers.Dropout(dropout_rate),  
        layers.Dense(32, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(3, activation="softmax")
        ]
    )

    # Print how the model looks like
    print(sequential_model.summary())
    # Define the metrics
    metrics_ = [
      keras.metrics.BinaryCrossentropy(name='categorical_crossentropy'),  # same as model's loss
      keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    # Define the optimizer with specified learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # Compile the model
    sequential_model.compile(optimizer=optimizer, 
                             loss='categorical_crossentropy',
                             metrics=metrics_)
    
    history_model = sequential_model.fit(
        X_train_dl, y_train_dl_reshaped, 
        epochs=epochs,
        callbacks=[keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-3 * 10 ** (epoch / 30))]
    )

    # Define the number of epochs used
    epochs_range = np.arange(1, epochs + 1)

    plt.figure(figsize=(18, 8))

    # Plot loss
    plt.plot(
        epochs_range, 
        history_model.history['loss'], 
        label='Loss', lw=3
    )
    
    # Plot accuracy
    plt.plot(
        epochs_range, 
        history_model.history['categorical_accuracy'], 
        label='Accuracy', lw=3
    )

    # Plot learning rate (adjusting for the number of epochs)
    learning_rates = 1e-3 * (10 ** (epochs_range / 30))
    plt.plot(
        epochs_range, 
        learning_rates[:len(epochs_range)], 
        label='Learning rate', color='#000', lw=3, linestyle='--'
    )

    # Customize the plot
    plt.title('Evaluation metrics', size=20)
    plt.xlabel('Epoch', size=14)
    plt.ylabel('Value', size=14)
    plt.legend()
    plt.show()

    # Plot Learning Rate vs. Loss
    plt.figure(figsize=(10, 6))
    plt.semilogx(
        learning_rates[:len(history_model.history['loss'])], 
        history_model.history['loss'], 
        lw=3, color='#000'
    )
    plt.title('Learning rate vs. loss', size=20)
    plt.xlabel('Learning rate', size=14)
    plt.ylabel('Loss', size=14)
    plt.show()


if __name__ == "__main__":
    learning_rate_optimization()