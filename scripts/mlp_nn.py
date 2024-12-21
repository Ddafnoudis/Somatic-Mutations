"""
Develop a Multilayer Perceptron for classification task 
"""
import os
import sys
import keras
import numpy as np
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
tf.random.set_seed(42)

def multilayer_perceptron(feat_dl, tar_dl, target_classes_dl, seed):
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
    epochs = 30
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
    optimizer = keras.optimizers.Adam(learning_rate=0.004)

    # Compile the model
    sequential_model.compile(optimizer=optimizer, 
                             loss='categorical_crossentropy',
                             metrics=metrics_)
    # Fit the model
    sequential_model_fit = sequential_model.fit(x=X_train_dl, y=y_train_dl_reshaped,
                                                epochs=epochs, batch_size=batch_size,
                                                validation_data=(X_test_dl, y_test_dl_reshaped),
                                                validation_split = 0.25,
                                                verbose=2, shuffle=True)

    
    # Evaluate the model
    evaluation_results = sequential_model.evaluate(X_test_dl, y_test_dl_reshaped, verbose=0)
    print("Evaluation Results:", evaluation_results)

    # Make predictions
    y_pred = np.argmax(sequential_model.predict(X_test_dl), axis=-1)

    # Print accuracy, confusion matrix, classification report
    print("Accuracy:", accuracy_score(y_test_dl, y_pred), '\n')
    print("Balanced Accuracy", balanced_accuracy_score(y_test_dl, y_pred), '\n')
    print("Confusion Matrix:\n", confusion_matrix(y_test_dl, y_pred), '\n')
    print("Classification Report:\n", classification_report(y_test_dl, y_pred, target_names=target_classes_dl), '\n')

    # Plot the performance of Sequencial Model per epoch
    # Define epochs as a list
    epochs_list = list(range(1, len(metrics_) + 1)) 

    # Define the metrics for the performance of the model
    metrics_list = ['loss', 'val_loss', 'accuracy', 
            'val_accuracy', 'precision', 'val_precision', 
            'recall', 'val_recall', 'auc', 'val_auc']

    # Create an empty list to store the results
    results = []

    # Iterate over metrics
    for metric in metrics_list:
        lines = go.Scatter(x=epochs_list, y=sequential_model_fit.history[metric], mode='lines+markers', name=metric.replace('_', ' ').title())
        results.append(lines)
    # Display the titles
    layout = go.Layout(
        title='Multilayer Perceptron Performance',
        xaxis=dict(title='Epochs'),
        yaxis=dict(title='Metrics'),
        showlegend=True
    )
    # Unite the figure's information
    fig = go.Figure(data=results, layout=layout)
    # Show the figure
    fig.show()

    return sequential_model, X_val_dl, y_val_dl, y_val_dl_reshaped, target_classes_dl


def validate_multilayer_perceptron(X_val_dl, y_val_dl, y_val_dl_reshaped, sequential_model, target_classes_dl):
    """
    Validate the Multilayer Perceptron model using validation data
    and the sequential model you used in the training set
    """
    # Evaluate the model on the validation data
    evaluation_results = sequential_model.evaluate(X_val_dl, y_val_dl_reshaped, verbose=0)
    print(f" Evaluation results:\n {evaluation_results}\n")
    
    # Make predictions
    y_pred = np.argmax(sequential_model.predict(X_val_dl), axis=-1)

    # Calculate confusion matrix and classification report
    val_accuracy = accuracy_score(y_val_dl, y_pred)
    val_balanced_accuracy = balanced_accuracy_score(y_val_dl, y_pred)
    val_confusion_matrix = confusion_matrix(y_val_dl, y_pred)
    val_classification_report = classification_report(y_val_dl, y_pred, target_names=target_classes_dl)

    # Print validation accuracy, confusion matrix, and classification report
    print("Validation Accuracy:", val_accuracy, '\n')
    print("Balanced Accuracy", val_balanced_accuracy, '\n')
    print("Confusion Matrix:\n", val_confusion_matrix, '\n')
    print("Classification Report:\n", val_classification_report, '\n')

    # Save performance metrics to a text file
    # if folder doesn't exist makedir 
    if not os.path.exists("result_files/mlp_folder"):
        os.makedirs("result_files/mlp_folder")
    with open("result_files/mlp_folder/validation_results.txt", "w", encoding="utf-8") as file:
        file.write("Validation Accuracy: {}\n\n".format(val_accuracy))
        file.write("Validation Balanced Accuracy: {}\n\n".format(val_balanced_accuracy))
        file.write("Confusion Matrix:\n{}\n\n".format(val_confusion_matrix))
        file.write("Classification Report:\n{}\n".format(val_classification_report))

    return val_accuracy, val_confusion_matrix, val_classification_report


if __name__ == "__main__":
    multilayer_perceptron()
    validate_multilayer_perceptron()
