"""
Develop a Multilayer Perceptron for classification task 
"""
import os
import random
import sys
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras import callbacks
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, classification_report


if not sys.stdout.encoding == 'UTF-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Ensure TensorFlow operations are deterministic
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU if necessary for exact reproducibility



def multilayer_perceptron(X_train_dl, X_test_dl, y_train_dl, epochs,
                          y_test_dl, X_val_dl, y_val_dl, 
                          num_classes, target_names, seed, best_params):
    """
    Classification process using Multilayer Perceptron
    """
    # Define the size of the features
    feature_size = len(X_train_dl.columns)

    # Define the Neural Network structure using layers and dropout rate
    sequential_model = keras.Sequential(
        [
        layers.Input(shape=(feature_size, )),
        layers.Dense(best_params["neurons_1st_layer"], activation="relu", kernel_initializer=keras.initializers.GlorotNormal(seed=seed)),
        layers.Dropout(best_params['dropout_rate']),  
        layers.Dense(best_params["neurons_2nd_layer"], activation="relu", kernel_initializer=keras.initializers.GlorotNormal(seed=seed)),
        layers.Dropout(best_params['dropout_rate']),
        layers.Dense(num_classes, activation="softmax", kernel_initializer=keras.initializers.GlorotNormal(seed=seed))
        ]
    )

    # Print how the model looks like
    print(sequential_model.summary())
    # Define the metrics
    metrics_ = [
      keras.metrics.CategoricalCrossentropy(name='categorical_crossentropy'),  # same as model's loss
      keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    # Define the optimizer with specified learning rate
    optimizer = keras.optimizers.Adam(best_params['learning_rate'])

    # Compile the model
    sequential_model.compile(optimizer=optimizer, 
                             loss='categorical_crossentropy',
                             metrics=metrics_)
    
    # Define the early stopping
    earlystopping = callbacks.EarlyStopping(
            # Quantity to be monitored.
            monitor="val_loss",
            # Stop training if the quantity has stopped decreasing.
            mode="min",
            # Number of epochs with no improvement
            patience=5,
            # Restore the best weights
            restore_best_weights=True)
    
    # Fit the model
    sequential_model_fit = sequential_model.fit(x=X_train_dl, y=y_train_dl,
                                                epochs=epochs, batch_size=best_params['batch_size'],
                                                validation_data=(X_val_dl, y_val_dl),
                                                verbose=2,
                                                callbacks=[earlystopping])

    
    # Evaluate the model
    evaluation_results = sequential_model.evaluate(X_test_dl, y_test_dl, verbose=0)
    print("Evaluation Results:", evaluation_results)

    # Make predictions
    y_pred = np.argmax(sequential_model.predict(X_test_dl), axis=-1)

    # Print accuracy, confusion matrix, classification report
    y_test_labels = np.argmax(y_test_dl, axis=-1)
    print("Accuracy:", accuracy_score(y_test_labels, y_pred), '\n')
    print("Balanced Accuracy", balanced_accuracy_score(y_test_labels, y_pred), '\n')
    print("Confusion Matrix:\n", confusion_matrix(y_test_labels, y_pred), '\n')
    print("Classification Report:\n", classification_report(y_test_labels, y_pred, target_names=target_names), '\n')

    # Plot the performance of Sequencial Model per epoch
    # Define epochs as a list
    epochs_list = list(range(1, best_params["epochs"] + 1)) 

    # Define the metrics for the performance of the model
    metrics_list = ['loss', 'val_loss', 'categorical_accuracy', 
            'val_categorical_accuracy', 'precision', 'val_precision', 
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

    return sequential_model, X_val_dl, y_val_dl, target_names


def validate_multilayer_perceptron(X_val_dl, y_val_dl_reshaped, sequential_model, target_classes_dl):
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
    y_val_labels = np.argmax(y_val_dl_reshaped, axis=-1)
    val_accuracy = accuracy_score(y_val_labels, y_pred)
    val_balanced_accuracy = balanced_accuracy_score(y_val_labels, y_pred)
    val_confusion_matrix = confusion_matrix(y_val_labels, y_pred)
    val_classification_report = classification_report(y_val_labels, y_pred, target_names=target_classes_dl)

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
