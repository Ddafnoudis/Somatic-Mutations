"""
A scripts that searches the best hyperparameters for a 
multilayer perceptron model.
"""
import os
import keras
import random
import numpy as np
from typing import Dict
import tensorflow as tf
from keras import layers
from keras import callbacks
from sklearn.metrics import balanced_accuracy_score


# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Ensure TensorFlow operations are deterministic
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU if necessary for exact reproducibility



def grid_search(X_train_dl, X_val_dl, y_train_dl, y_val_dl, epochs, num_classes, param_grid, seed) -> Dict[str, float]:
    """
    Create a for loop to iterate over the hyperparameters
    and return the best parameters.
    """
    # Define the size of the features
    feature_size = len(X_train_dl.columns)

    # Initialize best score and best parameters
    best_score = 0
    best_params = None

    # Iterate over the parameters
    for dropout_rate in param_grid['dropout_rate']:
        for learning_rate in param_grid['learning_rate']:
            for batch_size in param_grid['batch_size']:
                for neurons_1st in param_grid['neurons_1st_layer']:
                    for neurons_2nd in param_grid['neurons_2nd_layer']:
                        # Ensure the second layer has twice the number of neurons in the first layer
                        if neurons_2nd == neurons_1st * 2:
                            # Build the model
                            model = keras.Sequential([
                                layers.Input(shape=(feature_size, )),
                                layers.Dense(neurons_1st, activation="relu", kernel_initializer=keras.initializers.he_normal(seed=seed)),
                                layers.Dropout(dropout_rate),
                                layers.Dense(neurons_2nd, activation="relu", kernel_initializer=keras.initializers.he_normal(seed=seed)),
                                layers.Dropout(dropout_rate),
                                layers.Dense(num_classes, activation="softmax", kernel_initializer=keras.initializers.he_normal(seed=seed))
                            ])

                            # Define the metrics
                            metrics_ = [
                                keras.metrics.CategoricalCrossentropy(name='categorical_crossentropy'),  # same as model's loss
                                keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None),
                                keras.metrics.Precision(name='precision'),
                                keras.metrics.Recall(name='recall'),
                                keras.metrics.AUC(name='auc'),
                                keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
                            ]

                            # Compile the model
                            model.compile(
                                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                                loss='categorical_crossentropy',
                                metrics=metrics_
                            )
                            
                            # Define the early stopping
                            earlystopping = callbacks.EarlyStopping(
                                    monitor="val_loss",
                                    mode="min",
                                    patience=5,
                                    restore_best_weights=True)

                            # Train the model
                            model.fit(X_train_dl, y_train_dl, 
                                      epochs=epochs, batch_size=batch_size, 
                                      validation_data=(X_val_dl, y_val_dl),
                                      verbose=2,  
                                      callbacks=[earlystopping])

                            # Predict on the validation set
                            y_pred = model.predict(X_val_dl)
                            y_val_labels = np.argmax(y_val_dl, axis=1)
                            y_pred_labels = np.argmax(y_pred, axis=1)    

                            # Compute the balanced accuracy
                            score = balanced_accuracy_score(y_val_labels, y_pred_labels)
                            print(f"Params:\ndropout_rate={dropout_rate},\nlearning_rate={learning_rate},\nbatch_size={batch_size},\nBalanced Accuracy={score},\nepochs={epochs},\nneurons_1st_layer={neurons_1st},\nneurons_2nd_layer={neurons_2nd}")

                            # Update best parameters
                            if score > best_score:
                                best_score = score
                                best_params = {'dropout_rate': dropout_rate, 
                                               'learning_rate': learning_rate, 
                                               'batch_size': batch_size,
                                               "epochs": epochs,
                                               'neurons_1st_layer': neurons_1st,
                                               'neurons_2nd_layer': neurons_2nd}

    return best_params


if __name__ == "__main__":
    grid_search()
