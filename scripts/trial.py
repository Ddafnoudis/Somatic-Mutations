import os
import random
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import balanced_accuracy_score


# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# # Ensure TensorFlow operations are deterministic
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU if necessary for exact reproducibility



# Define the grid search function
def grid_search(X_train_dl, X_test_dl, y_train_dl, y_test_dl, num_classes, seed, param_grid):
   
    # Define the size of the features
    feature_size = len(X_train_dl.columns)


    best_score = 0
    best_params = None

    # Grid search over hyperparameters
    for dropout_rate in param_grid['dropout_rate']:
        for learning_rate in param_grid['learning_rate']:
            for batch_size in param_grid['batch_size']:
                for epochs in param_grid['epochs']:
                    for neurons_1st in param_grid['neurons_1st_layer']:
                        for neurons_2nd in param_grid['neurons_2nd_layer']:
                            # Build the model
                            model = keras.Sequential([
                                layers.Input(shape=(feature_size, )),
                                layers.Dense(neurons_1st, activation="relu", kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
                                layers.Dropout(dropout_rate),
                                layers.Dense(neurons_2nd, activation="relu", kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
                                layers.Dropout(dropout_rate),
                                layers.Dense(num_classes, activation="softmax", kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))
                            ])

                            # Compile the model
                            model.compile(
                                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                                loss='categorical_crossentropy',
                                metrics=['accuracy']
                            )

                            # Train the model
                            model.fit(X_train_dl, y_train_dl, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True)

                            # Predict on the test set
                            y_pred = model.predict(X_test_dl)
                            y_test_labels = np.argmax(y_test_dl, axis=1)  # True labels
                            y_pred_labels = np.argmax(y_pred, axis=1)    # Predicted labels

                            # Compute balanced accuracy
                            score = balanced_accuracy_score(y_test_labels, y_pred_labels)
                            print(f"Params:
                                  \n dropout_rate={dropout_rate}, 
                                  \nlearning_rate={learning_rate},
                                  \nbatch_size={batch_size},
                                  \nBalanced Accuracy={score},
                                  \nepochs={epochs}, 
                                  \nneurons_1st_layer={neurons_1st}, 
                                  \nneurons_2nd_layer={neurons_2nd}")

                            # Update best parameters
                            if score > best_score:
                                best_score = score
                                best_params = {'dropout_rate': dropout_rate, 
                                               'learning_rate': learning_rate, 
                                               'batch_size': batch_size,
                                               'epochs': epochs,
                                               'neurons_1st_layer': neurons_1st,
                                               'neurons_2nd_layer': neurons_2nd}

    print(f"Best Balanced Accuracy: {best_score}")
    print(f"Best Parameters: {best_params}")


    return best_params


if __name__ == "__main__":
    grid_search()