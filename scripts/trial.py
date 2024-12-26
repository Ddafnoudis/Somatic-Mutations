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
def grid_search(X_train_dl, X_test_dl,y_train_dl, y_test_dl, X_val_dl, 
                y_val_dl, target_classes_dl, seed, param_grid, epochs):
   
    # Define the size of the features
    feature_size = len(X_train_dl.columns)

    # Number of classes
    num_classes = len(np.unique(target_classes_dl))

    # One-hot encode the target values
    y_train_dl_reshaped = np.eye(num_classes)[y_train_dl]
    y_test_dl_reshaped = np.eye(num_classes)[y_test_dl]
    y_val_dl_reshaped = np.eye(num_classes)[y_val_dl]

    # Remove the extra dimension
    y_train_dl_reshaped = np.squeeze(y_train_dl_reshaped)
    y_test_dl_reshaped = np.squeeze(y_test_dl_reshaped)
    y_val_dl_reshaped = np.squeeze(y_val_dl_reshaped)


    best_score = 0
    best_params = None

    # Grid search over hyperparameters
    for dropout_rate in param_grid['dropout_rate']:
        for learning_rate in param_grid['learning_rate']:
            for batch_size in param_grid['batch_size']:
                # Build the model
                model = keras.Sequential([
                    layers.Input(shape=(feature_size, )),
                    layers.Dense(32, activation="relu", kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
                    layers.Dropout(dropout_rate),
                    layers.Dense(64, activation="relu", kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
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
                model.fit(X_train_dl, y_train_dl_reshaped, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True)

                # Predict on the test set
                y_pred = model.predict(X_test_dl)
                y_pred_labels = np.argmax(y_pred, axis=1)

                # Compute balanced accuracy
                score = balanced_accuracy_score(y_test_dl, y_pred_labels)
                print(f"Params: dropout_rate={dropout_rate}, learning_rate={learning_rate}, batch_size={batch_size} Balanced Accuracy={score}")

                # Update best parameters
                if score > best_score:
                    best_score = score
                    best_params = {'dropout_rate': dropout_rate, 'learning_rate': learning_rate, 'batch_size': batch_size}

    print(f"Best Balanced Accuracy: {best_score}")
    print(f"Best Parameters: {best_params}")


    return best_params


if __name__ == "__main__":
    grid_search()