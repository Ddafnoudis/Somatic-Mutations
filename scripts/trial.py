"""
A scripts that searches the best hyperparameters for a 
multilayer perceptron model.
"""
import os
import time
import keras
import random
import itertools
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import layers
from typing import Dict, Any
from datetime import timedelta
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, classification_report, make_scorer, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42) 
tf.random.set_seed(42)

# Ensure TensorFlow operations are deterministic
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# Disable GPU if necessary for exact reproducibility
os.environ['CUDA_VISIBLE_DEVICES'] = '' 


def optimization(X_train, X_val, y_train, y_val, 
                 search_space, epochs, feature_size,
                 seed, best_params_path, hidden_layer_options,
                 num_classes, best_model_path)-> Dict[str, int] | Any:
    """
    Perform hyperparameter optimization for a 
    multilayer perceptron (MLP) using Bayesian search.

    The search space defines hidden_layer_sizes as an integer 
    between 0 and 5. This integer represents the index in 
    the HIDDEN_LAYER_OPTIONS list. 
    Adapt the search_space_mlp_ (with tuple-style hidden_layer_sizes) 
    to fit the flattened integer-based space inside the MLPWrapper.
    Convert your config into an index-based categorical and 
    map it to the actual tuples inside the wrapper.
    
    Args:
        X_train (array-like): Training feature data
        X_val (array-like): Validation feature data
        y_train (array-like): Training target labels
        y_val (array-like): Validation target labels
        search_space (dict): Hyperparameter search space for Bayesian optimization
        epochs (int): Number of training epochs
        feature_size (int): Number of input features
        seed (int): Random seed for reproducibility
        best_params_path (str): File path to save best hyperparameters
        hidden_layer_options (list): List of possible hidden layer configurations
        num_classes (int): Number of output classes
    
    Returns:
        Dict: A tuple containing the best hyperparameters
        HDF5 format: The best MLP model
    """
    # Start time (calculate duration)
    start_time = time.perf_counter()
    print(f"Start time: {start_time}")

    class KerasMLP(BaseEstimator, ClassifierMixin):
        def __init__(self, hidden_layer_sizes=None, activation='relu', 
                    alpha=None, learning_rate=None, 
                    batch_size=32, dropout_rate=0.1,
                    hidden_layer_options=None, seed=42):
            # Initialize Hyperparameters
            self.hidden_layer_sizes: int = hidden_layer_sizes # Index of the hidden layer sizes
            self.hidden_layer_options: list[tuple] = hidden_layer_options 
            self.activation: str = activation
            self.alpha: float = alpha
            self.learning_rate: float = learning_rate
            self.batch_size: int = batch_size
            self.seed: int = seed
            self.dropout_rate: float = dropout_rate
            # Delay model creation
            self.model = None

        def _build_model(self):
            # Sequential model
            model = keras.Sequential()
            # Input size
            model.add(layers.Input(shape=(feature_size, )))

            # Use the selected hidden layer configuration
            hidden_layers: tuple = self.hidden_layer_options[self.hidden_layer_sizes]
            # Add hidden layers
            for units in hidden_layers:
                print(f"Adding layer with {units} units\n")
                # Dense layer with given units and activation
                model.add(layers.Dense(units,
                                       activation=self.activation,
                                       kernel_regularizer=keras.regularizers.l2(self.alpha),
                                       kernel_initializer=keras.initializers.HeNormal(seed=self.seed)))
                # Dropout layer
                model.add(layers.Dropout(self.dropout_rate))
            
            # Output layer
            model.add(layers.Dense(num_classes, activation='softmax',
                                   kernel_initializer=keras.initializers.HeNormal(seed=self.seed)))
            
            # Compile the model
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                       metrics=[
                       # Calculates how often predictions match integer labels
                         keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                         # Calculates the crossentropy loss between the labels and predictions
                         keras.metrics.SparseCategoricalCrossentropy(name="loss"),
                     ],
                     loss="sparse_categorical_crossentropy")
            
            return model
               
        # Fit method for the model
        def fit(self, X, y):
            self.model = self._build_model()
            # Enhanced callbacks
            callbacks = [
                TrainingPlotter(validation_data=(X_val, y_val)),
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                ),
                # keras.callbacks.ModelCheckpoint(
                #     "best_model_temp.keras",  # Temporary file
                #     save_best_only=True,
                #     monitor='val_loss'
                # )
                ]
            
            # Modified fit method with callback
            self.model.fit(
                X, y, 
                batch_size=self.batch_size, 
                epochs=epochs, 
                verbose=1,
                class_weight=class_weight_dict,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
            )
            return self
        
        # Predict method for the model
        def predict(self, X):
            # Predict
            probs = self.model.predict(X)
            return np.argmax(probs, axis=-1)
                

        def score(self, X, y):
            y_pred = np.argmax(self.predict(X), axis=-1)
            return balanced_accuracy_score(y, y_pred)
    
    class TrainingPlotter(keras.callbacks.Callback):
        """
        Persistent live plotter for training and validation metrics across epochs and hyperparameter trials.
        """
        # Static/global counter for hyperparameter trial ID
        _global_run_id = 0  

        def __init__(self, validation_data=None):
            """
            Initialize the TrainingPlotter callback with validation data and tracking metrics.
    
            Args:
                validation_data (tuple, optional): A tuple of (X_val, y_val) for validation tracking. Defaults to None.
    
            Attributes:
                validation_data (tuple): Validation dataset for tracking model performance.
                metrics (dict): Dictionary to track training and validation loss and accuracy.
                best_epoch (int): Epoch with the best validation performance.
                best_metrics (dict): Metrics from the best performing epoch.
                run_id (int): Unique identifier for the current training run.
            """
            super().__init__()
            self.validation_data = validation_data
            self.metrics = {'loss': {'train': [], 'val': []}, 'accuracy': {'train': [], 'val': []}}
            self.best_epoch = 0
            self.best_metrics = {}
            # Unique identifier for the current training run.
            self.run_id = TrainingPlotter._global_run_id
            TrainingPlotter._global_run_id += 1

            # Create the plot
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
            # Enable interactive mode
            plt.ion()  

        def on_epoch_end(self, epoch, logs=None):
            # Store metrics
            for metric in ['loss', 'accuracy']:
                self.metrics[metric]['train'].append(logs.get(metric))
                self.metrics[metric]['val'].append(logs.get(f"val_{metric}"))

            # Track best val loss
            val_loss = logs.get('val_loss')
            if val_loss <= min(self.metrics['loss']['val']):
                self.best_epoch = epoch
                self.best_metrics = logs

            # Update plots
            self.ax1.clear()
            self.ax2.clear()
   
            # Visualizes the loss curves for both training and validation datasets,
            self.ax1.plot(self.metrics['loss']['train'], label='Train Loss')
            self.ax1.plot(self.metrics['loss']['val'], label='Val Loss')
            self.ax1.axvline(self.best_epoch, linestyle='--', color='k')
            self.ax1.set_title(f"Loss (Run {self.run_id})")
            # Add a legend and identify the training and validation loss curves
            self.ax1.legend()

            self.ax2.plot(self.metrics['accuracy']['train'], label='Train Acc')
            self.ax2.plot(self.metrics['accuracy']['val'], label='Val Acc')
            self.ax2.axvline(self.best_epoch, linestyle='--', color='k')
            self.ax2.set_title(f"Accuracy (Run {self.run_id})")
            self.ax2.legend()

            # Add a title to show the current epoch and the best epoch identified during training.
            plt.suptitle(f"Epoch {epoch+1} | Best Epoch: {self.best_epoch+1}")
    
        def on_train_end(self, logs=None):
            print(f"\n[Run {self.run_id}] Training complete — Best Epoch: {self.best_epoch+1}, Best Val Acc: {self.best_metrics.get('val_accuracy', 'N/A'):.4f}")

            # Define the storage folder for the plots
            plot_path = "training_plots/"
            # Define the base folder
            base_folder = "result_files/mlp_folder"
            # Define the directory for the plots
            plot_dir = os.path.join(base_folder, plot_path)
            # Create the directory if it doesn't exist
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            # Save the plot
            plt.savefig(os.path.join(plot_dir, f"training_plot_run_{self.run_id}.png"))

            # Final pause to display, then close figure
            # plt.pause(0.5)
            plt.close(self.fig)
    
    # Define the KerasMLPW rapper
    mlp = KerasMLP(hidden_layer_options=hidden_layer_options, seed=seed)

    # BayesiansearchCV for hyperparameter optimization
    opt = BayesSearchCV(
        estimator=mlp,
        search_spaces=search_space,
        scoring={
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
            # Macro for equal weight to each class
            "precision": make_scorer(precision_score, average="macro"),
            "recall": make_scorer(recall_score, average="macro"),
            "f1": make_scorer(f1_score, average="macro")
            },
        n_iter=50,
        cv=3,
        n_jobs=1,
        verbose=1,
        refit="f1",
        random_state=seed,
    )
    print(f"Optimizer: {opt}")

    # Calculate class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Fit the model
    opt.fit(X_train, y_train)


    # Define the best model
    best_model = opt.best_estimator_
    print(f"Best Model: {best_model}")
    
    # Make predictions on the validation set
    y_pred = opt.predict(X_val)

    # Evaluate the optimized model
    print("Best parameters found:", opt.best_params_)
    print("Best cross-validation score:", opt.best_score_)
    print("Validation set Balanced Accuracy:", balanced_accuracy_score(y_val, y_pred))
    
    # Best parameters as dictionary
    best_params: dict = dict(opt.best_params_)
    # Replace the index with the actual hidden_layer_sizes tuple
    best_params['hidden_layer_sizes'] = hidden_layer_options[opt.best_params_['hidden_layer_sizes']]

    # Clean hidden_layer_sizes from the model
    best_model.hidden_layer_sizes = best_params['hidden_layer_sizes']
    best_model.hidden_layer_options = None

    # Save the best params to a file
    with open(str(best_params_path), "w") as f:
        f.write(str(best_params))

    # Calculate the duration of the optimization process
    duration = timedelta(seconds = time.perf_counter() - start_time)
    print(f"\nJob took: {duration}\n")

    # Save the best model
    best_model.model.save(best_model_path)
    
    return best_params, best_model


if __name__ == "__main__":
    optimization()



def test_model(X_test, y_test, best_model_path: str, num_classes: int, target_names: list[str]) -> None:
    """
    Evaluate the trained MLP model and generate diagnostic plots:
    - Confusion Matrix
    - ROC Curves (One-vs-Rest for multiclass)
    - Class Prediction Distributions
    """
    # Load the model
    model = keras.models.load_model(best_model_path)

    # Compile the model
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy")
        ]
    )

    # Evaluate
    test_loss, test_acc, top2_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Top-2 Accuracy: {top2_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    predictions = model.predict(X_test)

    # Reduce dimensionality for visualization (UMAP/t-SNE)
    import umap  # pip install umap-learn
    reducer = umap.UMAP(random_state=42)
    embeddings = reducer.fit_transform(predictions)
    
    # Plot class separation
    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(target_names):
        plt.scatter(embeddings[y_test == i, 0], embeddings[y_test == i, 1], 
                    label=class_name, alpha=0.6)
    plt.title("Feature Space Visualization (Last Hidden Layer)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.savefig("result_files/mlp_folder/feature_space_separation.png")
    plt.close()
    # Class-wise mean distance analysis
    from sklearn.metrics import pairwise_distances
    class_means = [np.mean(predictions[y_test == i], axis=0) for i in range(len(target_names))]
    
    # Create an empty dictionary
    mean_dist = {}
    # Define the distance matrix
    distance_matrix = pairwise_distances(class_means)
    # Iterate over the distance matrix
    for i, row in enumerate(distance_matrix):
        # Add the means to the dictionary
        mean_dist[target_names[i]] = format(np.mean(np.delete(row, i)), ".4f")
    # Save the mean distances to a file
    with open("result_files/mlp_folder/class_feature_space_distances.txt", "w") as f:
        f.write(str(mean_dist))
        


    # Generate diagnostic plots
    def plots(model, X_test, y_test, num_classes, target_classes=target_names):
        """
        Generate diagnostic plots for the MLP model:
        - Confusion Matrix
        - ROC Curves (One-vs-Rest for multiclass)
        - Class Prediction Distributions
        """
        # Generate predictions
        y_pred_probs = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_probs, axis=-1)

        # --- Plot 1: Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
    
        # Add value annotations
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        tick_marks = np.arange(len(target_classes))
        plt.xticks(tick_marks, target_classes, rotation=45)
        plt.yticks(tick_marks, target_classes, rotation=45)
        plt.savefig("result_files/mlp_folder/confusion_matrix.png")
        plt.close()

        # --- Plot 2: ROC Curves (for multiclass) ---
        if num_classes > 2:
            # Binarize labels for ROC
            y_test_bin = label_binarize(y_test, classes=np.arange(len(target_classes)))
            fpr, tpr, roc_auc = {}, {}, {}

            plt.figure(figsize=(10, 8))
            for i, class_name in enumerate(target_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], label=f"{class_name} (AUC = {roc_auc[i]:.2f})")

            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves (One-vs-Rest)")
            plt.legend()
            plt.savefig("result_files/mlp_folder/roc_curves.png")
            plt.close()

         # --- Plot 3: Prediction Confidence Distribution ---
        plt.figure(figsize=(12, 6))
        for class_id, class_name in enumerate(target_names):
            # Get probabilities assigned to the true class
            true_class_probs = y_pred_probs[y_test == class_id, class_id]
            
            # Safety check: Ensure probabilities are valid
            if np.any(true_class_probs < 0) or np.any(true_class_probs > 1):
                true_class_probs = np.clip(true_class_probs, 0.0, 1.0)
            
            # Plot distribution
            sns.kdeplot(true_class_probs, label=class_name, fill=True, alpha=0.5)

        plt.xlabel("Predicted Probability for True Class")
        plt.ylabel("Density")
        plt.title("Prediction Confidence Distribution")
        # Explicitly enforce valid probability range
        plt.xlim(0, 1)  
        plt.legend()
        plt.tight_layout()
        plt.savefig("result_files/mlp_folder/prediction_distribution.png")
        plt.close()

        # --- Classification Report ---
        print("\nClassification Report:")
        report = classification_report(y_test, y_pred_classes, target_names=target_names, digits=2)
        print(report)
        with open("result_files/mlp_folder/classification_report.txt", "w") as f:
            f.write(report)

        # --- Balanced Accuracy ---
        balanced_acc = balanced_accuracy_score(y_test, y_pred_classes)
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        with open("result_files/mlp_folder/test_balanced_accuracy.txt", "w") as f:
            f.write(f"Balanced Accuracy: {balanced_acc:.4f}")

    plots(model=model, X_test=X_test, y_test=y_test, num_classes=num_classes)


if __name__ == "__main__":
    test_model()
