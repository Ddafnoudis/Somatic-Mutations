"""
Random Forest hyperparameter tuning
"""
import os
import numpy as np
import seaborn as sns
from typing import Dict
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from skopt.callbacks import VerboseCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer, precision_score, recall_score, f1_score


class PlottingCallback:
    """
    Custom callback to plot scoring metrics during Bayesian optimization.
    """
    def __init__(self):
        self.iterations = []
        self.scores = []

    def __call__(self, optim_result):
        # Append the iteration number and score
        self.iterations.append(len(self.iterations) + 1)
        self.scores.append(optim_result.fun)
        
        # Live plot update
        plt.figure(figsize=(8, 5))
        sns.lineplot(x=self.iterations, y=self.scores, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Balanced Accuracy")
        plt.title("Bayesian Optimization Progress")
        plt.grid()
        plt.show()
        plt.pause(0.1)  # Pause to update the plot
        # plt.tight_layout()


def random_forest_tuning(X_train, y_train, X_test, y_test, search_space_rf: Dict) -> Dict:
    """
    Performs Bayesian optimization for Random Forest hyperparameters with Balanced Accuracy optimization.
    
    Parameters:
    - X_train, X_test: np.ndarray, training/testing features
    - y_train, y_test: np.ndarray, training/testing labels
    - search_space_rf: Dict, hyperparameter search space
    
    Returns:
    - Dict containing the best hyperparameters found
    """
    y_train = np.ravel(y_train)  # Flatten target
    rfc = RandomForestClassifier()
    
    rfc_search = BayesSearchCV(
        estimator=rfc,
        search_spaces=search_space_rf,
        n_iter=10,
        n_jobs=-1,
        scoring={
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
            "precision": make_scorer(precision_score, average="weighted"),
            "recall": make_scorer(recall_score, average="weighted"),
            "f1": make_scorer(f1_score, average="weighted")
        },
        refit="balanced_accuracy",
        verbose=2,
        cv=2,
        return_train_score=True,
    )
    
    print("Starting Bayesian Optimization...")
    
    # Instantiate the plotting callback
    plot_callback = PlottingCallback()
    
    # Train model with callbacks for verbose logging and plotting
    rfc_search.fit(X_train, y_train, callback=[VerboseCallback(n_total=10), plot_callback])
    
    print(f"The score is:\n{rfc_search.score(X_test, y_test):.4f}\n\n")
    
    rf_best_params = rfc_search.best_params_
    print("\nBest Parameters Found:", rf_best_params)
    
    best_model = rfc_search.best_estimator_
    y_pred = best_model.predict(X_test)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy on Test Set: {balanced_acc:.4f}")
    
    os.makedirs("result_files/rf_folder", exist_ok=True)
    with open("result_files/rf_folder/rf_best_params.txt", "w") as f:
        f.write(str(rf_best_params))
    
    return rf_best_params


if __name__ == "__main__":
    random_forest_tuning()
