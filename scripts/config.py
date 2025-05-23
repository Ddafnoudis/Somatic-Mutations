"""
Configuration module for model optimization.

This module defines search spaces for hyperparameter optimization of 
Random Forest and Multilayer Perceptron models using scikit-optimize.
The search spaces are designed to be used with BayesianSearchCV.
"""
from typing import Dict, Tuple
from skopt.space import Real, Integer, Categorical


def create_model_search_space(hidden_layer_options)-> Dict[str, Tuple[int, int]]:
    """
    Creates a search space for Random Forest hyperparameter optimization.
    Creates a search space for Multilayer Perceptron hyperparameter optimization.
    
    Returns:
        Two dictionaries for RF and MLP, respectively.
    """
    # RF search space
    search_space_rf = {
        'n_estimators': Integer(50, 500, name='n_estimators'),
        'max_depth': Integer(5, 50, name='max_depth'),
        'min_samples_split': Integer(2, 20, name='min_samples_split'),
        'min_samples_leaf': Integer(1, 10, name='min_samples_leaf'),
        'max_features': Categorical(['sqrt', 'log2', None], name='max_features'),
        'bootstrap': Categorical([True, False], name='bootstrap'),
        'class_weight': Categorical(['balanced', 'balanced_subsample', None], name='class_weight')
    }

    # Define mapping of index → actual hidden layer tuple
    """
    Search space for MLP model optimization.
    """
    # MLP search space
    search_space_mlp_ = {
        # Number of neurons in each hidden layer (1 to 3 layers, each 10–100 neurons)
        'hidden_layer_sizes': Integer(0, len(hidden_layer_options)-1),
        'activation': Categorical(['relu', 'tanh']),
        'alpha': Real(1e-2, 1e-1, prior='log-uniform'),
        'learning_rate': Real(1e-4, 1e-1, prior='log-uniform'),
        'batch_size': Integer(32, 200),
        'dropout_rate': Real(0.1, 0.5, prior='uniform'),
        }

    return search_space_rf, search_space_mlp_


if __name__ == "__main__":
    create_model_search_space()
