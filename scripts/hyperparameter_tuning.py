"""
Random Forest hyperparameter tuning
"""
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def random_forest_tuning(X_train, y_train, seed: int, forest_params: Dict)-> Dict:
    """
    Tuning process for Random Forest
    """
    best_score = 0
    
    # Instantiate the Random Forest Classifier
    rfc = RandomForestClassifier()

    rfc_search = RandomizedSearchCV(estimator=rfc,
                                    param_distributions=forest_params,
                                    scoring="balanced_accuracy",
                                    cv=5, random_state=seed,
                                    verbose=2)
    # Fit the model
    rfc_search.fit(X_train, y_train)
    # Best parameters
    rf_best_params = rfc_search.best_params_
    print(rf_best_params)
    
    return rf_best_params


if __name__=="__main__":
    random_forest_tuning()
