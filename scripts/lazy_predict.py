"""
Lazy predict
"""
from lazypredict.Supervised import LazyClassifier


def lazy_predict(X_train, X_test, y_train, y_test, seed):
    """
    Run lazy predict package
    """
    # Define lazy precict classifier
    lazy_predict_all_data = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None, random_state=seed)
    # Fit the model
    model_all_data, predictions = lazy_predict_all_data.fit(X_train, X_test, y_train, y_test)
    # Save the model predictions
    model_all_data.to_csv("result_files/lzp_results.tsv", sep="\t")

    return model_all_data


if __name__ == "__main__":
    lazy_predict()
