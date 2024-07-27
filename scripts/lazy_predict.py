"""
Lazy predict
"""
from lazypredict.Supervised import LazyClassifier
from concurrent.futures import ThreadPoolExecutor


def lazy_predict(X_train, X_test, y_train, y_test, seed):
    """
    Run lazy predict package
    """
    # Define lazy precict classifier
    lazy_predict_all_data = LazyClassifier(verbose=0, 
                                           ignore_warnings=False, 
                                           custom_metric=None,
                                           predictions=True,
                                           random_state=seed)
    # Fit the model
    model_all_data, predictions = lazy_predict_all_data.fit(X_train, X_test, y_train, y_test)
    # Save the model predictions
    model_all_data.to_csv("result_files/lzp_results.tsv", sep="\t")

    return model_all_data

def run_lazy_predict_with_threading(X_train, X_test, y_train, y_test, seed):
    """
    Use ThreadPoolExecutor to run lazy predict in a separate thread
    """
    # ThreadPooldExecuter with four workers
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit the lazy_predict funtion to the executer
        future = executor.submit(lazy_predict, X_train, X_test, y_train, y_test, seed)
        # Define the result 
        model_all_data = future.result()

    return model_all_data


if __name__ == "__main__":
    run_lazy_predict_with_threading()
