"""
Feature selection
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif


def anova_f_value(X_train, X_test, y_train):
    """
    Performs ANOVA F-test feature selection on the training data `X_train` and `y_train`.
    The `SelectKBest` transformer is used to select the top features based on the ANOVA F-test. 
    The p-values of the F-test are transformed to scores using `-np.log10(pvalues)`,
    and the scores are normalized to the range [0, 1]. 
    The normalized scores are returned as a feature importance metric.
    """
    print('Shape of feature train and test sets-->', X_train.shape, X_test.shape,'\n')
    selector = SelectKBest(f_classif, k="all").fit(X_train, y_train)
    scores = -np.log10(selector.pvalues_)
    scores[np.isinf(scores)] = 0
    scores /= scores.max()
    print(selector.pvalues_)

    # Assuming X_train is a DataFrame and its columns are the feature names
    feature_names = X_train.columns

    X_indices = np.arange(len(feature_names))  # Adjusted to the number of features
    plt.figure(figsize=(10, 6))  # Combine creation and size adjustment of the figure
    plt.bar(X_indices - 0.05, scores, width=0.5)
    plt.title("ANOVA F-value feature importance", fontsize=16)
    plt.xlabel("Feature", fontsize=14)  # Changed xlabel to just "Feature"
    plt.ylabel("Score ($-Log(p_{value})$)", fontsize=14)
    plt.xticks(ticks=X_indices, labels=feature_names, rotation=45, ha='right')  # Set tick labels to feature names
    plt.yticks(np.arange(0.0, 2, 0.1))
    plt.ylim(0.0, 1)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig("result_files/ANOVA_F_value_feature_importance.png")  # Save figure to file
    plt.show()


def mutual_info_class(X_train, X_test, y_train):
    """
    Plots a bar chart of the mutual information 
    between the predictors and the target variable.
    The mutual information is calculated using the `mutual_info_classif` function 
    from the scikit-learn library. 
    The resulting values are sorted in ascending order and plotted as a bar chart.
    The x-axis shows the feature names, and the y-axis shows the 
    mutual information values. The chart is displayed using the `plt.show()` function.
    """
    mutal_inf_class = mutual_info_classif(X_train, y_train)
    mutal_inf_class = pd.Series(mutal_inf_class)
    mutal_inf_class.index = X_train.columns
    mutal_inf_class.sort_values(ascending=True).plot.bar(figsize=(10, 6))
    plt.ylabel('Mutual Information')
    plt.title("Mutual information between predictors and target")
    plt.xticks(rotation=60, ha='right')
    plt.savefig("result_files/mutual_info_class.png")
    plt.show()
    
    return X_train, X_test, y_train


if __name__ == "__main__":
    anova_f_value()
    mutual_info_class()
