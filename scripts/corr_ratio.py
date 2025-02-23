import numpy as np
import pandas as pd
from scipy.stats import kruskal

def correlation_ratio(categories, numerical_values):
    """
    Compute the Correlation Ratio (η) for categorical vs. numerical variables.
    
    :param categories: Categorical column (Series)
    :param numerical_values: Numerical column (Series)
    :return: Correlation Ratio (η)
    """
    unique_cats = np.unique(categories)
    category_groups = [numerical_values[categories == cat] for cat in unique_cats]
    
    ss_between = sum(len(group) * (group.mean() - numerical_values.mean()) ** 2 for group in category_groups)
    ss_total = sum((numerical_values - numerical_values.mean()) ** 2)
    
    return np.sqrt(ss_between / ss_total) if ss_total != 0 else 0

def kruskal_test(categories, numerical_values):
    """
    Perform Kruskal-Wallis H test for categorical vs. numerical variables.
    
    :param categories: Categorical column (Series)
    :param numerical_values: Numerical column (Series)
    :return: Kruskal-Wallis H-statistic and p-value
    """
    unique_cats = np.unique(categories)
    category_groups = [numerical_values[categories == cat] for cat in unique_cats]
    
    if len(category_groups) > 1:
        h_stat, p_value = kruskal(*category_groups)
        return h_stat, p_value
    else:
        return np.nan, np.nan

def analyze_categorical_numerical_correlation(categorical_data, numerical_data):
    """
    Perform Correlation Ratio (η) and Kruskal-Wallis H test for categorical vs. numerical variables.

    :param categorical_data: DataFrame containing categorical columns
    :param numerical_data: DataFrame containing numerical columns
    :return: DataFrame with correlation results
    """
    categorical_cols = categorical_data.columns  # Extract categorical column names
    numerical_cols = numerical_data.columns      # Extract numerical column names

    results = []

    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            eta_value = correlation_ratio(categorical_data[cat_col], numerical_data[num_col])
            h_stat, p_value = kruskal_test(categorical_data[cat_col], numerical_data[num_col])
            results.append((cat_col, num_col, eta_value, h_stat, p_value))

    # Convert results to DataFrame
    correlation_df = pd.DataFrame(results, columns=['Categorical', 'Numerical', 'Eta_Correlation', 'Kruskal_H', 'p_value'])

    # Sort results by highest correlation
    correlation_df = correlation_df.sort_values(by='Eta_Correlation', ascending=False)

    return correlation_df


if __name__ == '__main__':
    correlation_ratio()
    kruskal_test()
    analyze_categorical_numerical_correlation()
    
# # Example Usage
# correlation_results = analyze_categorical_numerical_correlation(categorical_data, numerical_data)

# # Display top 5 correlations
# print(correlation_results.head())

# # Filter significant correlations (p-value < 0.05)
# significant_correlations = correlation_results[correlation_results['p_value'] < 0.05]
# print("\nSignificant Categorical-Numerical Correlations:")
# print(significant_correlations.head())
