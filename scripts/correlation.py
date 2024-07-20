"""
Apply for correlation of columns in the dataset
Plot the correlation matrix
Save the statistical significance of the correlation matrix
"""
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
from pandas import DataFrame
import matplotlib.pyplot as plt


def correlation(full_data: DataFrame):

    def cramers_v(confusion_matrix: DataFrame):
        """
        Calculates the Cramér's V statistic for a given confusion matrix.
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    def categorical_correlation(full_data: DataFrame):
        """
        Calculates the categorical correlation matrix.
        The function iterates through each pair of columns in the DataFrame, 
        calculates the confusion matrix, and then computes Cramér's V statistic 
        to measure the association between the two categorical variables.
        """
        categorical_corr = DataFrame(index=full_data.columns, columns=full_data.columns)
        for i in full_data.columns:
            for j in full_data.columns:
                confusion_matrix = pd.crosstab(full_data[i], full_data[j])
                categorical_corr.loc[i, j] = cramers_v(confusion_matrix)
        return categorical_corr

    # Assuming `features` is a DataFrame containing only categorical features
    categorical_corr_matrix = categorical_correlation(full_data)

    def plot_categorical_correlation(corr_matrix):
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix.astype(float), cmap='coolwarm', linewidths=0.5)

        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix.columns)):
                plt.text(j+0.5, i+0.5, '{:.2f}'.format(corr_matrix.iloc[i, j]), ha='center', va='center', fontsize=10)

        plt.title('Categorical Correlation Matrix')
        plt.xlabel('Features')
        plt.ylabel('Features')
        plt.savefig('result_files/categorical_correlation.png')
        # plt.show()

    plot_categorical_correlation(categorical_corr_matrix)

    # Assess statistical significance of the correlations
    significance_threshold = 0.05
    results = []
    for i in range(len(categorical_corr_matrix)):
        for j in range(i+1, len(categorical_corr_matrix.columns)):
            corr = float(categorical_corr_matrix.iloc[i, j])
            p_value = ss.chi2_contingency(pd.crosstab(full_data.iloc[:, i], full_data.iloc[:, j]))[1]
            if p_value < significance_threshold:
                results.append(f"Statistically significant correlation between {categorical_corr_matrix.columns[i]} "
                               f"and {categorical_corr_matrix.columns[j]} (Cramer's V = {corr}, p-value = {p_value})")
            else:
                results.append(f"No statistically significant correlation between {categorical_corr_matrix.columns[i]} "
                               f"and {categorical_corr_matrix.columns[j]} (Cramer's V = {corr}, p-value = {p_value})")
    print("----------------------------------------------")
    # Save the results to a text file
    with open('result_files/correlation_results.txt', 'w') as file:
        for result in results:
            file.write(result + '\n')
  
    return full_data


if __name__ == "__main__":
    correlation()
