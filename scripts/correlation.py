"""
This script applys the Cramers V to the categorical dataset and 
Spearman to the numerical dataset. 
Returns:
    - Spearman correlation tsv file
    - Cramers V correlation tsv file
    - Categorical correlation heatmap
"""
import pandas as pd
import numpy as np
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.feature_selection import chi2, f_classif


def correlation(categorical_dataset: DataFrame, 
                numerical_dataset: DataFrame,
                target: DataFrame):
    """
    Compute the correlation between categorical and numerical features.
    - ANOVA F-value: numerical dataset
    - Cramér's V statistic: categorical dataset
    """

    def anova_correlation(numerical_dataset, target):
        """
        Calculates the ANOVA F-value.
        """
        # Compute the ANOVA F-value and p-value
        anova_scores, p_values = f_classif(numerical_dataset, target)

        # Calculate degrees of freedom
        n_groups = len(target)  # 3 leukemia types
        n_samples = len(numerical_dataset)
        df_between = n_groups - 1
        df_within = n_samples - n_groups
        
        # Calculate eta-squared
        eta_sq = (anova_scores * df_between) / (anova_scores * df_between + df_within)
    
        # Create a dataframe with the ANOVA F-value and p-value
        anova_results = pd.DataFrame({
            "Feature": numerical_dataset.columns,
            "ANOVA F-Score": anova_scores,
            "p-value": p_values,
            "eta-squared": eta_sq
        })

    
        # Save the results to a tsv file
        anova_results.to_csv("result_files/correlation_folder/anova_correlation.tsv", sep="\t", index=False)
        
        # Return the results
        return anova_results

    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2_stat, _, _, _ = chi2_contingency(confusion_matrix)
        n = confusion_matrix.sum().sum()
        phi2 = chi2_stat / n
        r, k = confusion_matrix.shape
        cramers_v = np.sqrt(phi2 / min(k-1, r-1))

        return cramers_v
    
    def chi_square_correlation(categorical_dataset: DataFrame, target: DataFrame):
        """
        Calculates the Chi-Square statistic for categorical features.
        """
        # Compute Cramer's V for each categorical feature
        effect_sizes = [cramers_v(categorical_dataset[col], target) for col in categorical_dataset.columns]
        # Compute the Chi-Square score and p-value
        chi_scores, p_values = chi2(pd.get_dummies(categorical_dataset), target)
        
        # Create a dataframe with the Chi-Square score and p-value
        chi_results = pd.DataFrame({
            "Feature": categorical_dataset.columns,
            "Chi-Square Score": chi_scores,
            "p-value": p_values,
            "cramers-v": effect_sizes
        })

        # Save the results to a tsv file
        chi_results.to_csv("result_files/correlation_folder/chi_square_correlation.tsv", sep="\t", index=False)
        
        # Return the results
        return chi_results
    

    def plot_categorical_correlation(chi_results):
        """
        Plots the Chi-Square correlation matrix with a publication-ready format.
        """
        # Figure size
        plt.figure(figsize=(12, 6))
        # Violin plot
        sns.barplot(data=chi_results, y="Chi-Square Score", x="Feature", palette="magma")
        
        plt.title('Chi-Square Correlation Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Chi-Square Correlation', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig('result_files/correlation_folder/categorical_correlation.png', dpi=300)
        # plt.show()

    def plot_numerical_correlation(anova_results):
        """
        Plots the ANOVA correlation matrix with a publication-ready format.
        """
        plt.figure(figsize=(6, 6))
        sorted_results = anova_results.sort_values(by="ANOVA F-Score", ascending=False)

        sns.barplot(data=sorted_results, y="ANOVA F-Score", x="Feature", palette="magma")
        
        plt.title('ANOVA Correlation Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('ANOVA F-Score', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig('result_files/correlation_folder/numerical_correlation.png', dpi=300)
        # plt.show()

    anova_results = anova_correlation(numerical_dataset, target)
    chi_results = chi_square_correlation(categorical_dataset, target)
    plot_categorical_correlation(chi_results)
    plot_numerical_correlation(anova_results)

    return anova_results, chi_results


if __name__ == "__main__":
    correlation()
