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
    - Chi-square and Cramér's V statistic: categorical dataset
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
        # Compute Cramér's V for each categorical feature
        effect_sizes = [round(cramers_v(categorical_dataset[col], target), 2) for col in categorical_dataset.columns]

        # Compute the Chi-Square score and p-values
        chi_scores, p_values = chi2(pd.get_dummies(categorical_dataset), target)

        # Round Chi-Square scores to whole numbers and p-values to 3 decimal places
        chi_scores = [round(score) for score in chi_scores]
        p_values = [round(p, 3) for p in p_values]
        
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
        Plots the Chi-Square correlation matrix.
        """
            # Sorting features by Chi-Square Score in ascending order
        chi_results = chi_results.sort_values(by="Chi-Square Score", ascending=True)

        labels = chi_results["Feature"].values
        chi_values = chi_results["Chi-Square Score"].values
        cramer_values = chi_results["cramers-v"].values

        # Convert Chi-Square values to float
        chi_values = np.array(chi_values, dtype=float)  

        # Rescale Chi-Square Scores to align with Cramér's V values
        scaled_chi_values = (chi_values - chi_values.min()) / (chi_values.max() - chi_values.min())
        scaled_chi_values = scaled_chi_values * (cramer_values.max() - cramer_values.min()) + cramer_values.min()

        midpoint = 0

        # Define colors
        color_low = '#6A5ACD' 
        color_high = '#FF4500'  

        # Y positions for bars
        ys = np.arange(len(labels))  

        plt.figure(figsize=(12, 8))
        for y, chi, cramer, scaled_chi in zip(ys, chi_values, cramer_values, scaled_chi_values):

            plt.broken_barh([
                (midpoint, scaled_chi),
                # Negative to position on the left
                (midpoint, -cramer)  
            ],
            # Bar thickness
            (y - 0.4, 0.8),  
            facecolors=[color_high, color_low],
            edgecolors=['black', 'black'],
            linewidth=0.5)

            # Add text labels for values
            plt.text(midpoint + scaled_chi + 0.02, y, f'{chi:.2e}', va='center', ha='left', fontsize=10)
            plt.text(midpoint - cramer - 0.02, y, f'{cramer:.2f}', va='center', ha='right', fontsize=10)

        # Vertical reference line at midpoint
        plt.axvline(midpoint, color='black', linewidth=1, linestyle='dashed')

        # Remove x-axis labels and ticks
        plt.xticks([])
        
        # Set labels and aesthetics
        plt.xlabel("Scores", fontsize=14)
        plt.yticks(ys, labels)
        plt.title("Features Cramer's V and Chi-Square chart", fontsize=14, fontweight='bold')
        plt.ylim(-0.5, len(labels) - 0.5)
        plt.tick_params(left=False)

        # Remove the frame/box lines
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        # Save the plot
        plt.savefig('result_files/correlation_folder/chi_cramers_feat.png', dpi=300)
        # Show plot
        plt.show()

    
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
