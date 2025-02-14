"""
Apply categorical and numerical correlation analysis on the dataset.
Apply correlation ration between the categorical & numerical features.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List
import scipy.stats as ss
from pandas import DataFrame
import matplotlib.pyplot as plt


def correlation(full_data: DataFrame):
    # full_data = pd.read_csv(full_data)
    categorical_features = ["Hugo_Symbol", 
                            "Chromosome", 
                            "Consequence",
                            "Variant_Classification", 
                            "Reference_Allele", 
                            "Tumor_Seq_Allele1", 
                            "Tumor_Seq_Allele2"]
    
    numerical_features = ["Start_Position", "End_Position", "t_ref_count", "t_alt_count"]

    def cramers_v(confusion_matrix: DataFrame):
        """
        Calculates the Cramér's V statistic for a given confusion matrix.
        """
        # # Pass the confusion_matrix instead of full_data and correct unpacking
        chi2, p = ss.chi2_contingency(confusion_matrix, correction=False)
        print(f'\nThe p-value is {p} and the Cramer V is: {chi2}\n')
        # chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    def categorical_correlation(full_data: DataFrame, categorical_features: List):
        """
        Calculates the categorical correlation matrix.
        The function iterates through each pair of columns in the DataFrame, 
        calculates the confusion matrix, and then computes Cramér's V statistic 
        to measure the association between the two categorical variables.
        """
        categorical_corr = DataFrame(index=full_data[categorical_features], columns=full_data[categorical_features])
        # print(f"This is the categorical corr :{categorical_corr}");exit()
        for first_feature in full_data[categorical_features]:
            # print(first_feature);exit()
            for second_feature in full_data[categorical_features]:
                if first_feature != second_feature:
                    confusion_matrix = pd.crosstab(full_data[first_feature], full_data[second_feature])
                    categorical_corr.loc[first_feature, second_feature] = cramers_v(confusion_matrix)
            print(categorical_corr)
        return categorical_corr

    
    categorical_corr_matrix = categorical_correlation(full_data, categorical_features)
    print(f'\n\nCategorical correlation matrix\n\n{categorical_corr_matrix}\n\n')

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
    


if __name__ == "__main__":
    correlation()
