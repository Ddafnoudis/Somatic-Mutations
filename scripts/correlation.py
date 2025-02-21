"""
Apply categorical and numerical correlation analysis on the dataset.
Apply correlation ration between the categorical & numerical features.
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
from scipy.stats import spearmanr
from pandas import DataFrame
import matplotlib.pyplot as plt


def correlation(categorical_dataset: DataFrame, numerical_dataset: DataFrame, significant_threshold: float):


    def spearman_correlation(numerical_dataset: DataFrame):
        """
        Calculates the Spearman correlation matrix.
        """
        # Define the dictionary 
        result_file = {
            "feature1": [],
            "feature2": [],
            "spearman": [],
            "p_value": []
        }

        for first_feature in numerical_dataset:
            for second_feature in numerical_dataset:
                # if second_feature.name != first_feature:
                rho, p = spearmanr(numerical_dataset[first_feature], 
                                   numerical_dataset[second_feature])
                if p < significant_threshold:
                    result_file["feature1"].append(first_feature)
                    result_file["feature2"].append(second_feature)
                    result_file["spearman"].append(rho)
                    result_file["p_value"].append(p)

        # Save the result file
        with open('result_files/spearman_correlation.txt', 'w') as file:
            for i in range(len(result_file["feature1"])):
                file.write(f"{result_file['feature1'][i]}, {result_file['feature2'][i]}, {result_file['spearman'][i]}, {result_file['p_value'][i]}\n")
        
        return rho

    # Execute and define the spearman function
    sperman = spearman_correlation(numerical_dataset)

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
    
    
    def categorical_correlation(categorical_dataset: DataFrame):
        """
        Calculates the categorical correlation matrix.
        The function iterates through each pair of columns in the DataFrame, 
        calculates the confusion matrix, and then computes Cramér's V statistic 
        to measure the association between the two categorical variables.
        """
        # Define the correlation matrix columns
        categorical_corr = DataFrame(index=categorical_dataset.columns, columns=categorical_dataset.columns)
        with tqdm(total=1000, desc="Computing Correlation", dynamic_ncols=True) as pbar:
            for first_feature in categorical_dataset.columns:
                for second_feature in categorical_dataset.columns:
                    # print(f"First feature: {i} \n Second feature: {j}\n\n")
                    confusion_matrix = pd.crosstab(categorical_dataset[first_feature], categorical_dataset[second_feature])
                    # print(f"Confusion Matrix: \n{confusion_matrix}\n\n")
                    categorical_corr.loc[first_feature, second_feature] = cramers_v(confusion_matrix)
            pbar.update(1000)        

        return categorical_corr
    
    # Assuming `features` is a DataFrame containing only categorical features
    categorical_corr_matrix = categorical_correlation(categorical_dataset)


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
    
    # Define the results dictionary
    results = {
        "feature1": [],
        "feature2": [],
        "correlation": [],
        "p_value": []
    }

    # Iterate through each pair of features and calculate the p-value
    with tqdm(total=1000, desc="Computing Correlation", dynamic_ncols=True) as pbar:
        for first_feature in range(len(categorical_corr_matrix)):
            for second_feature in range(first_feature+1, len(categorical_corr_matrix.columns)):
                # Define the correlation number 
                corr = float(categorical_corr_matrix.iloc[first_feature, second_feature])
                # Define the p-value
                p_value = ss.chi2_contingency(pd.crosstab(categorical_dataset.iloc[:, first_feature], categorical_dataset.iloc[:, second_feature]))[1]
                # Set condition
                if p_value < significant_threshold:
                    if corr < 0.25:
                        # Append the results to the dictionary's lists
                        results["feature1"].append(categorical_corr_matrix.columns[first_feature])
                        results["feature2"].append(categorical_corr_matrix.columns[second_feature])
                        results["correlation"].append(corr)
                        results["p_value"].append(p_value)
        pbar.update(1000)
        
    numerical_dataset = pd.DataFrame(results, columns=["feature1", "feature2", "correlation", "p_value"])
    numerical_dataset.to_csv("results/correlation_results.tsv", "\t")
    print(numerical_dataset);exit()
    # Save the results to a text file
    with open('result_files/correlation_results.txt', 'w') as file:
        for i in range(len(results["feature1"])):
            file.write(f"{results['feature1'][i]}, {results['feature2'][i]}, {results['correlation'][i]}, {results['p_value'][i]}\n")
    # Define the columns from the categorical and numerical


if __name__ == "__main__":
    correlation()
