"""
A script that returns a dataframe after correlation by
removing features with a p-value < 0.05.
"""
import itertools
import numpy as np
import pandas as pd
from pandas import DataFrame


def data_after_correlation(full_data, corr_folder, significant_threshold)-> DataFrame:
    """"
    Remove features with a p-value < 0.05.
    """
    # Define the dataframes
    anova_df = pd.read_csv(str(corr_folder) + "/anova_correlation.tsv", sep="\t")
    chi_df = pd.read_csv(str(corr_folder) + "/chi_square_correlation.tsv", sep="\t")

    # Sort DataFrames by p-value (ascending)
    anova_sorted = anova_df.sort_values(by="ANOVA F-Score", ascending=False)
    chi_sorted = chi_df.sort_values(by="Chi-Square Score", ascending=False)  

    # Extract sorted p-values
    anova_p = anova_sorted["p-value"]
    chi_p = chi_sorted["p-value"]
    
    # Define values greater that p-value threshold
    anova_sign_feat = anova_df[anova_p.values <= significant_threshold]["Feature"].tolist()
    chi_sign_feat = chi_df[chi_p.values <= significant_threshold]["Feature"].tolist()
    
    # Combine all features to keep
    features_sign = list(set(chi_sign_feat + anova_sign_feat ))
    # print(features_sign, type(features_sign))

    # Filter the full_data to keep only significant features
    filtered_data = full_data[features_sign]
    print(filtered_data.columns)

    # Add column based on the Tumor_Sample_Barcode if it exists in full_data
    if 'Tumor_Sample_Barcode' in full_data.columns:
        filtered_data['Disease_Type'] = np.where(full_data['Tumor_Sample_Barcode'].str.startswith('SJ'), 'ALL', 
                                np.where(full_data['Tumor_Sample_Barcode'].str.startswith('TCGA'), 'LAML', 'CLL'))
   
    # Save the full_data to a tsv file
    filtered_data.to_csv("datasets/full_data.tsv", sep="\t", index=False)
    
 
if __name__ == "__main__":
    data_after_correlation()
