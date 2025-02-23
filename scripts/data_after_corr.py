"""
A script that returns A dataframe after correlation
"""
import itertools
import numpy as np
import pandas as pd
from pandas import DataFrame


def data_after_correlation(full_data, spearman_file, cramers_file)-> DataFrame:
    """"
    Returns a dataframe after correlation
    """
    # Get the features from the correlation files in lists
    feature1 = spearman_file["feature1"].tolist()
    feature2 = spearman_file["feature2"].tolist()
    cram_feature1 = cramers_file["feature1"].tolist()
    cram_feature2 = cramers_file["feature2"].tolist()

    # Combine the lists of features and remove duplicates directly
    speaman_list = list(set(itertools.chain(feature1, feature2)))
    cramers_list = list(set(itertools.chain(cram_feature1, cram_feature2)))
    # Total features
    total_features = list(set(itertools.chain(speaman_list, cramers_list)))
    
    # Remove columns from the full_data that do not exist in the total_features
    categorical_data = full_data[cramers_list]
    numerical_data = full_data[speaman_list]

    # Concatenate categorical data with numerical data
    full_data = pd.concat([categorical_data, numerical_data], axis=1)

    # Assuming full_data is your DataFrame containing the 'Tumor_Sample_Barcode' column
    full_data['Disease_Type'] = np.where(full_data['Tumor_Sample_Barcode'].str.startswith('SJ'), 'ALL', 
                                np.where(full_data['Tumor_Sample_Barcode'].str.startswith('TCGA'), 'LAML', 'CLL'))
    
    # Drop the 'Tumor_Sample_Barcode' column
    full_data = full_data.drop(columns=['Tumor_Sample_Barcode'])

    # Save the full_data to a tsv file
    full_data.to_csv("datasets/full_data.tsv", sep="\t", index=False)

    return full_data

    
if __name__ == '__main__':
    data_after_correlation()
