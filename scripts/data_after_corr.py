"""
A script that returns A dataframe after correlation
"""
import itertools
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
    # Print the lists of features
    print(f"Spearmns features column 1:\n {speaman_list} and Cramers2:\n {cramers_list}")

    # Remove columns from the full_data that do not exist in the total_features
    categorical_data = full_data[cramers_list]
    numerical_data = full_data[speaman_list]
    print(f"Data after correlation: {categorical_data} and {numerical_data}")
    
    return categorical_data, numerical_data

    
if __name__ == '__main__':
    data_after_correlation()
