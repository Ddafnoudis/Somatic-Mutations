"""
A script that returns three dataframes from the original dataset
for correlation analysis.
"""
# Import libraries
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder


 # Define the categorical features 
categorical_feat_name = ["Hugo_Symbol", 
                        "Chromosome", 
                        "Consequence",
                        "Variant_Classification", 
                        "Reference_Allele",
                        "Tumor_Seq_Allele1", 
                        "Tumor_Seq_Allele2"]

# Define the target
target_name = "Tumor_Sample_Barcode"

# Define the numerical features    
numerical_feat_name = ["Start_Position", "t_ref_count", "t_alt_count"]


def corr_data_preproc(full_data: DataFrame)-> DataFrame:
    """
    Return 3 dataframes: 
    1: Categorical features, 
    2: Numerical features
    3: Target
    """
    # Select only the categorical columns from the full dataset
    categorical_dataset = full_data[categorical_feat_name]

    # Label encode the categorical dataset
    categorical_dataset_encoded = categorical_dataset.apply(LabelEncoder().fit_transform)
    
    # Select and convert the numerical features to integers
    numerical_dataset = full_data[numerical_feat_name].astype(int)
    
    # Define the data target column
    target = full_data[target_name]
    
    # Return data for further analysis
    return categorical_dataset_encoded, numerical_dataset, target


if __name__ == "__main__":
    corr_data_preproc()
