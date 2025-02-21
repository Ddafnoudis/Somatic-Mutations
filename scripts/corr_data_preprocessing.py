"""
A script that returns two dataframes from the original dataset
for correlation analysis"""

from pandas import DataFrame


 # Define the categorical and numerical columns
categorical_features = ["Hugo_Symbol", 
                        "Chromosome", 
                        "Consequence",
                        "Variant_Classification", 
                        "Reference_Allele", 
                        "Tumor_Seq_Allele1", 
                        "Tumor_Seq_Allele2"]
    
numerical_features = ["Start_Position", "t_ref_count", "t_alt_count"]


def corr_data_preproc(full_data: DataFrame)-> DataFrame:
    """
    Return two dataframes: 1: Categorical features, 2: Numerical features
    """
    # Define the categorical dataset
    categorical_dataset = full_data[categorical_features]

    # Define the numerical dataset
    numerical_dataset = full_data[numerical_features].astype(int)

    print(f"\nCategorical Dataset:\n {categorical_dataset.head()},\n{categorical_dataset.columns}\n\n")
    print(f"\nNumerical Dataset:\n {numerical_dataset.head()},\n{numerical_dataset.columns}\n\n")

    return categorical_dataset, numerical_dataset


if __name__ == "__main__":
    corr_data_preproc()
