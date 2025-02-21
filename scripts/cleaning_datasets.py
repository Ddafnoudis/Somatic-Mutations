"""
Cleaning datasets for downstream analysis.
1) Every dataset has on type of cancer.
2) Remove missing data
3) Concatenate all datasets into one dataframe based on similar columns.
"""
import numpy as np
import pandas as pd
import plotly.express as px


path = "datasets/"

def clean_dataframes():
    """
    Cleaning, Merging, Concatenating 4 datasets.
    """
    df1 = pd.read_csv(path + "all_stjude_2015_data_clinical_sample.txt", sep="\t", header=4)
    df2 = pd.read_csv(path + "all_stjude_2015_data_mutations.txt", sep="\t")
    # Exlude uneeded columns
    exclude_col = ['SAMPLE_ID', 'MLL_STATUS', 'LEUKEMIA_TYPE',
       'MUTATION_RATE', 'GENE_PANEL', 'CANCER_TYPE','ONCOTREE_CODE', 
       'SOMATIC_STATUS', 'TMB_NONSYNONYMOUS']
    df1 = df1.drop(exclude_col, axis=1)

    # Remove uneccessary cancer types from this dataset (keep only ALL)
    col_cancer_type = "CANCER_TYPE_DETAILED"
    for cancer in df1[col_cancer_type]:
        if cancer in ["Acute Myeloid Leukemia", "Acute Undifferentiated Leukemia", "Leukemia"]:
            df1 = df1[df1[col_cancer_type]!= cancer]
   
    # Merge by using PATIENT_ID (df1) and Tumor_Sample_Barcode (df2)
    first_df = pd.merge(df1, df2, left_on= "PATIENT_ID", right_on="Tumor_Sample_Barcode")
    df4 = pd.read_csv(path + "all_stjude_2016_data_mutations.txt", sep="\t")
    df6 = pd.read_csv(path + "laml_tcga_pan_can_atlas_2018_data_mutations.txt", sep="\t", low_memory=False)
    df8 = pd.read_csv(path + "cll_broad_2015_data_mutations.txt", sep="\t")

    # Fill NaN with "-"
    dfs_list = [first_df, df4, df6, df8]
    for df in dfs_list:
        df.replace("-", np.NaN, inplace=True)
        df.replace(".", np.NaN, inplace=True)
    
    # Get the column names that each dataframe share
    columns_all_2015 = set(first_df.columns)
    columns_all_2016 = set(df4.columns)
    columns_laml_2018 = set(df6.columns)
    columns_cll_2015 = set(df8.columns)
    # The intersection() method returns a set that contains the similarity between two or more sets.
    common_columns = columns_all_2015.intersection(columns_all_2016, columns_laml_2018, columns_cll_2015)

    # Get the uncommon columns
    uncommon_columns_all_2015 = columns_all_2015 - common_columns
    uncommon_columns_all_2016 = columns_all_2016 - common_columns
    uncommon_columns_laml_2018 = columns_laml_2018 - common_columns
    uncommon_columns_cll_2015 = columns_cll_2015 - common_columns

    # Drop the columns that are not in the common_columns
    concatenated_common_collumns = pd.concat(dfs_list, join="inner", ignore_index=True)

    # Drop the uncommon columns
    concatenated_common = concatenated_common_collumns.drop(
    ['Entrez_Gene_Id', 'Center', 'dbSNP_RS', 'dbSNP_Val_Status',
     'dbSNP_Val_Status','Matched_Norm_Sample_Barcode', 'Match_Norm_Seq_Allele1',
       'Match_Norm_Seq_Allele2', 'Tumor_Validation_Allele1',
       'Tumor_Validation_Allele2', 'Match_Norm_Validation_Allele1',
       'Match_Norm_Validation_Allele2', 'Verification_Status',
       'Validation_Status', 'Mutation_Status', 'Sequencing_Phase',
       'Sequence_Source', 'Validation_Method', 'Score', 'BAM_File',
       'Sequencer','n_ref_count', 'n_alt_count',
    'HGVSc', 'HGVSp', 'HGVSp_Short', 'RefSeq',
       'Protein_position', 'Codons', 'Hotspot'], axis = 1)
    
    # Replace "-", "." with NaN
    nan_dataset = concatenated_common.replace("-", np.NaN)
    nan_dataset = concatenated_common.replace(".", np.NaN)

    # Create a missing data heatmap using Plotly 
    fig = px.imshow(nan_dataset.isna().transpose(), 
                    labels=dict(x="Rows", y="Columns"),
                    color_continuous_scale="YlGnBu",
                    color_continuous_midpoint=30,
                    title="Missing Data Heatmap")

    fig.update_layout(coloraxis_colorbar=dict(title="Missing Data"))
    # fig.show()
    
    # Print the total number of missing values and convert it into percentage
    nan_dataset_percent = nan_dataset.isna().sum().sum() / len(nan_dataset) * 100
    print(f"The Percent of missing values is {nan_dataset_percent}")  

    # Drop the rows that have missing data
    nan_dataset.dropna(ignore_index=True, inplace=True)
    # Rename dataframe
    full_data = nan_dataset

    # Missing data heatmap using Plotly Express
    fig = px.imshow(full_data.isna().transpose(), 
                    labels=dict(x="Rows", y="Columns"),
                    color_continuous_scale="YlGnBu",
                    color_continuous_midpoint=30,
                    title="Missing Data Heatmap")

    fig.update_layout(coloraxis_colorbar=dict(title="Missing Data"))
    # fig.show()

    # Assuming full_data is your DataFrame containing the 'Tumor_Sample_Barcode' column
    full_data['Disease_Type'] = np.where(full_data['Tumor_Sample_Barcode'].str.startswith('SJ'), 'ALL', 
                                np.where(full_data['Tumor_Sample_Barcode'].str.startswith('TCGA'), 'LAML', 'CLL')
                                )
    # Define the disease types 
    disease_types = full_data['Disease_Type'].value_counts()

    fig = px.pie(values=disease_types, names=disease_types.index, title='Disease-Associated Genomic Variations: A Pie Chart Overview')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    # fig.show()

    # Remove constant columns
    full_data = full_data.drop(columns=["NCBI_Build", "Strand", "Tumor_Sample_Barcode", "Transcript_ID", "End_Position"], axis=1)
    full_data.to_csv("datasets/full_columns_data.tsv", sep="\t", index=False)

    return full_data


if __name__ == "__main__":
    clean_dataframes()
