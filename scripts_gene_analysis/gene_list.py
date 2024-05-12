"""
Generate lists of genes out of the dataset
"""
import pandas as pd


def gene_list_(df):
    """
    Generate 4 files with the lists of genes
    1) The first should contain all the genes from the dataset
    2) The rest should contain the genes for each cancer type
    """
    df = pd.read_csv('datasets/full_columns_data.tsv', sep='\t', dtype=object, index_col=False)
    gene_list = df["Hugo_Symbol"]
    gene_list.to_csv('gene_folder/gene_list.txt', index=False)

    # For example, let's extract genes where 'Variant_Classification' is 'Missense_Mutation'
    all_cancer_type = "ALL"
    laml_cancer_type = "LAML"
    cll_cancer_type = "CLL"

    all_selected_genes = df.loc[df['Disease_Type'] == all_cancer_type, 'Hugo_Symbol'].values.tolist()
    with open("scripts_gene_analysis/gene_folder/ALL_gene_list.txt", "w") as f:
        f.write(str(all_selected_genes))
    
    laml_selected_genes = df.loc[df['Disease_Type'] == laml_cancer_type, 'Hugo_Symbol'].values.tolist()
    with open("scripts_gene_analysis/gene_folder/LAML_gene_list.txt", "w") as f:
        f.write(str(laml_selected_genes))

    cll_selected_genes = df.loc[df['Disease_Type'] == cll_cancer_type, 'Hugo_Symbol'].values.tolist()
    with open("scripts_gene_analysis/gene_folder/CLL_gene_list.txt", "w") as f:
        f.write(str(cll_selected_genes))
    
    return gene_list, all_selected_genes, laml_selected_genes, cll_selected_genes


if __name__ == "__main__":
    gene_list_()