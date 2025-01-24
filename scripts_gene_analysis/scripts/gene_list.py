"""
Generate lists of genes out of the dataset
"""
from typing import List
from pathlib import Path
from pandas import DataFrame


def gene_list_(dataset: DataFrame, gene_file_folder: Path)-> List[str]:
    """
    Generate 4 files with the lists of genes
    1) The first should contain all the genes from the dataset
    2) The rest should contain the genes for each cancer type
    """
    # Define the gene list
    gene_list = dataset["Hugo_Symbol"]

    # Define the abbreviation of the three leukemias
    all_cancer_type = "ALL"
    laml_cancer_type = "LAML"
    cll_cancer_type = "CLL"

    # Locate the ALL genes in the dataset and save them in separate file
    all_selected_genes = dataset.loc[dataset['Disease_Type'] == all_cancer_type, 'Hugo_Symbol']
    all_selected_genes.to_csv(gene_file_folder / "ALL_gene_list.txt", index=False, header=False)
    
    # Locate the AML genes in the dataset and save them in separate file
    laml_selected_genes = dataset.loc[dataset['Disease_Type'] == laml_cancer_type, 'Hugo_Symbol']
    laml_selected_genes.to_csv(gene_file_folder / "LAML_gene_list.txt", index=False, header=False)
    
    # Locate the CLL genes in the dataset and save them in separate file
    cll_selected_genes = dataset.loc[dataset['Disease_Type'] == cll_cancer_type, 'Hugo_Symbol']
    cll_selected_genes.to_csv(gene_file_folder / "CLL_gene_list.txt", index=False, header=False)
    
    return gene_list, all_selected_genes, laml_selected_genes, cll_selected_genes


if __name__ == "__main__":
    gene_list_()
