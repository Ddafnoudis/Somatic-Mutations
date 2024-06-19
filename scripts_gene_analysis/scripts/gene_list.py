"""
Generate lists of genes out of the dataset
"""


def gene_list_(dataset):
    """
    Generate 4 files with the lists of genes
    1) The first should contain all the genes from the dataset
    2) The rest should contain the genes for each cancer type
    """
    gene_list = dataset["Hugo_Symbol"]
    gene_list.to_csv('gene_folder/gene_list.txt', index=False)

    all_cancer_type = "ALL"
    laml_cancer_type = "LAML"
    cll_cancer_type = "CLL"

    all_selected_genes = dataset.loc[dataset['Disease_Type'] == all_cancer_type, 'Hugo_Symbol']
    all_selected_genes.to_csv("gene_folder/ALL_gene_list.txt", index=False, header=False)

    laml_selected_genes = dataset.loc[dataset['Disease_Type'] == laml_cancer_type, 'Hugo_Symbol']
    laml_selected_genes.to_csv("gene_folder/LAML_gene_list.txt", index=False, header=False)

    cll_selected_genes = dataset.loc[dataset['Disease_Type'] == cll_cancer_type, 'Hugo_Symbol']
    cll_selected_genes.to_csv("gene_folder/CLL_gene_list.txt", index=False, header=False)
    
    return gene_list, all_selected_genes, laml_selected_genes, cll_selected_genes


if __name__ == "__main__":
    gene_list_()