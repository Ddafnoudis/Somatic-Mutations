"""
"""
import pandas as pd
import gseapy as gp


def enrich_gene_analysis():
    """
    """
    list_of_genes=[]
    with open('gene_folder/gene_list.txt') as txt:
    # Read each line from the file
        for line in txt:
            # Split the line into individual gene IDs based on newline character
            genes_in_line = line.strip().split('\n')
            # Extend the list_of_genes with the gene IDs from this line
            list_of_genes.extend(genes_in_line)
    print(list_of_genes);exit()
    # print(type(list_of_genes));exit()
    enr = gp.enrichr(gene_list=list_of_genes, 
                     gene_sets=['MSigDB_Hallmark_2020','KEGG_2021_Human'], 
                     organism='human')
    print(enr.results.head(5))


if __name__ == "__main__":
    enrich_gene_analysis()