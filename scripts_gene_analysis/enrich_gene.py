"""
"""
import pandas as pd
import gseapy as gp


def enrich_gene_analysis():
    """
    """
    df = pd.read_csv('gene_folder/ALL_gene_list.txt', sep='\t', dtype=object)
    # convert dataframe or series to list
    glist = df.squeeze().str.strip().to_list()
    # print(glist[:10]);exit()
    enr = gp.enrichr(gene_list=glist, 
                     gene_sets=['MSigDB_Hallmark_2020','KEGG_2021_Human'], 
                     organism='human')
    print(enr.results.head(5))
    print(enr.results.columns)


if __name__ == "__main__":
    enrich_gene_analysis()