"""
"""
import os
import pandas as pd
from scripts.gene_list import gene_list_
from scripts.config_gene_fun import parse_configuration_files
from scripts.enrich_gene import over_representation_analysis
from scripts.enr_result_p_value import common_pathways
from scripts.enr_result_p_value import enrich_res_sorted_top15


def main():
    # Define config file
    config = parse_configuration_files(fname='configuration_gene.yaml')

    # Define configuration variables
    working_gene_dir=config["WORK_GENE_DIR"] 
    dataset=config['DATASET']
    gmt_folder=config["GMT_FOLDER"]
    gene_file_folder=config["GENE_FILES_FOLDER"]
    hallmark_results=config["HALLMARK_RESULTS"]
    cll_enr_res=config["CLL_ENR_RESULTS"]
    enr_res_folder=config["ENR_RESULT_FOLDER"]
    
    # Parse the dataset
    df = pd.read_csv(dataset, sep='\t', dtype=object)
    # Generate files only with genes based on cancer types
    gene_list_(dataset = df)

    if cll_enr_res.exists():
        print("Gene set enrichments analysis is done!")
    else:
        os.mkdir(enr_res_folder)
       # Over-representation analysis
        over_representation_analysis()

    # Read the enrichment results for each gene list
    all_enr_reactome_22 = pd.read_csv("hallmark/ern_res_p_values_15/ALL_gene_list_top_enriched_pathways.tsv", sep="\t", index_col=False)
    laml_enr_reactome_22 = pd.read_csv("hallmark/ern_res_p_values_15/LAML_gene_list_top_enriched_pathways.tsv", sep="\t", index_col=False)
    cll_enr_reactome_22 = pd.read_csv("hallmark/ern_res_p_values_15/CLL_gene_list_top_enriched_pathways.tsv", sep="\t", index_col=False)

    # Find the 10 first enriched pathways based on p-values
    enrich_res_sorted_top15(all_enr_reactome_22, laml_enr_reactome_22, cll_enr_reactome_22)
    common_pathways(all_enr_reactome_22, laml_enr_reactome_22, cll_enr_reactome_22)


if __name__ == "__main__":
    main()
