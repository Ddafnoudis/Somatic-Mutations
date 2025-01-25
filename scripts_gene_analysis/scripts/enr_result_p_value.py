"""
Find the 10 first enriched pathways based on p-values
"""
from pathlib import Path
from pandas import DataFrame


def enrich_res_sorted_top15(all_enr_reactome_22: DataFrame, laml_enr_reactome_22: DataFrame,
                            cll_enr_reactome_22: DataFrame, enr_15_folder: Path):
    # Create a dictionary
    reactome_list = [
        (all_enr_reactome_22, "ALL_gene_list"),
        (laml_enr_reactome_22, "LAML_gene_list"),
        (cll_enr_reactome_22, "CLL_gene_list")
    ]

    for enr_reactome, name in reactome_list:
        # Sort values by column
        enr_reactome = enr_reactome.sort_values("Adjusted P-value")
        
        # Get the top 15 enriched pathways
        top_enriched = enr_reactome.head(15)

        # Save the top enriched pathways to a file
        top_enriched.to_csv(f"{enr_15_folder}/{name}_top_enriched_pathways.tsv", sep='\t', index=False)


def common_pathways(all_enr_reactome_22, laml_enr_reactome_22, cll_enr_reactome_22):
    """
    A funtion that prints that common pathways of 
    all leukemia types
    """
    # Common common pathways in total of leukemia types
    total_common_element = set(
        all_enr_reactome_22['Term']).intersection(
            set(laml_enr_reactome_22['Term']).intersection(set(cll_enr_reactome_22["Term"])))
    print(f"Common pathways in all leukemia types:\n {list(total_common_element)[:5]}\n")

    # AML and ALL common pathways
    all_aml_common_pathway = set(all_enr_reactome_22["Term"]).intersection(set(laml_enr_reactome_22["Term"]))
    # Print the first 5 common pathways
    print(f"AML and ALL common pathways:\n {list(all_aml_common_pathway)[:5]}\n")
    
    # AML and CLL common pathways   
    aml_cll_common_pathway= set(laml_enr_reactome_22['Term']).intersection(set(cll_enr_reactome_22["Term"]))
    # Print the first 5 common pathways
    print(f"AML and CLL common pathways:\n {list(aml_cll_common_pathway)[:5]}\n")

    # ALL and CLL common pathways
    lympho_leukemia_common_element = set(all_enr_reactome_22['Term']).intersection(set(cll_enr_reactome_22["Term"]))
    # Print the first 5 common pathways
    print(f"ALL and CLL common pathway:\n {list(lympho_leukemia_common_element)[:5]}\n")


if __name__ == "__main__":
    enrich_res_sorted_top15()
    common_pathways()
