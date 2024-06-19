"""
Find the 10 first enriched pathways based on p-values
"""


def enrich_res_sorted_top15(all_enr_reactome_22, laml_enr_reactome_22, cll_enr_reactome_22):
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
        top_enriched.to_csv(f"hallmark/ern_res_p_values_15/{name}_top_enriched_pathways.tsv", sep='\t', index=False)


def common_pathways(all_enr_reactome_22, laml_enr_reactome_22, cll_enr_reactome_22):
    """
    A funtion that prints that common pathways of 
    all leukemia types
    """
    # Common common pathways in total of leukemia types
    total_common_element = set(
        all_enr_reactome_22['Term']).intersection(
            set(laml_enr_reactome_22['Term']).intersection(set(cll_enr_reactome_22["Term"])))
    print(f"Common pathways in total of leukemia types:\n {total_common_element}\n")

    # AML and CLL common pathways   
    aml_cll_common_pathway= set(
        laml_enr_reactome_22['Term']).intersection(set(cll_enr_reactome_22["Term"]))
    print(f"AML and CLL common pathway:\n {aml_cll_common_pathway}\n")


    # ALL and CLL common pathways
    lympho_leukemia_common_element = set(
        all_enr_reactome_22['Term']).intersection(set(cll_enr_reactome_22["Term"]))
    print(f"ALL and CLL common pathway:\n {lympho_leukemia_common_element}\n")


if __name__ == "__main__":
    enrich_res_sorted_top15()
    common_pathways()
