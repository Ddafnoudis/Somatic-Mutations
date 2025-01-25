"""
Perform over-representation analysis using hallmark gene set files.
"""
import pandas as pd
import gseapy as gp
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt 


def over_representation_analysis(gene_file_folder: Path, enr_res_folder: Path, enr_plots: Path):
    """
    Perform over-representation analysis using hallmark gene set files.
    """
    # Define the gene lists and corresponding titles
    files = ["ALL_gene_list.txt", "LAML_gene_list.txt", "CLL_gene_list.txt"]
    title_names = ["Acute Lymphoblastic Leukemia", "Acute Myeloid Leukemia", "Chronic Lymphocytic Leukemia"]

    # Iterate through each gene list file and its corresponding title
    for file, title_name in zip(files, title_names):
        # Read the gene list from the file
        df = pd.read_csv(f'{gene_file_folder}/{file}', sep='\t', dtype=object)
        
        # Print available gene set libraries
        names = gp.get_library_name(organism="Human")
        print(f"Names: \n{names[:5]}\n")
        
        # Convert dataframe to list
        glist = df.squeeze().str.strip().to_list()
        print(f"List of first 10 genes: \n{glist[:10]}\n")
        
        # Define the hallmark gene set file and significance threshold
        significance_threshold = 0.05
                
        # Perform over-representation analysis
        enr_over_repr = gp.enrichr(gene_list=glist,
                                organism="human",
                                gene_sets=["Reactome_2022"],
                                cutoff=significance_threshold,
                                verbose=True)
        
        # Get the enrichment results
        enr_results = enr_over_repr.results

        # Save the enrichment results to the ora_results folder
        enr_results.to_csv(f"{enr_res_folder}/{title_name}_enrichment_results.csv", sep='\t', index=False)
        
        # Filter significant pathways
        significant_pathways = enr_results[enr_results['Adjusted P-value'] < significance_threshold]

        # Plot the top 15 most significant pathways
        if not significant_pathways.empty:
            top15_pathways = significant_pathways.nsmallest(15, 'Adjusted P-value')
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=top15_pathways,
                x='Term',
                y='Adjusted P-value',
                size='Adjusted P-value',
                hue='Gene_set',
                palette='viridis',
                sizes=(100, 500),
                legend=False,
                marker='o'
            )
            plt.axhline(y=significance_threshold, color='red', linestyle='--', label='Significance Threshold (0.05)')
            plt.xticks(rotation=90)
            plt.yticks(rotation=55)
            plt.title(f"Top 15 Significant Pathways for {title_name}")
            plt.xlabel('Gene Set')
            plt.ylabel('Adjusted P-value')
            plt.tight_layout()
            plt.savefig(f"{enr_plots}/{title_name}_plot.png")
            plt.close()

    return enr_results


if __name__ == "__main__":
    over_representation_analysis()
