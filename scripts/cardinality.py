"""
Cardinality
"""
from category_encoders import TargetEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def cardinality(filtered_data):
    """
    Analyzes the frequency of gene occurrences and visualizes the distribution.
    
    Parameters:
    filtered_data (pd.DataFrame): Data containing a column "Hugo_Symbol" with gene names.

    Returns:
    None - Saves a TSV file and displays plots.
    """
    # Count occurrences of each gene
    hugo_counts = filtered_data["Hugo_Symbol"].value_counts()

    # Ensure column names are properly assigned
    frequency_distribution = hugo_counts.value_counts().rename_axis("Frequency").reset_index(name="Gene_Count")

    # Save the frequency distribution
    frequency_distribution.to_csv("result_files/frequency_distribution.tsv", sep="\t", index=False)

    # Print first 5 rows for debugging
    print(frequency_distribution.head())
    frequency_distribution.columns = ["Frequency", "Gene_Count"]

    # Save the frequency distribution
    frequency_distribution.to_csv("result_files/frequency_distribution.tsv", sep="\t", index=False)

    # Create a dictionary of genes grouped by frequency
    genes_by_frequency = {}
    for freq in frequency_distribution["Frequency"]:
        genes_by_frequency[freq] = hugo_counts[hugo_counts == freq].index.tolist()

    # Save gene lists
    for freq, genes in genes_by_frequency.items():
        filename = f"result_files/genes_with_{freq}_occurrences.tsv"
        pd.DataFrame(genes, columns=["Gene"]).to_csv(filename, sep="\t", index=False)

    # Visualization - Barplot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=frequency_distribution["Frequency"], y=frequency_distribution["Gene_Count"], color="blue")
    plt.yscale("log")  # Log scale to handle large differences in counts
    plt.xlabel("Number of Times a Gene Appears")
    plt.ylabel("Number of Genes")
    plt.title("Gene Occurrence Distribution")
    plt.xticks(rotation=90)
    plt.grid(True, which="both",  linewidth=0.5)
    plt.savefig("result_files/gene_cardinality_plot.png", dpi=300)
    plt.show()
    exit()

    
if __name__=="__main__":
    # cardinality()

    #  # Count occurrences of each gene
    # hugo_counts = filtered_data["Hugo_Symbol"].value_counts()

    # # Group by frequency (e.g., how many genes appear 5 times, 10 times, etc.)
    # frequency_distribution = hugo_counts.value_counts().rename_axis("Frequency").reset_index(name="Gene_Count")

    # # Save the frequency distribution
    # frequency_distribution.to_csv("result_files/frequency_distribution.tsv", sep="\t", index=False)

    # # Scatter plot
    # plt.figure(figsize=(12, 6))
    
    # # Scatter plot with point size proportional to frequency
    # scatter = plt.scatter(
    #     frequency_distribution["Frequency"],  # X-axis: How many times a gene appears
    #     frequency_distribution["Gene_Count"],  # Y-axis: How many genes have this occurrence
    #     s=frequency_distribution["Frequency"] * 2,  # Bubble size scales with frequency
    #     c=frequency_distribution["Frequency"],  # Color based on frequency
    #     cmap="viridis", alpha=0.7, edgecolors="black"
    # )

    # # Add labels to each point
    # for i, row in frequency_distribution.iterrows():
    #     plt.text(
    #         row["Frequency"], row["Gene_Count"], str(row["Frequency"]),
    #         fontsize=10, ha="right", va="bottom", fontweight="bold", color="black"
    #     )

    # # Plot settings
    # plt.xscale("log")  # Log scale to spread out large values
    # plt.yscale("log")  # Log scale for better visibility
    # plt.xlabel("Number of Times a Gene Appears", fontsize=12)
    # plt.ylabel("Number of Genes", fontsize=12)
    # plt.title("Gene Occurrence Scatter Plot", fontsize=14)
    # plt.colorbar(scatter, label="Gene Presence (Color Intensity)")
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # # Save plot
    # plt.savefig("result_files/gene_cardinality_scatterplot.png", dpi=300)
    # plt.show()