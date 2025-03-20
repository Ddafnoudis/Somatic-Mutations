"""
A script that plots the number of genes in the dataset,
and preprocess data, aiming to reduce cardinality effects.
"""
import os
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder


def cardinality(filtered_data):
    """
    """

    def plot_cardinality(filtered_data):
        """
        Analyzes the frequency of gene occurrences and visualizes the distribution.
        """
         # Count occurrences of each gene
        hugo_counts = filtered_data["Hugo_Symbol"].value_counts()

        # Group by frequency (e.g., how many genes appear 5 times, 10 times, etc.)
        frequency_distribution = hugo_counts.value_counts().rename_axis("Frequency").reset_index(name="Gene_Count")

        # Save the frequency distribution
        # frequency_distribution.to_csv("result_files/frequency_distribution.tsv", sep="\t", index=False)

        # Scatter plot
        plt.figure(figsize=(12, 6))

        # Scatter plot with point size proportional to frequency
        scatter = plt.scatter(
            ## X-axis: How many times a gene appears
            frequency_distribution["Frequency"],  
            # Y-axis: How many genes have this occurrence
            frequency_distribution["Gene_Count"],  
              # Bubble size scales with frequency
            s=frequency_distribution["Frequency"] * 2,
            # Color based on frequency
            c=frequency_distribution["Frequency"],  
            cmap="viridis", alpha=0.7, edgecolors="black"
        )

        # Add labels to each point
        for i, row in frequency_distribution.iterrows():
            plt.text(
                row["Frequency"], row["Gene_Count"], str(row["Frequency"]),
                fontsize=10, ha="right", va="bottom", fontweight="bold", color="black"
            )

        # Log scale to spread out large values
        plt.xscale("log")  
        # Log scale for better visibility
        plt.yscale("log")  
        # Labels
        plt.xlabel("Gene presence count", fontsize=12)
        plt.ylabel("Number of Genes", fontsize=12)
        # Title
        plt.title("Gene Occurrence Scatter Plot", fontsize=14)
        plt.colorbar(scatter, label="Gene Presence (Color Intensity)")
        # Remove background lines
        plt.grid(False, which="both")

        # Create folder
        if not os.path.exists("result_files/cardinality"):
            os.makedirs("result_files/cardinality")
        
        # Save plot
        plt.savefig("result_files/cardinality/gene_cardinality_scatterplot.png", dpi=300)
        # Show plot
        plt.show()


    def cardinality_prepr(filtered_data):
        """
        Preprocess data to reduce cardinality effects.
        """
        # Define the target
        target = filtered_data["Disease_Type"]
        # Transform only Hugo_symbol with targetencoder
        hugo_symbol_encoded = TargetEncoder().fit_transform(filtered_data["Hugo_Symbol"], target)
        # Replace the original Hugo_Symbol column with the encoded one
        filtered_data["Hugo_Symbol"] = hugo_symbol_encoded
        # Save the preprocessed data
        filtered_data.to_csv("datasets/full_data.tsv", sep="\t", index=False)

    # Execute function
    plot_cardinality(filtered_data)
    # Execute function
    cardinality_prepr(filtered_data)


if __name__=="__main__":
    cardinality()
