"""
A script that plots the number of genes in the dataset,
and preprocess data, aiming to reduce cardinality effects.
"""
import os
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def cardinality(filtered_data):
    """
    - Plots the number of occurrences of genes in the dataset.
    - Develops a dynamic target encoder for categorical features.
    - Preprocesses the data to reduce cardinality effects.
    """
    
    class DynamicTargetEncoder:
        """
        A dynamic target encoder that provides adaptive smoothing for categorical encoding.
        
        Attributes:
            - smoothing (str or float): Smoothing strategy. 'auto' uses log-based smoothing, 
                                       or a custom float value can be provided.
            - min_samples_leaf (int): Minimum number of samples required for category-specific encoding.
            - _mapping (dict): Stores encoded values for each category after fitting.
        
        """
        def __init__(self, smoothing="auto", min_samples_leaf=1):
            self.smoothing = smoothing
            # Minimum number of samples a category
            self.min_samples_leaf = min_samples_leaf
            self._mapping = None

        def _compute_smoothing(self, n):
            if self.smoothing == "auto":
                # Dynamic smoothing based on the number of samples
                return np.maximum(1, np.log1p(n.mean()))  
            return float(self.smoothing)

        def fit(self, X, y):
            # Create an emtpy mapping dictionary
            self._mapping = {}
            # Define the categories based on the unique feature values 
            categories = np.unique(X)
            
            # Convert y to numerical 
            if y.dtype == 'object':
                # Define the LabelEncoder
                le = LabelEncoder()
                # Fit and transform the target variable
                y_encoded = le.fit_transform(y)
            else:
                # Keep original values
                y_encoded = y.values

            # Iterate through unique categories in the input data
            for category in categories:
                # Create a label of True/False for each sample
                mask: bool = (X == category)
                # Number of samples
                n = mask.sum()
                # Compute the target mean for the current category
                target_mean = y_encoded[mask].mean()
                # Compute the global mean for the target variable
                global_mean = y_encoded.mean()
                
                # Define the value taking the function for dynamic smoothing
                smoothing_value = self._compute_smoothing(n)
                # Define the weight for smoothing
                weight = expit((n - self.min_samples_leaf) / smoothing_value)
                
                # This line computes a smoothed target mean for each category and stores it in the encoder's mapping dictionary
                self._mapping[category] = weight * target_mean + (1 - weight) * global_mean
            return self

        def transform(self, X):
            """
            Converts a categorical array X into a numerical array using the mapping learned during fit().
            """
            if self._mapping is None:
                raise ValueError("Encoder not fitted yet")
            return np.array([self._mapping.get(x, np.nan) for x in X])

        def fit_transform(self, X, y):
            return self.fit(X, y).transform(X)
        

    def plot_cardinality(filtered_data):
        """
        Analyze the frequency of gene occurrences and visualizes the distribution.
        """
         # Count occurrences of each gene
        hugo_counts = filtered_data["Hugo_Symbol"].value_counts()

        # Group by frequency (e.g., how many genes appear 5 times, 10 times, etc.)
        frequency_distribution = hugo_counts.value_counts().rename_axis("Frequency").reset_index(name="Gene_Count")

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
        # plt.show()


    def cardinality_prepr(filtered_data):
        """
        Preprocess data to reduce cardinality effects.
        """
        # Define the target (keep original Disease_Type)
        target = filtered_data["Disease_Type"]
        
        # Transform Hugo_Symbol dynamically
        encoder = DynamicTargetEncoder(smoothing="auto")
        hugo_symbol_encoded = encoder.fit_transform(filtered_data["Hugo_Symbol"].values, target)
        
        # Replace the original Hugo_Symbol column with the encoded one
        filtered_data["Hugo_Symbol"] = hugo_symbol_encoded
        
        # Save the preprocessed data
        filtered_data.to_csv("datasets/full_data.tsv", sep="\t", index=False)


    # Execute functions
    plot_cardinality(filtered_data)
    cardinality_prepr(filtered_data)


if __name__=="__main__":
    cardinality()
