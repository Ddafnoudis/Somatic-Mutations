"""
A script that loads the dataframe after the correlation and 
returns the target and the features of the dataframe.
"""
import pandas as pd

def full_dataframe(data):
    """
    """
    # Read the data
    data = pd.read_csv(data, sep="\t", dtype=object)
    # Define features and target
    features = data.iloc[:, :-1]
    target = data["Disease_Type"]
    target_classes = target.unique().tolist()

    return features, target, target_classes



if __name__ == "__main__":
    full_dataframe()