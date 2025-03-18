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
    # Define features
    features = data.iloc[:, :-1]
    # Print features
    print(f"Features shape and features: \n{features.shape}\n{features}")
    # Define the target
    target = data["Disease_Type"]
    print(f"Target shape and target: \n{target.shape}\n{target}")
    # Define the target classes
    target_classes = target.unique().tolist()

    return features, target, target_classes



if __name__ == "__main__":
    full_dataframe()