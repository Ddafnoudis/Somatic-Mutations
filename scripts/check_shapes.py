"""
Give the shape of all the train, test and 
validation sets for the trial and the MLP model.
"""

def check_shape(X_train, X_test, y_train, y_test, X_val, y_val)-> None:
    """
    Debugging function to check the shapes
    """
    print("\n\nSHAPEs\n")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    return None


if __name__ == "__main__":
    check_shape()
