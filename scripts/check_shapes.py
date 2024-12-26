"""
Give the shape of all the train, test and 
validation sets for the trial and the MLP model.
"""

def check_shape(X_train, X_test,
                y_train, y_test,
                X_val, y_val,
                y_train_dl_reshaped, y_test_dl_reshaped, y_val_dl_reshaped)-> None:
    """
    Debugging function to check the shapes
    """
    print(f"\n\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_train_dl_reshaped shape: {y_train_dl_reshaped.shape}")
    print(f"y_test_dl_reshaped shape: {y_test_dl_reshaped.shape}")
    print(f"y_val_dl_reshaped shape: {y_val_dl_reshaped.shape}\n\n")
    
    return None


if __name__ == "__main__":
    check_shape()
