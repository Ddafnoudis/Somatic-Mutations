"""
1) Encode dataset
2) Stratified k-fold cross-validation
"""
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split


def encode_data(feat, tar, seed):
    """
    Encode data
    """
    # Exclude numerical features from one-hot encoding
    numerical_features = ['Start_Position', 'Hugo_Symbol']
    categorical_features = feat.drop(columns=numerical_features)
    
    # One hot encoding for categorical features
    categorical_features_enc = pd.get_dummies(categorical_features.astype(str), drop_first=True, dtype=int)
    
    # Combine numerical and encoded categorical features
    features_enc = pd.concat([feat[numerical_features], categorical_features_enc], axis=1)
    
    print(f"Encoding Features shape: {features_enc.shape}")
    
    # Label Encoding for target
    lb = LabelEncoder()
    target_enc = lb.fit_transform(tar)
    target_enc = pd.DataFrame(target_enc, columns=["Disease_Type"])
    
    # Shuffle the data
    features_enc, target_enc = shuffle(features_enc, target_enc, random_state=seed)   
    
    return features_enc, target_enc


def stratified_k_fold(feat_enc, tar_enc, target_classes_dl, seed):
    """
    Stratified K-fold
    """
    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, test_index in skf.split(feat_enc, tar_enc):
        X_train, X_test = feat_enc.iloc[train_index], feat_enc.iloc[test_index]
        y_train, y_test = tar_enc.iloc[train_index], tar_enc.iloc[test_index]
    
    # Validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed)

    # Define the number of classes
    num_classes = len(np.unique(target_classes_dl))
    # Reshape the target values
    y_train_dl_resh = np.eye(num_classes)[y_train]
    y_test_dl_resh = np.eye(num_classes)[y_test]
    y_val_dl_resh = np.eye(num_classes)[y_val]
    # Remove the extra dimension
    y_train_dl_reshaped = np.squeeze(y_train_dl_resh)
    y_test_dl_reshaped = np.squeeze(y_test_dl_resh)
    y_val_dl_reshaped = np.squeeze(y_val_dl_resh)

    return X_train, X_test, X_val, y_train, y_test, y_val, y_train_dl_reshaped, y_test_dl_reshaped, y_val_dl_reshaped, num_classes 


if __name__ == "__main__":
    encode_data()
    stratified_k_fold()
