"""
1) Encode dataset
2) Stratified k-fold cross-validation
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split


def encode_data(feat, tar, seed)-> DataFrame:
    """
    Encode data for Random Forest
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


def stratified_k_fold(feat_enc, tar_enc, target_classes_dl, seed)-> np.array:
    """
    Stratified K-fold.
    
    Returns:
        The train, test and validation sets into np.array format.
        The MetaPerceptron package requires the data to be in np.array format 
        for optimization, training and evaluation.
    """
    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, test_index in skf.split(feat_enc, tar_enc):
        X_train, X_test = feat_enc.iloc[train_index], feat_enc.iloc[test_index]
        y_train, y_test = tar_enc.iloc[train_index], tar_enc.iloc[test_index]
    
    # Validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed)

     # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    # Convert to numpy arrays
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    y_val = y_val.values.ravel()

    # List of splitted data
    dt_list = [X_train, X_test, X_val, y_train, y_test, y_val]
    # Iterate over the dt_list
    for dt in dt_list:
        # Print the type of each element in the list
        print(type(dt))
        continue

    return X_train, X_test, X_val, y_train, y_test, y_val


if __name__ == "__main__":
    encode_data()
    stratified_k_fold()
