"""
1) Encode dataset
2) Stratified k-fold cross-validation
"""
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split


def encode_data(feat, tar, seed):
    """
    Encode data
    """
    feat_str = feat.astype(str)
    # One hot encoding
    features_enc = pd.get_dummies(feat_str, drop_first=True, dtype=int)
    print(f"Encoding Features shape: {features_enc.shape}")
    # Label Encoding
    lb = LabelEncoder()
    target_enc = lb.fit_transform(tar)
    target_enc = pd.DataFrame(target_enc, columns=["Disease_Type"])
    # Shuffle the data
    features_enc, target_enc = shuffle(features_enc, target_enc, random_state=seed)
    # Concatenate encoded features and target
    data = pd.concat([features_enc, target_enc], axis=1)    
    
    return data, features_enc, target_enc


def stratified_k_fold(feat_enc, tar_enc, seed):
    """
    Stratified K-fold
    """
    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, test_index in skf.split(feat_enc, tar_enc):
        X_train, X_test = feat_enc.iloc[train_index], feat_enc.iloc[test_index]
        y_train, y_test = tar_enc.iloc[train_index], tar_enc.iloc[test_index]
    # Validation test
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,  random_state=seed)

    return X_train, X_test, X_val, y_train, y_test, y_val


if __name__ == "__main__":
    encode_data()
    stratified_k_fold()
