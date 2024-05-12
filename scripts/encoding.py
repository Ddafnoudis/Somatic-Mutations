"""
1) Encode dataset
2) Stratified k-fold cross-validation
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


def over_sampling_encode_data(feat, tar, seed):
    """
    Over sample the minor classes and encode data
    """
   # Oversampling
    ros = RandomOverSampler(random_state=seed)
    features, target = ros.fit_resample(feat, tar)
    features = features.astype(str)
    # Ordinal encoding
    ordinal_encoder = OrdinalEncoder()
    # Define columns
    columns = ['Hugo_Symbol', 'Chromosome', 'Start_Position', 'Variant_Classification',
       'Tumor_Seq_Allele2', 't_ref_count', 't_alt_count']
    # Encode features and target and set as DataFrame
    features_enc = pd.DataFrame(ordinal_encoder.fit_transform(features), columns=columns)
    # Create the label encoder
    label_enc = LabelEncoder()
    # Encode target
    target_enc = label_enc.fit_transform(target)
    # Tranform target into Dataframe
    target_enc_df = pd.DataFrame(target_enc, columns=["Disease_Type"])
    # Concatenate encoded features and target
    data = pd.concat([features_enc, target_enc_df], axis=1)    
    
    return data, features_enc, target_enc


def stratified_k_fold(feat_enc, tar_enc, seed):
    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, test_index in skf.split(feat_enc, tar_enc):
        X_train, X_test = feat_enc.iloc[train_index], feat_enc.iloc[test_index]
        y_train, y_test = tar_enc[train_index], tar_enc[test_index]
    # Validation test
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,  random_state=seed)

    return X_train, X_test, X_val, y_train, y_test, y_val


if __name__ == "__main__":
    over_sampling_encode_data()
    stratified_k_fold()
