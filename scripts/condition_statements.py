"""
Condition statements
"""
from pathlib import Path
from scripts.correlation import correlation
from scripts.lazy_predict import lazy_predict
from scripts.cleaning_datasets import clean_dataframes
from scripts.mlp_nn import multilayer_perceptron, validate_multilayer_perceptron
from scripts.random_forest import random_forest_train_test_validation 
from scripts.feature_selection import anova_f_value, mutual_info_class
from scripts.encoding import over_sampling_encode_data, stratified_k_fold


def condition_statement(working_dir: Path, output_dir: Path,
                        data: Path,
                        corr_image: Path, corr_results: Path,
                        anova: Path, mit: Path, 
                        lzp_results: Path, cross_val: Path, 
                        accuracy: Path, report_rf: Path, 
                        confusion_mtx: Path, seed: int,
                        mlp_results: Path):
    """
    Create conditions statements for the presence of the
    results files you need to have in the result_files/
    folder
    """
    full_data = clean_dataframes()
    # Find genes that are abundant in the 3 cancer types
    # Cramer_v for finding the correlation between features
    if corr_image.exists() and corr_results.exists():
        print(f"Correlation has been completed already. Location: {output_dir}/\n")
    else:
        print("Correlation process begins!\n")
        correlation(full_data)
    # Remove columns that are highly correlated
    full_data = full_data.drop(columns=["Transcript_ID", "End_Position", "Variant_Type",
                                    "Tumor_Seq_Allele1", "Reference_Allele", 
                                    "Tumor_Sample_Barcode", "Consequence"])
    # Save file
    if data.exists():
        print("Full data exists")
    else:
        full_data.to_csv("datasets/full_data.tsv", sep="\t", index=False)
    # Define features and target
    features = full_data.iloc[:, :-1]
    target = full_data["Disease_Type"]
    target_classes = target.unique().tolist()
    # Oversampling minor classes
    data, features_enc, target_enc = over_sampling_encode_data(feat=features, tar=target, seed=seed)
    print(f"The shape of the data after oversampling is:\n{data.shape}")

    # Train-test-validation stratified k-fold split
    X_train, X_test, X_val, y_train, y_test, y_val = stratified_k_fold(feat_enc=features_enc, tar_enc=target_enc, seed=seed)
    
    if anova.exists() and mit.exists():
        print(f"Anova and Mutual info have been done! Location: {output_dir}/\n")
    else:
        print("Feature selection begins!")
        anova_f_value(X_train, X_test, y_train)
        mutual_info_class(X_train, X_test, y_train)
    
    if lzp_results.exists():
        print(f"Lazy predict has done its predictions! Location: {output_dir}/\n")
    else:
        lazy_predict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    if report_rf.exists():
        print(f"Random Forest has been completed. Location: {output_dir}/\n")
    else:
        print("Starting Random Forest")
        random_forest_train_test_validation(X_train=X_train, y_train=y_train,
                                            X_test=X_test, y_test=y_test, 
                                            X_val=X_val, y_val=y_val, 
                                            target_classes=target_classes, seed=seed)

    # Multilayer Perceptron (Sequential)
    if mlp_results.exists():
        print("Multilayer Result exist!")
    else:
        sequential_model, X_val_dl, y_val_dl, y_val_dl_reshaped, target_classes_dl = multilayer_perceptron(feat_dl=features, tar_dl=target, target_classes_dl=target_classes, seed=seed)
        validate_multilayer_perceptron(X_val_dl=X_val_dl, y_val_dl=y_val_dl, y_val_dl_reshaped=y_val_dl_reshaped, 
                                       sequential_model=sequential_model, target_classes_dl=target_classes)


if __name__ == "__main__":
    condition_statement()
