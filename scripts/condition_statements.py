"""
Condition statements
"""
import os
from typing import Dict
from pathlib import Path
from scripts.trial import grid_search
from scripts.correlation import correlation
from scripts.check_shapes import check_shape
from scripts.lazy_predict import lazy_predict
from scripts.cleaning_datasets import clean_dataframes
from scripts.encoding import encode_data, stratified_k_fold
from scripts.hyperparameter_tuning import random_forest_tuning
from scripts.random_forest import random_forest_train_test_validation 
from scripts.mlp_nn import multilayer_perceptron, validate_multilayer_perceptron


def condition_statement(output_dir: Path, data: Path, 
                        corr_image: Path, corr_results: Path, 
                        lzp_results: Path, report_rf: Path, 
                        seed: int, best_params: Path, 
                        rf_best_parameters: Path, mlp_results: Path, 
                        epochs: int, param_grid: Dict,
                        rf_parameters: Dict):
    """
    Create conditions statements for the presence of the
    results files you need to have in the result_files/
    folder
    """
    # Clean data from missing values
    full_data = clean_dataframes()

    # Check if the results folder exists
    if not output_dir.exists():
        os.mkdir(output_dir)
    else:
        print(f"The {output_dir} exists!")

    # Cramer_v for finding the correlation between features
    if corr_image.exists() and corr_results.exists():
        print(f"Correlation has been completed already. Location: {output_dir}/\n")
    else:
        print("Correlation process begins!\n")
        # Perform Cramer's V correlation
        correlation(full_data)
    # Remove columns that are highly correlated
    full_data = full_data.drop(columns=["Transcript_ID", "End_Position", "Variant_Type",
                                    "Tumor_Seq_Allele1", "Reference_Allele", 
                                    "Tumor_Sample_Barcode", "Consequence"])
    # If the data are not save then save the full dataset that we will work with
    if data.exists():
        print("Full data exists")
    else:
        full_data.to_csv("datasets/full_data.tsv", sep="\t", index=False)
    
    # Define features and target
    features = full_data.iloc[:, :-1]
    target = full_data["Disease_Type"]
    target_classes = target.unique().tolist()

    # Encode the data 
    features_enc, target_enc = encode_data(feat=features, tar=target, seed=seed)
    # print(f"The shape of the data after ordinal enocding is:\n{data.shape}")

    # Train-test-validation stratified k-fold split
    X_train, X_test, X_val, y_train, y_test, y_val, y_train_dl_reshaped, y_test_dl_reshaped, y_val_dl_reshaped, num_classes = stratified_k_fold(feat_enc=features_enc, 
                                                                       tar_enc=target_enc, 
                                                                       target_classes_dl=target_classes, 
                                                                       seed=seed)
    # Shape of train, test and validation sets
    check_shape(X_train=X_train, X_test=X_test, 
                y_train=y_train, y_test=y_test, 
                X_val=X_val, y_val=y_val, y_train_dl_reshaped=y_train_dl_reshaped,
                y_test_dl_reshaped=y_test_dl_reshaped, y_val_dl_reshaped=y_val_dl_reshaped)
    
    # Results of Random Forest
    if report_rf.exists() or rf_best_parameters.exists():
        print(f"Random Forest has been completed. Location: {output_dir}/\n")
    else:
        # Random Forest hyperparameter tuning
        print("Starting Random Forest parameters tuning process!\n")
        rf_best_params = random_forest_tuning(X_train=X_train, y_train=y_train, seed=seed, forest_params=rf_parameters)
        # Perform a Random Forest classification
        print("Random Forest classification begins!\n")
        random_forest_train_test_validation(X_train=X_train, y_train=y_train,
                                            X_test=X_test, y_test=y_test, 
                                            X_val=X_val, y_val=y_val, 
                                            target_classes=target_classes, 
                                            rf_best_params=rf_best_params,
                                            seed=seed)
    # Results of lazy predict
    if lzp_results.exists():
        print(f"Lazy predict has done its predictions! Location: {output_dir}/\n")
    else:
        # Perform lazy predict classification 
        print("Start Lazy Predict classification!")
        lazy_predict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, seed=seed)

    # Grid search optimization
    if os.path.exists(best_params):
        print("Best params for the MLP exist!")
    else:
        # MLP grid search optimization
        best_params = grid_search(X_train_dl=X_train, X_val_dl=X_val,
                                  y_train_dl=y_train_dl_reshaped, y_val_dl=y_val_dl_reshaped,
                                  epochs=epochs, num_classes=num_classes, 
                                  seed=seed, param_grid=param_grid)


    # Multilayer Perceptron (Sequential)
    if mlp_results.exists():
        print("Multilayer Result exist!")
    else:
        sequential_model, X_val_dl, y_val_dl, target_classes = multilayer_perceptron(
            X_train_dl=X_train, X_test_dl=X_test, epochs=epochs,
            y_train_dl=y_train_dl_reshaped, y_test_dl=y_test_dl_reshaped, 
            X_val_dl=X_val, y_val_dl=y_val_dl_reshaped,
            num_classes=num_classes, target_names=target_classes, seed=seed, best_params=best_params
            )
        # Validate the model
        validate_multilayer_perceptron(X_val_dl=X_val_dl, y_val_dl_reshaped=y_val_dl_reshaped, 
                                       sequential_model=sequential_model, target_classes_dl=target_classes)


if __name__ == "__main__":
    condition_statement()
