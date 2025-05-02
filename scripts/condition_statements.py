"""
Condition statements
"""
import os
import pandas as pd
from typing import Any
from pathlib import Path
from scripts.trial import test_model
from scripts.trial import optimization
from scripts.cardinality import cardinality
from scripts.correlation import correlation
from scripts.check_shapes import check_shape
from scripts.lazy_predict import lazy_predict
from scripts.target_features import full_dataframe
from scripts.config import create_model_search_space
from scripts.cleaning_datasets import clean_dataframes
from scripts.data_after_corr import data_after_correlation
from scripts.encoding import encode_data, stratified_k_fold
from scripts.corr_data_preprocessing import corr_data_preproc
from scripts.hyperparameter_tuning import random_forest_tuning
from scripts_gene_analysis.scripts.gene_list import gene_list_
from scripts.random_forest import random_forest_train_test_validation 
from scripts_gene_analysis.scripts.enr_result_p_value import common_pathways
from scripts_gene_analysis.scripts.enrich_gene import over_representation_analysis
from scripts_gene_analysis.scripts.enr_result_p_value import enrich_res_sorted_top15

# Hidden layers. Neuron number
HIDDEN_LAYER_OPTIONS = [
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
    (256, 512)
    ]

def condition_statement(dataset_somatic_mutation: Path, 
                        corr_folder: Path,
                        significant_threshold: float,
                        gene_file_folder: Path,
                        hallmark_results: Path,
                        enr_res_folder: Path,
                        enr_15_folder: Path,
                        enr_plots: Path,
                        aml_enrich: Path,
                        aml_enrich_15: Path,
                        aml_plot: Path,
                        rf_folder: Path,
                        output_dir: Path, data: Path, 
                        lzp_results: Path, report_rf: Path, 
                        seed: int, best_params_path: Path, 
                        mlp_results: Path, 
                        epochs: int,
                        best_model_path: Path, hidden_layer_options=HIDDEN_LAYER_OPTIONS) -> Any:
                        
    """
    Create conditions statements for the presence of the
    results files you need to have in the result_files/
    folder
    """
    # Clean data from missing values
    full_data = clean_dataframes()

    # Parse the dataset for the upstream analysis
    df_mutations = pd.read_csv(dataset_somatic_mutation, sep='\t', dtype=object)
    
    # Check if the files exist
    if aml_enrich.exists() and aml_enrich_15.exists() and aml_plot.exists():
        print("Upstream analysis has been completed already!\n")

    elif gene_file_folder.exists() and hallmark_results.exists() and enr_res_folder.exists() and enr_15_folder.exists() and enr_plots.exists():
        print("Folders exist but not the files!")
        # Generate files only with genes based on cancer types
        gene_list_(dataset=df_mutations, gene_file_folder=gene_file_folder)
       # Over-representation analysis
        over_representation_analysis(enr_res_folder=enr_res_folder,
                                     enr_plots=enr_plots,
                                    gene_file_folder=gene_file_folder)
        
        # Read the enrichment results for each gene list
        all_enr_reactome_22 = pd.read_csv(enr_res_folder / "Acute Lymphoblastic Leukemia_enrichment_results.csv", sep="\t", index_col=False)
        laml_enr_reactome_22 = pd.read_csv(enr_res_folder / "Acute Myeloid Leukemia_enrichment_results.csv", sep="\t", index_col=False)
        cll_enr_reactome_22 = pd.read_csv(enr_res_folder / "Chronic Lymphocytic Leukemia_enrichment_results.csv", sep="\t", index_col=False)

        # Find the 10 first enriched pathways based on p-values
        enrich_res_sorted_top15(all_enr_reactome_22=all_enr_reactome_22, 
                                laml_enr_reactome_22=laml_enr_reactome_22, 
                                cll_enr_reactome_22=cll_enr_reactome_22, 
                                enr_15_folder=enr_15_folder)
        # Find common pathways
        common_pathways(all_enr_reactome_22, laml_enr_reactome_22, cll_enr_reactome_22)

    else:
        print("Gene file folders does not exist!")
        # Create the multiple folders
        folders = [gene_file_folder, hallmark_results, enr_res_folder, enr_15_folder, enr_plots]
        # Iterate over the folders 
        for folder in folders:
            # Create them
            os.mkdir(folder)
        # Generate files only with genes based on cancer types
        gene_list_(dataset = df_mutations, gene_file_folder=gene_file_folder)
        # Over-representation analysis
        over_representation_analysis(enr_res_folder=enr_res_folder,
                                     enr_plots=enr_plots,
                                    gene_file_folder=gene_file_folder)
        
        # Read the enrichment results for each gene list
        all_enr_reactome_22 = pd.read_csv(enr_res_folder / "Acute Lymphoblastic Leukemia_enrichment_results.csv", sep="\t", index_col=False)
        laml_enr_reactome_22 = pd.read_csv(enr_res_folder / "Acute Myeloid Leukemia_enrichment_results.csv", sep="\t", index_col=False)
        cll_enr_reactome_22 = pd.read_csv(enr_res_folder / "Chronic Lymphocytic Leukemia_enrichment_results.csv", sep="\t", index_col=False)

        # Find the 10 first enriched pathways based on p-values
        enrich_res_sorted_top15(all_enr_reactome_22=all_enr_reactome_22, 
                                laml_enr_reactome_22=laml_enr_reactome_22, 
                                cll_enr_reactome_22=cll_enr_reactome_22, 
                                enr_15_folder=enr_15_folder)
        # Find common pathways
        common_pathways(all_enr_reactome_22, laml_enr_reactome_22, cll_enr_reactome_22)

    # Preprocessing dataset for correlation analysis
    categorical_dataset_encoded, numerical_dataset, target = corr_data_preproc(full_data=full_data)

    # Check if the results folder exists
    if not output_dir.exists():
        os.mkdir(output_dir)
    else:
        print(f"The {output_dir} exists!")

    # Cramer_v for finding the correlation between features
    if data.exists() and corr_folder.exists():
        print(f"Correlation has been completed already. Location: {output_dir}/\n")
    else:
        print("Correlation process begins!\n")
        # Create the correlation folder
        if not os.path.exists(corr_folder):
            os.makedirs(corr_folder, exist_ok=True)
        # Perform Cramer's V correlation and spearman
        correlation(target=target, categorical_dataset=categorical_dataset_encoded, numerical_dataset=numerical_dataset)
        # Return the full data after correlation
        filtered_data = data_after_correlation(full_data=full_data, corr_folder=corr_folder, significant_threshold=significant_threshold)
        cardinality(filtered_data=filtered_data)

    # Define the features, target and target classes of the dataset
    features, target, target_classes = full_dataframe(data=data)

    # Encode the data 
    features_enc, target_enc = encode_data(feat=features, tar=target, seed=seed)

    # Train-test-validation stratified k-fold split sets
    X_train, X_test, X_val, y_train, y_test, y_val= stratified_k_fold(feat_enc=features_enc, 
                                                                       tar_enc=target_enc, 
                                                                       target_classes_dl=target_classes, 
                                                                       seed=seed)
    
    # Shape of train, test and validation sets
    check_shape(X_train=X_train, X_test=X_test, 
                y_train=y_train, y_test=y_test, 
                X_val=X_val, y_val=y_val)
    
    # Define the feature size
    feature_size = X_train.shape[1]
    # print(f"The feature size is: {feature_size}\n");exit()
    
    # Create the search space for the models
    search_space_rf, search_space_mlp_ = create_model_search_space(hidden_layer_options=hidden_layer_options)
    
    # Results of Random Forest
    if report_rf.exists() and rf_folder.exists():
        print(f"Random Forest has been completed. Location: {output_dir}/\n")
    else:
        if not os.path.exists(rf_folder):
            os.makedirs(rf_folder, exist_ok=True)
        # Random Forest hyperparameter tuning
        print("Starting Random Forest parameters tuning process!\n")
        rf_best_params = random_forest_tuning(X_train=X_train, y_train=y_train, search_space_rf=search_space_rf, X_test=X_test, y_test=y_test)
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
        # Create the lazy predict folder
        if not os.path.exists(lzp_results):
            os.makedirs(lzp_results, exist_ok=True)
        # Perform lazy predict classification 
        print("Start Lazy Predict classification!")
        lazy_predict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, seed=seed)
    
    # MLP search space 
    if os.path.exists(best_params_path):
        print("Best params for the MLP exist!")
        # Open the file following the path
        with open(best_params_path, "r") as f:
            # Read the content of the file
            best_params = eval(f.read())
    else:
        os.makedirs(mlp_results, exist_ok=True)
        # MLP Bayes search optimization
        optimization(X_train=X_train, y_train=y_train,
                    y_val=y_val, X_val=X_val,
                    feature_size=feature_size,
                    search_space=search_space_mlp_,
                    epochs=epochs, seed=seed, 
                    best_params_path=best_params_path, 
                    hidden_layer_options=hidden_layer_options,
                    num_classes=len(target_classes),
                    best_model_path=best_model_path)
    
    # test model
    test_model(X_test=X_test, y_test=y_test, best_model_path=best_model_path,
               num_classes=len(target_classes), target_names=target_classes)


if __name__ == "__main__":
    condition_statement()
