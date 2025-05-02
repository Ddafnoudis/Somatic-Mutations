"""
"""
from lightning import seed_everything
from scripts.config_fun import parse_configuration_files
from scripts.condition_statements import condition_statement


def main():
    config = parse_configuration_files(fname='configuration.yaml')
    
    # Seed everything
    seed_everything(config["SEED"], workers=True)

    condition_statement(
        dataset_somatic_mutation=config["DATASET_SOMATIC_MUTATION"],
        corr_folder = config["CORR_FOLDER"],
        significant_threshold=config["SIGNIFICANT_THRESHOLD"],
        gene_file_folder=config["GENE_FILES_FOLDER"], 
        hallmark_results=config["HALLMARK_RESULTS"], 
        enr_res_folder=config["ENRICH_FOLDER"],
        enr_15_folder=config["ENR_15_FOLDER"], 
        enr_plots=config["ENR_PLOTS"],
        aml_enrich=config["AML_ENRICH"],
        aml_enrich_15=config["AML_ENRICH_15"],
        aml_plot=config["AML_PLOT"],
        seed=config["SEED"], 
        data=config["DATASET"], 
        output_dir = config["OUTPUT_DIR"], 
        lzp_results = config['LZP_RESULTS'], 
        report_rf = config["REPORT_RF"], 
        rf_folder = config["RF_FOLDER"],
        mlp_results=config["MLP_RESULTS"], 
        best_params_path = config["BEST_PARAMS"],
        epochs=config["EPOCHS"],
        best_model_path=config["BEST_MODEL"],
        )


if __name__ == "__main__":
    main()  
