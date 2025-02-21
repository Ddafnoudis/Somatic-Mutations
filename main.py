"""
"""
from scripts.config_fun import parse_configuration_files
from scripts.condition_statements import condition_statement


def main():
    config = parse_configuration_files(fname='configuration.yaml')

    condition_statement(
        working_gene_dir=config["WORK_GENE_DIR"],
        dataset_somatic_mutation=config["DATASET_SOMATIC_MUTATION"],
        dataset=config['DATASET'], 
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
        corr_image =config["CORR_IMAGE"],
        corr_results =config['CORR_RESULTS'],
        lzp_results = config['LZP_RESULTS'], 
        report_rf = config["REPORT_RF"], 
        rf_best_parameters=config["RF_BEST_PARAMS"],
        mlp_results=config["MLP_RESULTS"], 
        best_params = config["BEST_PARAMS"],
        rf_parameters=config["RF_PARAMS"], 
        epochs=config["EPOCHS"], 
        param_grid=config["PARAM_GRID"]
        )


if __name__ == "__main__":
    main()  
