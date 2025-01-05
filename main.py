"""
"""
from scripts.config_fun import parse_configuration_files
from scripts.condition_statements import condition_statement


def main():
    config = parse_configuration_files(fname='configuration.yaml')

    condition_statement(
        seed=config["SEED"], data=config["DATASET"], 
        output_dir = config["OUTPUT_DIR"], corr_image =config["CORR_IMAGE"],
        corr_results =config['CORR_RESULTS'], lzp_results = config['LZP_RESULTS'], 
        report_rf = config["REPORT_RF"], mlp_results=config["MLP_RESULTS"],
        best_params = config["BEST_PARAMS"], rf_parameters=config["RF_PARAMS"],
        epochs=config["EPOCHS"], param_grid=config["PARAM_GRID"]
        )


if __name__ == "__main__":
    main()  
