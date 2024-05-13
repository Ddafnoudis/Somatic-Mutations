"""
"""
from pathlib import Path
from scripts.config_fun import parse_configuration_files
from scripts.condition_statements import condition_statement


def main():
    config = parse_configuration_files(fname='configuration.yaml')

    condition_statement(seed=config["SEED"], working_dir=config["WORK_DIR"], 
                        data=config["DATASET"], output_dir = config["OUTPUT_DIR"],
                        corr_image =config["CORR_IMAGE"], corr_results =config['CORR_RESULTS'],
                        lzp_results = config['LZP_RESULTS'], anova = config["ANOVA"], 
                        mit = config["MIT"], cross_val = config['CROSS_VAL'], 
                        accuracy = config['ACCURACY'], report_rf = config["REPORT_RF"],
                        confusion_mtx = config["CONFUSION_MTX"], mlp_results=config["MLP_RESULTS"])


if __name__ == "__main__":
    main()  
