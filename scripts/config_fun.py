"""
Configuration file parse
"""
import yaml
from typing import Dict
from pathlib import Path

def parse_configuration_files(fname) -> Dict[int, str]:
    """
    Define the seed and the path to the datasets from the
    configuration file
    """
    # Load the configuration file
    with open(fname) as stream:
        config = yaml.safe_load(stream)
    
    config['WORK_GENE_DIR'] = Path(config['WORK_GENE_DIR'])
    config['DATASET'] = Path(config['DATASET'])
    config["CORR_FOLDER"] = Path(config["CORR_FOLDER"])
    config["DATASET_SOMATIC_MUTATION"] = Path(config['DATASET_SOMATIC_MUTATION'])
    config["SIGNIFICANT_THRESHOLD"] = float(config["SIGNIFICANT_THRESHOLD"])
    config['GENE_FILES_FOLDER'] = Path(config['GENE_FILES_FOLDER'])
    config['HALLMARK_RESULTS'] = Path(config['HALLMARK_RESULTS'])
    config["ENRICH_FOLDER"] = Path(config["ENRICH_FOLDER"])
    config["ENR_15_FOLDER"] = Path(config["ENR_15_FOLDER"])
    config["ENR_PLOTS"] = Path(config["ENR_PLOTS"])
    config["AML_ENRICH"] = Path(config["AML_ENRICH"])
    config["AML_ENRICH_15"] = Path(config["AML_ENRICH_15"])
    config["AML_PLOT"] = Path(config["AML_PLOT"])
    config['SEED'] = int(config['SEED'])
    config['WORK_DIR'] = Path(config['WORK_DIR'])
    config['DATASET'] = Path(config['DATASET'])
    config['OUTPUT_DIR'] = Path(config['OUTPUT_DIR'])
    config['CORR_IMAGE'] = Path(config['CORR_IMAGE'])
    config['CORR_RESULTS'] = Path(config['CORR_RESULTS'])
    config['LZP_RESULTS'] = Path(config['LZP_RESULTS'])
    config['ANOVA'] = Path(config["ANOVA"])
    config["RF_FOLDER"] = Path(config["RF_FOLDER"])
    config['MIT'] = Path(config["MIT"])
    config['CROSS_VAL'] = Path(config['CROSS_VAL'])
    config['ACCURACY'] = Path(config['ACCURACY'])
    config['REPORT_RF'] = Path(config['REPORT_RF'])
    config['CONFUSION_MTX'] = Path(config['CONFUSION_MTX'])
    config["MLP_RESULTS"] = Path(config["MLP_RESULTS"])
    
    return config


if __name__ == '__main__':
    parse_configuration_files()
