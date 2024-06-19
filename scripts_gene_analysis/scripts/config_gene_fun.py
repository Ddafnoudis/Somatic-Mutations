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
    config['GMT_FOLDER'] = Path(config['GMT_FOLDER'])
    config['GENE_FILES_FOLDER'] = Path(config['GENE_FILES_FOLDER'])
    config['HALLMARK_RESULTS'] = Path(config['HALLMARK_RESULTS'])
    config['CLL_ENR_RESULTS'] = Path(config['CLL_ENR_RESULTS'])

    return config


if __name__ == '__main__':
    parse_configuration_files()
