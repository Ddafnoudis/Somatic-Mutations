# Project: Interpretative analysis of human somatic mutations in Leukemia.
![output](https://github.com/user-attachments/assets/6465134d-f158-499b-8ed5-8cdaab94bba9)

## Introduction
This project is separated in two sections:

The first section provides an analysis of somatic variants using datasets from cBioportal. The Mutation Annotation Format (MAF) file allows us to use tools such as maftools to summarize and visualize the mutations of leukemia.

The second section includes the application of Machine Learning algorithms, optimization and fine-tuning of Multi-Linear Perceptron model for the classification of patients with Leukemia.

## Data source
- [cBioPortal](https://www.cbioportal.org/datasets)

## Python version
3.12.3

## Requirements (Python)
 ```bash
numpy==1.26.4
pandas==2.1.4
tensorflow==2.16.1
keras==3.3.3
keras-utils==1.0.13
matplotlib==3.8.4
plotly==5.9.0
scikit-learn==1.4.2
scikit-optimize==0.10.2
pyaml==25.1.0           
scipy==1.11.4
lightning==2.5.1
lazypredict==0.2.12                
seaborn==0.12.2                  
```

## R version
4.2.3

## Requirements (R)
```bash
maftools=2.14.0
```

## Install requirements
```bash
pip install -r requirements.txt
```

> [!NOTE] 
> The ``extract_files.sh`` is a sript in bash. Provide the full path of the stored data and the full path of the output directory. The output directory should be the folder with the extracted data files (mutations and clinical sample).
