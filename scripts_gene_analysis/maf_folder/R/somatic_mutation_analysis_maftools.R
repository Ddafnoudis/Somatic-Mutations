### Analysis of somatic variants in ALL, LAML and CLL cancer. ###

# Import maftools library
library(maftools)
# Version of the package
packageVersion("maftools")

# Set working directory
setwd("/Somatic-Mutations/scripts_gene_analysis/datasets/")

# Define the names of the ALL files
all_2015 <- "all_stjude_2015_data_mutations.txt"
all_2016 <- "all_stjude_2016_data_mutations.txt"

# Read the files
all_2015 = read.csv(all_2015, sep = "\t")
all_2016 = read.csv(all_2016, sep = "\t")

# Merge ALL DataFrames
all_df_list <- list(all_2015, all_2016)
all_total <- Reduce(function(x, y) merge(x, y, all=TRUE), all_df_list)


# Define data and read DataFrame
laml_data_mut <- "laml_tcga_pan_can_atlas_2018_data_mutations.txt"
cll_data_mut <- "cll_broad_2015_data_mutations.txt"



### Mutational Landscape Analysis ###
#####################################

# Read files as maf
laml_maf <- read.maf(maf = laml_data_mut)
all_maf <- read.maf(maf = all_total)
cll_maf <- read.maf(maf = cll_data_mut)

# Create a list of the maf files
maf_list = list(all_maf, laml_maf, cll_maf)

# Get summary of samples
getSampleSummary(laml_maf)
getSampleSummary(all_maf)
getSampleSummary(cll_maf)

# Show gene summary
getGeneSummary(laml_maf)
getGeneSummary(all_maf)
getGeneSummary(cll_maf)

# Plot the summary of maf file
sum_LAML <- plotmafSummary(maf = laml_maf, rmOutlier = TRUE, addStat = 'median', dashboard = TRUE, titvRaw = FALSE)
sum_ALL <- plotmafSummary(maf = all_maf, rmOutlier = TRUE, addStat = 'median', dashboard = TRUE, titvRaw = FALSE)
sum_CLL <- plotmafSummary(maf = cll_maf, rmOutlier = TRUE, addStat = 'median', dashboard = TRUE, titvRaw = FALSE)

# Plot the genes that are mutated 
mafbarplot(maf = laml_maf, fontSize = 0.8)
title(main = "LAML mutated genes")
mafbarplot(maf = all_maf, fontSize = 0.8)
title(main = "ALL mutated genes")
mafbarplot(maf = cll_maf, fontSize = 0.8)
title(main = "CLL mutated genes")

# Draw oncoplot (10 first genes)
for(maf in maf_list){
  oncoplot(maf = maf, top = 10)
}

# Classification of SNPs into Transitions and Transversions
for(maf in maf_list){
  maf.titv = titv(maf=maf, plot = FALSE, useSyn = TRUE)
  plotTiTv(res = maf.titv)  
}


# Kataegis plot
for(maf in maf_list){
  rainfallPlot(maf=maf, detectChangePoints = TRUE, pointSize = 0.5)
  }

# Plot the Variant allele Frequencies
for(maf in maf_list){
  plotVaf(maf = maf)
}


#exclusive/co-occurrence event analysis on top 25 mutated genes. 
for(maf in maf_list){
  somaticInteractions(maf = maf, top = 25, pvalue = c(0.05, 0.1), countsFontSize = 0.2)
}


# Detecting cancer driver genes based on positional clustering
for(maf in maf_list){
  maf.sig = oncodrive(maf= maf, AACol = 'Protein_position', minMut = 5, pvalMethod = 'zscore')
  # Plot the oncodrive
  plotOncodrive(res = maf.sig, 
                fdrCutOff = 0.05, 
                bubbleSize = 0.5, 
                useFraction = FALSE, 
                labelSize = 0.5,
                #colCode = "Black"
                )  
}

# Enrichment of known oncogenic pathways
for(maf in maf_list){
  OncogenicPathways(maf, pathways = NULL, fontSize = 1, panelWidths = c(2, 4, 4))
}


