---
layout: post
title:  "Unsupervised Clustering methods to identify clusters in Patients"
info: "Explore different clustering techniques - K-Means, GMM, K-Modes and K-Prototypes to segregate patients data and visualize it using PCA decomposition"
tech: "python"
img: "/assets/img/clustering/main.png" 
img_dimensions: ["300","300"]
tags: "ML"
concepts: "Unsupervised Learning, Clustering"
type: "project"
img: "/assets/img/clustering/main.png"
tags: ["Visualization", "Clustering"]
---


## NHAMCS-

Applying supervised and unsupervised clustering techniques healthcare data from
- NHAMCS dataset : stratified survey sampled dataset. Clustering techniques can be beneficial in drawing out insights from the genberal population.
- Yale dataset.

The primary purpose of this study is to identify clusters of patients data using anecdotal as well as analytical heuristics.  
The plots generated from this study are presented in a Principal Component Analysis (PCA) plot for the sake of visual clarity.   

This study narrows down from a general approach to a specific one by analyzing specific health condition.
Triage phenotype data is used for this study:

__Heuristics methods applied:__ <br>
__Hopkins test__: A statistical test which allow to guess if the data follow an uniform distribution. If the test is positve, (an hopkins score which tends to 0) it means that clustering is useless for the dataset. A score between 0 and 1, a score around 0.5 express no clusterability and a score tending to 0 express a high cluster tendency.<br>

__Elbow method__:
"Elbow" or "knee of a curve" as a cutoff point is a common heuristic in mathematical optimization to choose a point where diminishing returns are no longer worth the additional cost.

Patient clustering techniques:
- PCA: 
Condense numerical features into two Principal Components to cluster data to be fed in for visualization purposes. Two principle components were able to capture most of the variation in the data so I found it suitable for visualization purposes. Also apply _known classes/labels_ for cluster analysis. <br>
- K-Means: 
Apply K-Means on numerical data and present the clusters in a PCA plot.<br>
- GMM Clustering
Apply GMM Clustering on numerical data and present the clusters in a PCA plot.<br>
- K-Modes:
Nominal and ordinal categorical data clustered and presented in a PCA plot.<br>
- K-Prototypes: euclidean distance in the numerical features whereas dissimilarity measures in categorical features.


<img src="/assets/img/clustering/gmm.png"> 
<br>
<img src="/assets/img/clustering/main.png"> 




__Datasets__:
"Hong WS, Haimovich AD, Taylor RA (2018) Predicting hospital admission at emergency department triage using machine learning. PLoS ONE 13(7): e0201016." (https://doi.org/10.1371/journal.pone.0201016)





