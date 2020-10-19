---
layout: post
title:  "Use Unsupervised Clustering methods to identify clusters in Patients"
info: "Explore different clustering techniques - K-Means, GMM, K-Modes and K-Prototypes to segregate patients data and visualize it using PCA decomposition"
tech: "python, spacy, nlt"
type: A Company
img: "/assets/img/clustering/main.png" 
img_dimensions: ["300","300"]
tags: "ML"
concepts: "Unsupervised Learning, Clustering"
type: "project"
img: "/assets/img/clustering/main.png"
---

  <div style="text-align: center">
  <i class="fa fa-code"></i> <a  href="https://www.google.com">Code to this Repo</a>
  </div>

<br>
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

Patient clustering techniques:
- PCA: 
Condense numerical features into two Principal Components to cluster data to be fed in for visualization purposes. Also apply _known classes/labels_ for cluster analysis. <br>
- K-Means: 
Apply K-Means on numerical data and present the clusters in a PCA plot.<br>
- GMM Clustering
Apply GMM Clustering on numerical data and present the clusters in a PCA plot.<br>
- K-Modes:
Nominal and ordinal categorical data clustered and presented in a PCA plot.<br>

- K-Prototypes: euclidean distance in the numerical features whereas dissimilarity measures in categorical features.
```python
from kmodes.kprototypes import KPrototypes
```

<img src="/assets/img/clustering/main.png"> 


Papers:

__Citation for secondary dataset__:
"Hong WS, Haimovich AD, Taylor RA (2018) Predicting hospital admission at emergency department triage using machine learning. PLoS ONE 13(7): e0201016." (https://doi.org/10.1371/journal.pone.0201016)


http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.474.8181&rep=rep1&type=pdf

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.15.4028&rep=rep1&type=pdf
