# StackCBEmbed
## Introduction 

Nucleic acids, proteins, carbohydrates and lipids are four important molecules for any organisms, among them carbohydrates come after the DNA and proteins and which is thought about the third important molecule of life. The carbohydrates communicate with other protein molecules and these protein-carbohydrate interactions have several roles in different biological processes (cellular adhesion, cellular recognition, protein folding, subcellular localization, ligand recognition as well as in human body. Moreover, they provide a protection of human cell against pathogens as well as they play an important role as biomarkers or drug targets.

In order to identify protein-carbohydrate interactions, there are several experimental techniques have been performed although weak binding affinity and synthetic complexity of individual carbohydrates has made the study more expensive, time consuming and challenging. Therefore, developing a computational technique for effectively predicting protein-carbohydrate binding sites has become an urgent necessity. The computational approaches concentrate on locating the sites of proteins that bind to carbohydrates. 

The computational studies involve different structure based and sequence-based methods. There are several structured based methods which have used to predict binding sites from a known protein structure. The problem of above structed-based techniques is that they are dependent on protein structures that are often not available.

In our thesis, we have proposed a computational sequence-based ensemble machine learning model named 'StackCBEmbed' to effectively classify protein-carbohydrate binding interactions at residue level from sequence-known proteins. For our research, we made use of the benchmark dataset and two separate testing datasets originating from the StackCBPred[[2]](#2) method. These datasets consist of carbohydrate-binding proteins with high-resolution data. By applying the Incremental Feature Selection approach, we extracted essential sequence-based features and selected the most effective ones. Additionally, we incorporated embedding features from the pre-trained transformer-based language model called 'T5-XL-Uniref50'. To the best of our knowledge, ours is the first attempt to apply protein language model in predicting protein-carbohydrate binding interactions. 

In the base layer of our ensemble methodology, a group of predictors (ET, XGB, and SVM) was trained using the incrementally selected sequence-based hand generated features and features from the protein language model. The selection of these base classifiers was guided by their average information gain score. The outcomes generated by these base classifiers were merged with the original features and subsequently employed as the input for the meta-layer predictor (XGB).

## Graphical abstract
[base_meta_layer_architecture.pdf](https://github.com/farah5112github/StackCBEmbed/files/12362675/base_meta_layer_architecture.pdf)

## References 
<a id="1">[1]</a>
Taherzadeh, Ghazaleh, et al. "Sequence-based prediction of protein–carbohydrate binding sites using support vector machines." Journal of chemical information and modeling 56.10 (2016): 2115-2122.<br />

<a id="2">[2]</a>
Gattani, Suraj, Avdesh Mishra, and Md Tamjidul Hoque. "StackCBPred: A stacking based prediction of protein-carbohydrate binding sites from sequence." Carbohydrate research 486 (2019): 107857.
