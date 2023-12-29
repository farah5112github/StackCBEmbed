# StackCBEmbed
## Introduction 

Carbohydrates, the third essential molecules after DNA and proteins, interact with proteins, influencing diverse biological processes and providing protection against pathogens. They also serve as biomarkers or drug targets in the human body.

Due to challenges in experimental techniques, there is a pressing need for developing computational methods to predict protein-carbohydrate binding sites. Computational studies use diverse structure-based and sequence-based methods, with some structured-based techniques relying on frequently unavailable protein structures.

We introduced 'StackCBEmbed,' a computational sequence-based ensemble model, to classify protein-carbohydrate binding interactions at the residue level, utilizing benchmark and testing datasets with high-resolution data. Applying Incremental Feature Selection, we extracted essential sequence-based features and incorporated embedding features from the 'T5-XL-Uniref50' language model, marking the first attempt to apply a protein language model in predicting these interactions. In our ensemble method, base predictors (ET, XGB, SVM), guided by average information gain scores, were trained on selected sequence-based and protein language model features, with their outcomes merged into original features for input into the meta-layer predictor (XGB).

## Graphical abstract
![my_new_diagram](https://github.com/farah5112github/StackCBEmbed/assets/60771070/10e0001e-6a61-4b76-ac7e-2c0c2922b393)

## Environmental Setup :
**Programming Language :** Python 3.10.12 <br />
**Machine Learning based library :** Scikitlearn 1.2.2 <br />
**IDE for python :** PyCharm 2023.3.2 <br />
**All libraries :** xgboost==2.0.3
scops==0.9.0
pickle5==0.0.11
scikit-learn==1.2.2
numpy==1.26.2
pandas==2.1.4
matplotlib==3.8.2
PyQt5==5.15.10
numpy==1.26.2
matplotlib==3.8.2
skops==0.9.0
pandas==2.1.4
imblearn==0.0

## Dataset (id, no of sequence, seqiences and target output):
**Benchmark(Training and validation) set:** https://drive.google.com/file/d/1lDSVqmTT2wNRjv4JKlwHIMDAZ1_9IOPD/view?usp=sharing <br />
**TS49 :** https://drive.google.com/file/d/1RfaecHJGfSsp3ny9-TE4xcc_HULiSpnc/view?usp=drive_link <br />
**TS88 :** https://drive.google.com/file/d/1EKf7vBQypdcLxFVi8etZAZOdkCUcfraU/view?usp=drive_link

## Input Requirements
- You need to place the test protein sequences, along with their IDs, into the text file (StackCBEmbed\input_files\input.txt) in FASTA format to obtain predictions for protein-carbohydrate 
  binding sites. It's possible to include several protein sequences simultaneously. Examples of these protein sequences are provided in the 'input.txt' file.
- [Optional] The PSSM file (.PSSM) containing evolution-derived information for the protein sequence mentioned in the .txt file.
  You need to obtain the PSSM file from 'PSI-BLAST'. Download link: https://blast.ncbi.nlm.nih.gov/Blast.cgi.
- You need to extract the protein sequence's features using the ProtT5-XL-UniRef50 pretrained model and save the features in a .csv file.
  Code link: https://github.com/agemagician/ProtTrans/blob/master/Embedding/TensorFlow/Advanced/ProtT5-XL-UniRef50.ipynb.


## Run the StackCBEmbed model
1. Go to the command prompt.
2. Write the following command in the command prompt to download the 'StackCBEmbed' repository.
   ```plaintext
   git clone "https://github.com/farah5112github/StackCBEmbed.git"
3. Navigate to the 'input_files' folder within the 'StackCBEmbed' directory.
4. Place all the required input files into the 'input_files' folder.
5. Substitute the existing protein sequence in the input.txt file with your own sequence. 
6. Proceed to the 'StackCBEmbed codes' folder and execute main.py from the command prompt using the following command. 
   ```plaintext
   python main.py
7. During execution, you will be prompted to choose between using the original model or the model with embeddings only.
8. The output files will be created in the 'StackCBEmbed' directory.
