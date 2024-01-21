# StackCBEmbed
## Introduction 

Carbohydrates, the third essential molecules after DNA and proteins, interact with proteins, influencing diverse biological processes and providing protection against pathogens. They also serve as biomarkers or drug targets in the human body.

Due to challenges in experimental techniques, there is a pressing need for developing computational methods to predict protein-carbohydrate binding sites. Computational studies use diverse structure-based and sequence-based methods, with some structured-based techniques relying on frequently unavailable protein structures.

We introduced 'StackCBEmbed,' a computational sequence-based ensemble model, to classify protein-carbohydrate binding interactions at the residue level, utilizing benchmark and testing datasets with high-resolution data. Applying Incremental Feature Selection, we extracted essential sequence-based features and incorporated embedding features from the 'T5-XL-Uniref50' language model, marking the first attempt to apply a protein language model in predicting these interactions. In our ensemble method, base predictors (SVM, MLP, ET and PLS), guided by average information gain scores, were trained on selected sequence-based and protein language model features, with their outcomes merged into original features for input into the meta-layer predictor (SVM).

## Graphical abstract of Training
![training](https://github.com/farah5112github/StackCBEmbed/blob/main/training.png)

## Graphical abstract of Prediction
![training](https://github.com/farah5112github/StackCBEmbed/blob/main/prediction.png)

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
skops==0.9.0
pandas==2.1.4
imblearn==0.0

## Dataset (protein_id, no of sequences, sequences and target output):
**Benchmark(Training and validation) set:** https://drive.google.com/file/d/1lDSVqmTT2wNRjv4JKlwHIMDAZ1_9IOPD/view?usp=sharing <br />
**TS49 :** https://drive.google.com/file/d/1RfaecHJGfSsp3ny9-TE4xcc_HULiSpnc/view?usp=drive_link <br />
**TS88 :** https://drive.google.com/file/d/1EKf7vBQypdcLxFVi8etZAZOdkCUcfraU/view?usp=drive_link

## Dataset PSSM:
**Benchmark(Training and validation) set:** https://drive.google.com/drive/folders/1aqwOzRwhjQ_dg2UPOeKcD7VAaYyZDJP1?usp=drive_link <br />
**TS49 :** https://drive.google.com/drive/folders/10riCKVAfq4pdMH3G10iTh35e02PFiM0z <br />
**TS88 :** https://drive.google.com/drive/folders/1WMNEPerLjyr4C8g-ARFt1lnNvrxlJ_na

## Dataset Embedding:
**Benchmark(Training and validation) set:** https://drive.google.com/drive/folders/1pVsBDg6p0bMd6aKleMPhMMS1mIfEbRmF <br />
**TS49 :** https://drive.google.com/drive/folders/1xWvgVNQNsfV6JM9GZPr9hv5rBVJ5yMsB <br />
**TS88 :** https://drive.google.com/drive/folders/17jZS-ygWVD-D2mJsFLLdnQv6IcedHQ0k

## Input Requirements
- You need to place the test protein sequences, along with their IDs, into the text file (StackCBEmbed\input_files\input.txt) in FASTA format to obtain predictions for protein-carbohydrate 
  binding sites. It's possible to include several protein sequences simultaneously. Examples of these protein sequences are provided in the 'input.txt' file.
- [Optional] The PSSM file (.PSSM) containing evolution-derived information for the protein sequence mentioned in the .txt file. <br />
  You need to obtain the PSSM file from 'PSI-BLAST'. Download link: https://blast.ncbi.nlm.nih.gov/Blast.cgi. <br />
  Details for how to retrieve the PSSM files: https://github.com/mrzResearchArena/BLAST <br />
  PSSM generation is costly.
- You need to extract the protein sequence's features using the ProtT5-XL-UniRef50 pretrained model of ProtTrans paper and save the features in a .csv file. <br />
  Code link(Requires collab pro for its high RAM requirement): https://github.com/agemagician/ProtTrans/blob/master/Embedding/TensorFlow/Advanced/ProtT5-XL-UniRef50.ipynb.
  Link to paper: https://pubmed.ncbi.nlm.nih.gov/34232869/ <br />
  (ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning - PubMed (nih.gov))


## Run the StackCBEmbed model
1. Go to the command prompt.
2. Write the following command in the command prompt to download the 'StackCBEmbed' repository.
   ```plaintext
   git clone "https://github.com/farah5112github/StackCBEmbed.git"
3. Navigate to the 'output_csvs' folder within the 'StackCBEmbed' directories(table_2_generation, table_3_generation, etc.). There you will find the target csvs for the paper.
4. Navigate to the 'all_required_csvs' folder within the 'StackCBEmbed' directories(table_2_generation, table_3_generation, etc.).Place all the required files into the 'all_required_csvs' folder if you want to run the main.py. main.py will generate the csv files. The "readme.txt" file in each 'all_required_csvs' folder has all the required csv list and download link.
5. For prediction folder,place all the required input files into the 'input_files' folder.
6. Substitute the existing protein sequence in the input.txt file with your own sequence. 
7. Proceed to the 'StackCBEmbed codes' folder and execute main.py from the command prompt using the following command. 
   ```plaintext
   python main.py
8. During execution, you will be prompted to choose between using the original model or the model with embeddings only.
9. The output files will be created in the 'output_csvs' directory.
