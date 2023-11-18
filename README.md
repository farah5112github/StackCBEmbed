# StackCBEmbed
## Introduction 

Carbohydrates, the third essential molecules after DNA and proteins, interact with proteins, influencing diverse biological processes and providing protection against pathogens. They also serve as biomarkers or drug targets in the human body.

Due to challenges in experimental techniques, there is a pressing need for developing computational methods to predict protein-carbohydrate binding sites. Computational studies use diverse structure-based and sequence-based methods, with some structured-based techniques relying on frequently unavailable protein structures.

We introduced 'StackCBEmbed,' a computational sequence-based ensemble model, to classify protein-carbohydrate binding interactions at the residue level, utilizing benchmark and testing datasets with high-resolution data. Applying Incremental Feature Selection, we extracted essential sequence-based features and incorporated embedding features from the 'T5-XL-Uniref50' language model, marking the first attempt to apply a protein language model in predicting these interactions. In our ensemble method, base predictors (ET, XGB, SVM), guided by average information gain scores, were trained on selected sequence-based and protein language model features, with their outcomes merged into original features for input into the meta-layer predictor (XGB).

## Graphical abstract
![my_new_diagram](https://github.com/farah5112github/StackCBEmbed/assets/60771070/10e0001e-6a61-4b76-ac7e-2c0c2922b393)

## Environmental Setup :
**Programming Language :** Python 3.10.4 <br />
**Machine Learning based library :** Scikitlearn 1.2.2 <br />
**IDE for python :** PyCharm 2021.3.3 <br />
**Other libraries :** Numpy, pandas, matplotlib, pickle etc.

## Input Requirements
- A text file (input.txt) containing the protein sequence in FASTA format.
- The PSSM file (.PSSM) containing evolution-derived information for the protein sequence mentioned in the .txt file.
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
7. During execution, you will be prompted to enter your choice between using a model with only embeddings or a model with both embeddings and PSSM.
8. The output files will be created in the 'StackCBEmbed' directory.
