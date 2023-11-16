# StackCBEmbed
## Introduction 

Carbohydrates, considered the third essential molecules after DNA and proteins, interact with proteins, influencing diverse biological processes such as cellular adhesion, recognition, folding, localization, and ligand recognition. They also provide protection against pathogens and serve as biomarkers or drug targets in the human body.

Due to challenges in experimental techniques, including weak binding affinity and synthetic complexity, there is a pressing need for developing computational methods to predict protein-carbohydrate binding sites, focusing on identifying protein sites that bind to carbohydrates. Computational studies employ diverse structure-based and sequence-based methods, with the limitation that some structured-based techniques depend on frequently unavailable protein structures.

We introduced 'StackCBEmbed,' a computational sequence-based ensemble model, to classify protein-carbohydrate binding interactions at the residue level, utilizing benchmark and testing datasets with high-resolution data. Applying Incremental Feature Selection, we extracted essential sequence-based features and incorporated embedding features from the 'T5-XL-Uniref50' language model, marking the first attempt to apply a protein language model in predicting these interactions. In our ensemble method, base predictors (ET, XGB, SVM), guided by average information gain scores, were trained on selected sequence-based and protein language model features, with their outcomes merged into original features for input into the meta-layer predictor (XGB).

## Graphical abstract
![framework](https://github.com/farah5112github/StackCBEmbed/assets/60771070/227c4f1e-1e87-4eef-8d81-8aa4013f6f1f)

![my_diagram](https://github.com/farah5112github/StackCBEmbed/files/13382446/my_diagram.pdf)


## Environmental Setup :
**Programming Language :** Python 3.10.4 <br />
**Machine Learning based library :** Scikitlearn 1.2.2 <br />
**IDE for python :** PyCharm 2021.3.3 <br />
**Other libraries :** Numpy, pandas, matplotlib, pickle etc.

## Input Requirements
- A text file (input.txt) containing the protein sequence in FASTA format.
- The PSSM file (.PSSM) containing evolution-derived information for the protein sequence mentioned in the .txt file.
  You need to obtain the PSSM file from 'PSI-BLAST'. Download link: https://blast.ncbi.nlm.nih.gov/Blast.cgi.
- The Spider file (.spd33) containing secondary structure information for the protein sequence mentioned in the .txt file.
  To obtain the Spider file, use 'SPIDER3'. Server link: https://sparks-lab.org/server/spider3/.
- The SPINE-X file (.spXout) contains secondary structure information, predictions of solvent accessible surface area, and
  backbone torsion angles for the protein sequence mentioned in the .txt file. Server link: http://sparks.informatics.iupui.edu/.
- You need to extract the protein sequence's features using the ProtT5-XL-UniRef50 pretrained model and save the features in a .csv file.
  Code link: https://github.com/agemagician/ProtTrans/blob/master/Embedding/TensorFlow/Advanced/ProtT5-XL-UniRef50.ipynb.


## Run the StackCBEmbed model
1. Go to the command prompt.
2. Write the following command in the command prompt to download the 'StackCBEmbed' repository.
   ```plaintext
   git clone "https://github.com/farah5112github/StackCBEmbed.git"
3. Navigate to the 'input_files' folder within the 'StackCBEmbed' directory.
4. Place all the required input files into the 'input_files' folder.
5. Substitute the existing protein sequence in the input.txt file with your own sequence. Note that StackCBEmbed can only handle a single sequence.
6. Proceed to the 'StackCBEmbed codes' folder and execute main.py from command prompt. During execution, you will be prompted to enter the absolute path of the StackCBEmbed folder (e.g., C:/Users/quazi/Desktop/StackCBEmbed/). Additionally, you will be asked to specify your choice between using a model with only embeddings or a model with both embeddings and PSSM.
    ```plaintext
   python main.py
7. The output file named 'output.csv' will be created in the 'StackCBEmbed' directory.

## References 
<a id="1">[1]</a>
Taherzadeh, Ghazaleh, et al. "Sequence-based prediction of protein–carbohydrate binding sites using support vector machines." Journal of chemical information and modeling 56.10 (2016): 2115-2122.<br />

<a id="2">[2]</a>
Gattani, Suraj, Avdesh Mishra, and Md Tamjidul Hoque. "StackCBPred: A stacking based prediction of protein-carbohydrate binding sites from sequence." Carbohydrate research 486 (2019): 107857.
