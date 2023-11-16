import numpy as np;
import math;
import pandas as pd;
import csv;
import time;

# output :
def fill_up_total_output_vector(seq, index, feature_vector_total_output):
    for j in list(range(0, len(seq))):
        feature_vector_total_output[index + j, 0] = seq[j]
def find_class_label(total_output):

    total_rows_output = 0;
    for i in list(range(0, len(total_output))):
        total_rows_output = total_rows_output + len(total_output[i])

    # create the numpy array for output named "feature_vector_total_output ":
    feature_vector_total_output = np.zeros((total_rows_output, 1), dtype=int)

    index = 0
    for i in list(range(0, len(total_output))):
        fill_up_total_output_vector(total_output[i], index, feature_vector_total_output)
        index = index + len(total_output[i])

    print('output : ', feature_vector_total_output.shape)

    return feature_vector_total_output;

# PSSM : 20 ;
def extract_PSSM(file_path):
    f = open(file_path, 'r')
    file = f.readlines()
    file = file[3:-6]

    PSSM_indi = np.zeros((len(file), 20), dtype=float);

    for i in range(0, len(file)):
        line = file[i]
        line = line.split(' ')
        s = []

        for ele in line:
            if ele.strip():
                s.append(ele)
        s = s[2:-22]

    for j in range(0, 20):
        PSSM_indi[i, j] = float(s[j]);

    return PSSM_indi;

def find_individual_PSSM_files(file_path,PSSM_total):

    PSSM_indi = extract_PSSM(file_path);

    for i in range(0,len(PSSM_indi)):
        for j in range(0, 20):
            PSSM_indi[i, j] = PSSM_indi[i, j] / 9;

    PSSM_total = np.concatenate(( PSSM_total,PSSM_indi ),axis=0);
    return PSSM_total;

def find_individual_embedding_files(file_path):
    D_feature = pd.read_csv(file_path, header=None)
    embedding_total = D_feature.iloc[:, 1:].values

    return embedding_total;


