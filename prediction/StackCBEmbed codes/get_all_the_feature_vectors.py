import numpy as np;
import math;
import pandas as pd;
import csv;
import time;

# output :
def find_class_label(total_output):
    # create the numpy array for output named "feature_vector_total_output ":
    feature_vector_total_output = []

    for i in list(range(0, len(total_output))):
        feature_vector_total_output.append([total_output[i]])
    feature_vector_total_output = np.array(feature_vector_total_output)
    print('output : ', feature_vector_total_output.shape)

    return feature_vector_total_output

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
            PSSM_indi[i, j] = float(s[j])

    return PSSM_indi

def find_individual_PSSM_files(file_path):

    PSSM_indi = extract_PSSM(file_path);

    for i in range(0,len(PSSM_indi)):
        for j in range(0, 20):
            PSSM_indi[i, j] = PSSM_indi[i, j] / 9

    return PSSM_indi

def find_individual_embedding_files(file_path):
    D_feature = pd.read_csv(file_path, header=None)
    embedding_total = D_feature.iloc[:, 1:].values

    return embedding_total


