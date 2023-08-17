import numpy as np;
import math;
import pandas as pd;
import csv;
import time;

# window size 5 :
def convert_feature_vector_to_window_size(X, y, ws):
    converted_feature_vector_total = np.zeros((1, ws * X.shape[1]), dtype=float);


    start = int(ws / 2);
    end = (len(X) - 1) + int(ws / 2);

    dummy = np.zeros((int(ws / 2), X.shape[1]), dtype=float);
    X = np.concatenate((dummy, X, dummy), axis=0)
    y = y.reshape(len(y), 1)


    start_time = time.time();

    for i in range(start, end + 1):
        temp = [1];
        temp = np.array(temp).reshape(1, 1);

        for j in range(i - int(ws / 2), i + int(ws / 2) + 1):
            neighbor = X[j, :].reshape(1, len(X[j, :]))
            # print('neighbor : ',neighbor.shape)

            temp = np.concatenate((temp, neighbor), axis=1)

        temp = np.delete(temp, 0, axis=1)


        converted_feature_vector_total = np.concatenate((converted_feature_vector_total, temp), axis=0)


    end_time = time.time();

    converted_feature_vector_total = np.delete(converted_feature_vector_total, 1, axis=0)
    converted_feature_vector_total = np.concatenate((y, converted_feature_vector_total), axis=1)
    print(converted_feature_vector_total.shape)

    print('total time required : ', (end_time - start_time) / 60, ' times');

    return converted_feature_vector_total;


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

# MG : 1 ;
def find_individual_MG_files(file_path,PSSM_monogram_total,total_input):

    PSSM_indi = extract_PSSM(file_path);

    r,c =  PSSM_indi.shape ;
    PSSM_monogram_indi = np.zeros( ( r, 1 ),dtype=float) ;

    residue_values = [] ;
    residue_with_monogram = {'A':0,'R':0,'N':0,'D':0,'C':0,'Q':0,'E':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0} ;

    for col in range(0,c):
        temp = 0;

        for row in range(0,r):
            temp += PSSM_indi[row,col] ;

        residue_values.append( temp / math.exp(6) ) ;

    #print(residue_values) ;
    residue_with_monogram = {'A':residue_values[0],'R':residue_values[1],'N':residue_values[2],
                             'D':residue_values[3],'C':residue_values[4],'Q':residue_values[5],
                             'E':residue_values[6],'G':residue_values[7],'H':residue_values[8],
                             'I':residue_values[9],'L':residue_values[10],'K':residue_values[11],
                             'M':residue_values[12],'F':residue_values[13],'P':residue_values[14],
                             'S':residue_values[15],'T':residue_values[16],'W':residue_values[17],
                             'Y':residue_values[18],'V':residue_values[19]} ;

    for r in range(0,len(total_input)):
        x = total_input[r] ;
        print(x) ;

        for i in range(0,len(x)):
            PSSM_monogram_indi[i,0] = residue_with_monogram[ x[i] ] ;

    PSSM_monogram_total = np.concatenate((PSSM_monogram_total,PSSM_monogram_indi),axis=0) ;
    return PSSM_monogram_total;

# ASA : 1 ;
def find_individual_ASA_files(ASA_file_path, ASA_total, total_input):

    f = open(ASA_file_path,'r')
    file = f.readlines()
    file = file[1:]

    ASA_indi = np.zeros( ( len(file),1  ),dtype=float) ;
    rsa_confirm = {'A' : 110.2 , 'R' : 229.0, 'N':146.4, 'D':144.1, 'C':140.4, 'Q':178.6,
               'E':174.7,'G':78.7,'H':181.9,'I':185.0,'L':183.1,'K':205.7,'M':200.1,
               'F':200.7,'P':141.9,'S':117.2,'T':138.7,'W':240.5,'Y':213.7,'V': 153.7} ;

    for i in range( 0,len(file) ):
        line = file[i]
        line = line.split(' ')
        s = []

        for ele in line:
            if ele.strip():
               s.append(ele)
        s = s[10]

        x = total_input[0][i] ;
        ASA_indi[i,0] = 2 * ( float(s) / rsa_confirm[x] ) - 1 ;

    ASA_total = np.concatenate((  ASA_total, ASA_indi ), axis=0) ;
    return ASA_total;

# torsion angles, HSE : 2 + 2 ;
def find_spider_info_SS(file_path,torsion_angles_total,HSE_total):

    f = open(file_path,'r')
    file = f.readlines()
    len_file = len(file) - 1

    torsion_angles_indi = np.zeros((len_file,2),dtype=float)
    HSE_indi = np.zeros((len_file,2),dtype=float)


    count = 0

    for i in list(range(1,len(file))):
        s = file[i]
        s = s.split(' ')

        for num in s[:]:  #iterate over a shallow copy
            if num == '':
               s.remove(num)

        # torsion angle :
        x = s[4:6]
        for j in list(range(0,2)):
            torsion_angles_indi[count,j] =  float(x[j])  ;

        # HSE :
        x = s[8:10]
        for j in list(range(0,2)):
            HSE_indi[count,j] =  float(x[j])  ;

        count = count + 1

    torsion_angles_total = np.concatenate((  torsion_angles_total, torsion_angles_indi ),axis=0) ;
    HSE_total = np.concatenate((  HSE_total, HSE_indi ),axis=0) ;

    return torsion_angles_total,HSE_total ;

# DPC : 400 ;

def find_DPC_PSSM_with_window_total(PSSM_norm_without_window_indi,DPC_PSSM_total,ws):

    o_length = len(PSSM_norm_without_window_indi)
    DPC_PSSM_indi = np.zeros((o_length,400), dtype=float) ;

    start_index = 0 + ws
    end_index = ( o_length - 1 ) + ws

    dummy = np.zeros((ws,20),dtype=object)

    PSSM_norm_without_window_indi_temp = PSSM_norm_without_window_indi.view()
    PSSM_norm_without_window_indi_temp = np.concatenate((dummy,PSSM_norm_without_window_indi_temp),axis=0)
    PSSM_norm_without_window_indi_temp = np.concatenate(( PSSM_norm_without_window_indi_temp ,dummy),axis=0)


    for residue in list(range(start_index,end_index+1)):

        #new part :
        col_index = 0;

        for i in range(0,20):

            for j in range(0,20):
                sum_k = 0;

                for temp in range(ws*(-1),ws):

                    k = residue + temp;
                    sum_k += PSSM_norm_without_window_indi_temp[k,i] * PSSM_norm_without_window_indi_temp[k+1,j];

                DPC_PSSM_indi[residue-ws,col_index] = sum_k;
                col_index = col_index + 1;

    #DPC_PSSM_total = np.concatenate(    (  DPC_PSSM_total, DPC_PSSM_indi ),axis=0  ) ;
    DPC_PSSM_total = DPC_PSSM_indi.copy() ;
    return DPC_PSSM_total;

def find_individual_DPC_files(file_path, DPC_PSSM_total,ws):

    PSSM_indi = extract_PSSM(file_path);
    PSSM_norm_without_window_indi = np.zeros((len(PSSM_indi),20),dtype=float)

    z_x = np.amax(PSSM_indi)
    z_y = np.amin(PSSM_indi)
    r,c = PSSM_norm_without_window_indi.shape

    for row in list(range(0,r)):
        for col in list(range(0,c)):
            PSSM_norm_without_window_indi[row,col] = ( PSSM_indi[row,col] - z_y ) / (z_x - z_y)

    DPC_PSSM_total = find_DPC_PSSM_with_window_total(PSSM_norm_without_window_indi,DPC_PSSM_total,ws);
    return DPC_PSSM_total;

# physicochemical properties : 7 ;

def find_individual_phy_property(phy_property_total, total_input):

    # physicochemical info :
    phy_property = {'A': [-0.39, -0.76, -0.75, -0.19, -0.19, 1, -0.53],
                    'G': [-1, -1, -1, -0.38, -0.2, -1, -1],
                    'V': [0.75, -0.32, -0.26, 0.37, -0.21, -0.03, 1],
                    'L': [0.24, -0.07, -0.01, 0.66, -0.21, 0.79, -0.06],
                    'I': [1, -0.07, -0.01, 0.72, -0.21, 0.17, 0.76],
                    'F': [0.4, 0.41, 0.46, 0.72, -0.3, 0.17, 0.35],
                    'Y': [0.4, 0.46, 0.6, 0.21, -0.3, -0.17, 0.53],
                    'W': [0.53, 1, 1, 1, -0.23, 0.31, 0.59],
                    'T': [0.45, -0.46, -0.36, -0.22, -0.32, -0.45, 0.24],
                    'S': [-0.37, -0.71, -0.6, -0.4, -0.29, -0.52, -0.24],
                    'R': [0.12, 0.41, 0.52, -1, 1, 0.59, -0.41],
                    'K': [-0.1, 0.07, 0.18, -0.99, 0.81, 0.31, -0.29],
                    'H': [0.43, 0.12, 0.15, -0.3, 0.22, -0.03, -0.12],
                    'N': [-0.24, -0.37, -0.27, -0.75, -0.08, -0.45, -0.59],
                    'Q': [-0.26, -0.12, -0.02, -0.52, -0.31, 0.59, -0.41],
                    'D': [-0.24, -0.46, -0.31, -0.85, -1, -0.17, -0.71],
                    'E': [-0.26, -0.27, -0.06, -0.77, -0.96, 1, -0.65],
                    'M': [0.12, 0.07, 0.1, 0.37, -0.29, 0.72, 0],
                    'P': [0.27, -1, -0.33, 0.06, -0.01, -1, 0.12],
                    'C': [-0.16, -0.37, -0.4, 0.56, -0.13, -0.72, 0.53],
                    }

    for i in list(range(0,len(total_input))):
        col_index = 0 ;
        x = total_input[i] ;
        print(x)

        for j in range(0,len(x)):
            col_index = 0 ;
            for k in range(0,7):
                phy_property_total[j,col_index] = phy_property[x[j]][k]  ;
                col_index = col_index + 1 ;


    return phy_property_total ;

def find_individual_embedding_files(file_path):
    D_feature = pd.read_csv(file_path, header=None)
    embedding_total = D_feature.iloc[:, 1:].values

    return embedding_total;


