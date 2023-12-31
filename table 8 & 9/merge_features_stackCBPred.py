import csv

import numpy as np
import pandas as pd
# get all the sequence codes :
def find_all_code_of_sequences(x):
    all_seq_code = []
    f = open('./StackCBPred/'+x+'.txt')
    file = f.readlines()

    # train : input sequence :
    s = file[0]
    s = s[1:6]
    all_seq_code.append(s)
    for i in list(range(3, len(file))):
        if i % 3 == 0:
            s = file[i]
            s = s[1:6]
            all_seq_code.append(s)
    return all_seq_code


with open('./StackCBPred/feature_file_test_ws1_TS49.csv', 'w') as f:
    writer = csv.writer(f)
    all_seq_code = find_all_code_of_sequences('TS49')
    print('all_seq_code : ', all_seq_code)
    print('len of seq_code : ', len(all_seq_code))
    merged = np.zeros((1, 34))
    for i in range(len(all_seq_code)):
        file_path = f'./StackCBPred/test/feature_file_test_ws1_{all_seq_code[i]}.csv'
        c = pd.read_csv(file_path, header=None).values
        merged = np.concatenate((merged, c), axis=0)
    merged = np.delete(merged, 0, axis=0)
    writer.writerows(merged)
print('merged.shape : ', merged.shape)

with open('./StackCBPred/feature_file_test_ws5_TS49.csv', 'w') as f:
    writer = csv.writer(f)
    all_seq_code = find_all_code_of_sequences('TS49')
    print('all_seq_code : ', all_seq_code)
    print('len of seq_code : ', len(all_seq_code))
    merged = np.zeros((1, 166))
    for i in range(len(all_seq_code)):
        file_path = f'./StackCBPred/test/feature_file_test_ws5_{all_seq_code[i]}.csv'
        c = pd.read_csv(file_path, header=None).values
        merged = np.concatenate((merged, c), axis=0)
    merged = np.delete(merged, 0, axis=0)
    writer.writerows(merged)
print('merged.shape : ', merged.shape)

with open('./StackCBPred/feature_file_test_ws1_TS88.csv', 'w') as f:
    writer = csv.writer(f)
    all_seq_code = find_all_code_of_sequences('TS88')
    print('all_seq_code : ', all_seq_code)
    print('len of seq_code : ', len(all_seq_code))
    merged = np.zeros((1, 34))
    for i in range(len(all_seq_code)):
        file_path = f'./StackCBPred/test/feature_file_test_ws1_{all_seq_code[i]}.csv'
        c = pd.read_csv(file_path, header=None).values
        c = c[1:, :]
        merged = np.concatenate((merged, c), axis=0)
    merged = np.delete(merged, 0, axis=0)
    writer.writerows(merged)
print('merged.shape : ', merged.shape)

with open('./StackCBPred/feature_file_test_ws5_TS88.csv', 'w') as f:
    writer = csv.writer(f)
    all_seq_code = find_all_code_of_sequences('TS88')
    print('all_seq_code : ', all_seq_code)
    print('len of seq_code : ', len(all_seq_code))
    merged = np.zeros((1, 166))
    for i in range(len(all_seq_code)):
        file_path = f'./StackCBPred/test/feature_file_test_ws5_{all_seq_code[i]}.csv'
        c = pd.read_csv(file_path, header=None).values
        c = c[1:, :]
        merged = np.concatenate((merged, c), axis=0)
    merged = np.delete(merged, 0, axis=0)
    writer.writerows(merged)
print('merged.shape : ', merged.shape)
