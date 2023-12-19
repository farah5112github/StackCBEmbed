# Essential libraries :
import pandas as pd
import numpy as np
import math
import os

def seperate_the_data_sets_into_lists(file_path,total_input):
    f = open(file_path)
    file = f.readlines()
    l = len(file)

    s = file[1]
    s = s[0:-1]
    total_input.append(s)

    for i in list(range(3,l,2)):
        s = file[i]
        s = s[0:-1]
        total_input.append(s)

    f.close();

    return total_input;

def find_all_code_of_sequences(file_path,all_seq_code):
    f = open(file_path);
    file = f.readlines()
    l = len(file)
    #print(l)

    s = file[0]
    s = s[1:6]
    all_seq_code.append(s)
    for i in list(range(2, l, 2)):
        s = file[i]
        s = s[1:6]
        all_seq_code.append(s)

    return all_seq_code ;

def sequence_processing(total_input, all_seq_code, folder_path):

    # step - 1 : get the folder path :
    os.chdir('..');
    folder_path = os.getcwd();
    #print(folder_path);

    fasta_file_path = folder_path + '\\input_files\\' + 'input.txt';
    #print(fasta_file_path);

    print('please enter a choice: 0 or 1. 0= model (without PSSM) and 1= model (with PSSM)');
    choice = int(input());

    # step - 2 : seperate_the_data_sets_into_lists() function is used to sepearate the dataset into input and output and put them into lists named "total_input" and "total_output" :
    #total_input, total_output = seperate_the_data_sets_into_lists(fasta_file_path, total_input, total_output);
    total_input = seperate_the_data_sets_into_lists(fasta_file_path, total_input);

    # step - 3 : show the total_input and total_output into list :
    #print('total_input : ', total_input);
    #print('total_output : ', total_output);

    # step - 4 : get all the sequence codes into list :
    all_seq_code = [];
    all_seq_code = find_all_code_of_sequences(fasta_file_path, all_seq_code);

    # step - 5 : show the sequence code :
    #print('all_seq_code : ', all_seq_code);
    #print('len of seq_code : ', len(all_seq_code));

    return total_input, all_seq_code, folder_path, choice;