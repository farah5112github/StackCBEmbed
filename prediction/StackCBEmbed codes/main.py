# Essential libraries :
import os;

from sequence_to_list_conversion import *
from get_all_the_feature_vectors import *
from base_meta_layer_processing import *
from read_write_operation import *


def main():
    total_input, total_output, all_seq_code, choice = sequence_processing()
    for i in range(0,len(all_seq_code)):

        # choice = 0-> no-pssm
        # get the feature files :
        embedding_file_path = str('../input_files/' + all_seq_code[i] + '.csv')

        # call the functions :
        embedding_total = find_individual_embedding_files(embedding_file_path)
        X_3 = embedding_total.copy()

        # print :
        print('embedding feature vector created successfully...')

        if choice == 1:
            # get the feature files :
            PSSM_file_path = str('../input_files/' + all_seq_code[i] + '.pssm')
            #print(PSSM_file_path)

            # call the functions :
            PSSM_total = find_individual_PSSM_files(PSSM_file_path)

            print('PSSM feature vector created successfully...')

            X_1 = PSSM_total.copy()
            converted_all_features_with_output = np.concatenate((X_1, X_3), axis=1)
            # print('merged features with output [model with PSSM] : ', converted_all_features_with_output.shape)

        else:
            converted_all_features_with_output =  X_3
            # print('merged features with output [model without PSSM] : ', converted_all_features_with_output.shape)
        
        # load the pickle files:
        test_base_layer_total = load_the_pickle_files_base_layer(converted_all_features_with_output,choice)
        test_meta_layer_with_output_total = np.concatenate((converted_all_features_with_output, test_base_layer_total), axis=1)
        # print('test_meta_layer_with_output_total : ', test_meta_layer_with_output_total.shape)

        y_pred = load_the_pickle_files_meta_layer(test_meta_layer_with_output_total, choice)
        y_test = find_class_label(total_output[i])
        output = np.array([['y_actual', 'y_pred']])
        y = np.concatenate((y_test, y_pred.reshape(len(y_pred), 1)), axis=1)

        output = np.concatenate((output, y), axis=0)

        write_into_folder(output, 'output_' + str(all_seq_code[i]))

        print('________________________________________________________________________________________')


if __name__ == "__main__":
    main()
