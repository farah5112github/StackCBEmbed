# Essential libraries :
import os;

from sequence_to_list_conversion import *
from get_all_the_feature_vectors import *
from base_meta_layer_processing import *
from read_write_operation import *

def main():

    total_input = [] ;
    total_output = [] ;
    all_seq_code = [];
    folder_path = '';

    total_input, total_output, all_seq_code, folder_path, choice = sequence_processing(total_input, total_output, all_seq_code, folder_path);

    for i in range(0,len(all_seq_code)):
        feature_vector_total_output = find_class_label(total_output[i]);

        # choice = 0-> no-pssm
        # get the feature files :
        embedding_file_path = str(folder_path + '\\input_files\\' + all_seq_code[i] + '.csv');

        # call the functions :
        embedding_total = find_individual_embedding_files(embedding_file_path);
        X_3 = embedding_total.copy();

        # print the shapes of the feature vectors :
        #print('embedding shape : ', embedding_total.shape);

        # print :
        print('embedding feature vector created successfully...');


        if (choice == 1):
            # initialize the feature vectors:
            ws = 1;
            PSSM_total = np.zeros((1, ws * 20), dtype=float);

            # get the feature files :
            PSSM_file_path = str(folder_path + '\\input_files\\' + all_seq_code[i] + '.pssm');
            #print(PSSM_file_path);

            # call the functions :
            PSSM_total = find_individual_PSSM_files(PSSM_file_path, PSSM_total);

            # necessary calculations :
            PSSM_total = np.delete(PSSM_total, 0, 0);
            # print the shapes of the feature vectors :
            #print('PSSM shape : ', PSSM_total.shape);
            # print :
            print('PSSM feature vector created successfully...');

            X_1 = PSSM_total.copy();
            converted_all_features_with_output = np.concatenate((feature_vector_total_output, X_1, X_3), axis=1);
            #print('merged features with output [model with PSSM] : ', converted_all_features_with_output.shape);


        else:
            converted_all_features_with_output = np.concatenate((feature_vector_total_output, X_3), axis=1);
            #print('merged features with output [model without PSSM] : ', converted_all_features_with_output.shape);

        # load the pickle files:
        test_base_layer_total = load_the_pickle_files_base_layer(converted_all_features_with_output, folder_path,choice);
        test_meta_layer_with_output_total = np.concatenate((converted_all_features_with_output, test_base_layer_total),axis=1);
        #print('test_meta_layer_with_output_total : ', test_meta_layer_with_output_total.shape);

        y_test, y_pred = load_the_pickle_files_meta_layer(test_meta_layer_with_output_total, folder_path, choice);

        output = np.array(['y_actual', 'y_pred']);
        output = output.reshape(1, len(output));
        y = np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), axis=1);
        output = np.concatenate((output, y), axis=0);

        write_into_folder(output, folder_path, '\\output_' + str(all_seq_code[i]) );

        print('________________________________________________________________________________________');

if __name__ == "__main__":
    main();
