# Essential libraries :
from sequence_to_list_conversion import *
from get_all_the_feature_vectors import *
from base_meta_layer_processing import *
from read_write_operation import *

def main():

    total_input = [] ;
    total_output = [] ;
    all_seq_code = [];
    folder_path = '';
    total_input, total_output, all_seq_code, folder_path = sequence_processing(total_input, total_output, all_seq_code, folder_path);

    # initialize the feature vectors and window size :
    ws = 1 ;
    PSSM_total = np.zeros((1, ws * 20), dtype=float) ;
    PSSM_monogram_total = np.zeros((1, ws * 1), dtype=float);
    ASA_total = np.zeros((1, 1), dtype=object);
    torsion_angles_total = np.zeros((1, ws * 2), dtype=float);
    HSE_total = np.zeros((1, ws * 2), dtype=float);
    DPC_PSSM_total = np.zeros((1, ws * 400), dtype=float);
    total_residues_count = 0;
    for i in range(0, len(total_input)):
        x = total_input[i];
        total_residues_count += len(x);
    phy_property_total = np.zeros((total_residues_count, ws * 7), dtype=float);

    # get the feature files :
    PSSM_file_path = str(folder_path + 'input_files/' + all_seq_code[i] + '.pssm');
    ASA_file_path = str(folder_path + 'input_files/' + all_seq_code[i] + '.spXout');
    file_path = str(folder_path + 'input_files/' + all_seq_code[i] + '.spd33');
    embedding_file_path = str(folder_path + 'input_files/' + all_seq_code[i] + '.csv');

    # call the fucntions :
    PSSM_total = find_individual_PSSM_files(PSSM_file_path, PSSM_total);
    PSSM_monogram_total = find_individual_MG_files(PSSM_file_path, PSSM_monogram_total, total_input);
    ASA_total = find_individual_ASA_files(ASA_file_path, ASA_total, total_input);
    torsion_angles_total, HSE_total = find_spider_info_SS(file_path, torsion_angles_total, HSE_total);
    DPC_PSSM_total = find_individual_DPC_files(PSSM_file_path, DPC_PSSM_total, ws);
    phy_property_total = find_individual_phy_property(phy_property_total,total_input);
    embedding_total = find_individual_embedding_files(embedding_file_path);


    # necessary calculations :
    PSSM_total = np.delete(PSSM_total, 0, 0);
    PSSM_monogram_total = np.delete(PSSM_monogram_total, 0, 0);
    ASA_total = np.delete(ASA_total, 0, 0);
    torsion_angles_total = np.delete(torsion_angles_total, 0, 0);
    HSE_total = np.delete(HSE_total, 0, 0);
    DPC_PSSM_total = np.delete(DPC_PSSM_total, 0, 0);

    # print the shapes of the feature vectors :
    print('PSSM shape : ', PSSM_total.shape);
    print('Monogram shape : ', PSSM_monogram_total.shape);
    print('ASA shape : ', ASA_total.shape);
    print('torsion_angles shape : ', torsion_angles_total.shape);
    print('HSE shape : ', HSE_total.shape);
    print('DPC shape : ', DPC_PSSM_total.shape);
    print('phy_property shape : ', phy_property_total.shape);
    print('embedding shape : ', embedding_total.shape)

    # print :
    print('PSSM feature vector created successfully...') ;
    print('MG feature vector created successfully...') ;
    print('ASA feature vector created successfully...');
    print('torsion_angles feature vector created successfully...');
    print('HSE feature vector created successfully...');
    print('DPC feature vector created successfully...');
    print('phy_property feature vector created successfully...');
    print('embedding feature vector created successfully...');

    # merge all the feature vectors :
    all_features_except_DPC_embedding = np.concatenate((PSSM_total,PSSM_monogram_total,ASA_total,torsion_angles_total,HSE_total,phy_property_total),axis = 1);

    # create window_size = 5 for all features :
    ws = 5 ;
    feature_vector_total_output = find_class_label(total_output) ;
    X_1 = convert_feature_vector_to_window_size(all_features_except_DPC_embedding, feature_vector_total_output, ws);
    X_2 = find_individual_DPC_files(PSSM_file_path, DPC_PSSM_total, ws);
    X_3 = embedding_total.copy() ;

    converted_all_features_with_output = np.concatenate((X_1,X_2,X_3),axis=1);
    print('merged features with output : ', converted_all_features_with_output.shape);

    test_base_layer_total = load_the_pickle_files_base_layer(converted_all_features_with_output, folder_path) ;
    test_meta_layer_with_output_total = np.concatenate((converted_all_features_with_output,test_base_layer_total),axis=1) ;
    print('test_meta_layer_with_output_total : ', test_meta_layer_with_output_total.shape) ;

    y_test, y_pred = load_the_pickle_files_meta_layer(test_meta_layer_with_output_total, folder_path) ;

    output = np.array(['y_actual','y_pred']);
    output = output.reshape(1,len(output));
    y = np.concatenate(   (  y_test.reshape(len(y_test),1) , y_pred.reshape(len(y_pred),1)  ) , axis=1 );
    output = np.concatenate( (output, y), axis=0 );

    write_into_folder(output, folder_path, 'output');

if __name__ == "__main__":
    main();
