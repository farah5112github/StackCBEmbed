import pickle;
import skops.io as sio
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


def load_the_pickle_files_base_layer(converted_all_features_with_output, folder_path, choice):

    test_X = converted_all_features_with_output[:,1:] ;
    test_base_output_total = np.zeros((len(test_X),1), dtype=float);

    #print('test X : ',test_X.shape);
    if(choice == 1):
        pickle_folder_path = str( folder_path + 'base_layer_pickle_files_with_PSSM/' ) ;
    else:
        pickle_folder_path = str(folder_path + 'base_layer_pickle_files_without_PSSM/');

    base_classifiers = ['SVM','ET','XGB'] ;


    for i in range(0,10):

        for base_classifier in base_classifiers:
            pickle_file_path = str(pickle_folder_path + base_classifier + '_base_layer_' + str(i) + '.sav' );
            #print(pickle_file_path);

            outfile = open(pickle_file_path, 'rb');
            clf = pickle.load(outfile);
            outfile.close();

            y_pred = clf.predict_proba(test_X);
            #print(y_pred.shape);
            test_base_output_total = np.concatenate(  (test_base_output_total,y_pred), axis=1);

    test_base_output_total = np.delete( test_base_output_total, 1, axis=1 );
    #print('test_base_output_total : ', test_base_output_total.shape);

    return test_base_output_total;

def load_the_pickle_files_meta_layer(converted_all_features_with_output, folder_path,choice):
    test_X = converted_all_features_with_output[:, 1:];
    test_y = converted_all_features_with_output[:, 0];

    #pickle_file_path = str(folder_path + 'necessary_pickle_files/XGB_meta_layer.sav');

    if (choice == 1):
        pickle_file_path = str(folder_path + 'base_layer_pickle_files_with_PSSM/XGB_meta_layer.sav');
    else:
        pickle_file_path = str(folder_path + 'base_layer_pickle_files_without_PSSM/XGB_meta_layer.sav');

    outfile = open(pickle_file_path, 'rb');
    clf = pickle.load(outfile);
    outfile.close();

    y_pred = clf.predict(test_X);

    return test_y, y_pred ;