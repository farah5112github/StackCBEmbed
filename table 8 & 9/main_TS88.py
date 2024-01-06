import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# import the classifiers :
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB

import random
import pickle
import csv

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,f1_score,matthews_corrcoef,average_precision_score
from sklearn.metrics import balanced_accuracy_score


def preprocess_the_dataset(feature_X):

    pt = PowerTransformer()
    pt.fit(feature_X)
    feature_X = pt.transform(feature_X)

    return feature_X


def make_string(s):
    str = ''
    for i in s:
        str += i + ", "
    return str[:-2]


def load_the_pickle_files_base_layer(converted_all_features_with_output):
    test_X = preprocess_the_dataset(converted_all_features_with_output)
    test_base_output_total = np.zeros((len(test_X), 1), dtype=float)

    if cmd == '1':
        pickle_folder_path = str('./base_layer_pickle_files_with_PSSM/')
    elif cmd == '2':
        pickle_folder_path = str('./base_layer_pickle_files_without_embedding/')
    else:
        pickle_folder_path = str('./base_layer_pickle_files_without_PSSM/')

    base_classifiers = ['XGB', 'SVM',  'MLP']

    for i in range(0, 10):
        for base_classifier in base_classifiers:
            pickle_file_path = str(pickle_folder_path + base_classifier + '_base_layer_' + str(i) + '.sav')
            # print(pickle_file_path)

            outfile = open(pickle_file_path, 'rb')
            clf = pickle.load(outfile)
            outfile.close()

            y_pred = clf.predict_proba(test_X)
            # print(y_pred.shape);
            test_base_output_total = np.concatenate((test_base_output_total, y_pred), axis=1)

    test_base_output_total = np.delete(test_base_output_total, 0, axis=1)

    return test_base_output_total


def load_the_pickle_files_meta_layer(converted_all_features_with_output):
    test_X = preprocess_the_dataset(converted_all_features_with_output)

    if cmd == '1':
        pickle_file_path = str('./base_layer_pickle_files_with_PSSM/XGB_meta_layer.sav')
    elif cmd == '2':
        pickle_file_path = str('./base_layer_pickle_files_without_embedding/XGB_meta_layer.sav')
    else:
        pickle_file_path = str('./base_layer_pickle_files_without_PSSM/XGB_meta_layer.sav')

    outfile = open(pickle_file_path, 'rb')
    clf = pickle.load(outfile)
    outfile.close()

    y_pred = clf.predict(test_X)
    y_prob = clf.predict_proba(test_X)

    return y_pred, y_prob


def find_metrics(y_predict, y_proba, y_test):

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()  # y_true, y_pred

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    bal_acc = balanced_accuracy_score(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)

    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)

    if prec == 0 and sensitivity == 0:
        f1_score_1 = 0
    else:
        f1_score_1 = 2 * prec * sensitivity / (prec + sensitivity)
    mcc = matthews_corrcoef(y_test, y_predict)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    auPR = average_precision_score(y_test, y_proba[:, 1])  # auPR

    return sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR


feature_paths = {
    'PSSM': './all_required_csvs/TS88_with_PSSM.csv',
    'ProtT5-XL-UniRef50': './all_required_csvs/TS88_embeddings.csv',
}

cmds = ['2', '0', '1']
with open('./output_csvs/metrics_TS88.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Predictor', 'Sensitivity', 'Specificity', 'Balanced_acc', 'Accuracy', 'F1-score', 'MCC', 'AUC', 'auPR'])
    for cmd in cmds:
        if cmd == '1':
            print('StackCBEmbed (Embedding + PSSM)')
            pssm = True
            embedding = True
        elif cmd == '0':
            print('StackCBEmbed (Embedding)')
            pssm = False
            embedding = True
        elif cmd == '2':
            print('StackCBEmbed (PSSM)')
            pssm = True
            embedding = False
        else:
            exit(0)
        file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-UniRef50']
        D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
        feature_y_Benchmark = D_feature.iloc[:, 0].values

        feature_X_Benchmark = np.zeros((feature_y_Benchmark.shape[0], 1), dtype=float)

        if embedding:
            file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-UniRef50']
            D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
            feature_X_Benchmark = np.concatenate((feature_X_Benchmark, D_feature.iloc[:, 1:].values), axis=1)

        if pssm:
            file_path_Benchmark_embeddings = feature_paths['PSSM']
            D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
            feature_X_Benchmark = np.concatenate((feature_X_Benchmark, D_feature.iloc[1:, 2:].values), axis=1)

        feature_X_Benchmark = np.delete(feature_X_Benchmark, 0, axis=1)

        X = feature_X_Benchmark.copy()
        X = preprocess_the_dataset(X)
        y = feature_y_Benchmark.copy()

        print('X : ', X.shape)
        print('y : ', y.shape)

        BLP = load_the_pickle_files_base_layer(X)
        X = np.concatenate((X, BLP), axis=1)

        print('X : ', X.shape)
        print('y : ', y.shape)

        y_pred, y_proba = load_the_pickle_files_meta_layer(X)

        sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(y_pred, y_proba, y)

        print('Sensitivity : {0:.4f}'.format(sensitivity))
        print('Specificity : {0:.4f}'.format(specificity))
        print('Balanced_acc : {0:.4f}'.format(bal_acc))
        print('Accuracy : {0:.4f}'.format(acc))
        print('F1-score: {0:.4f}'.format(f1_score_1))
        print('MCC: {0:.4f}'.format(mcc))
        print('auROC: {0:.4f}'.format(auc))
        print('auPR: {0:.4f}'.format(auPR))

        if cmd == '0':
            writer.writerow(['StackCBEmbed (Embedding)', '{0:.4f}'.format(sensitivity), '{0:.4f}'.format(specificity), '{0:.4f}'.format(bal_acc), '{0:.4f}'.format(acc), '{0:.4f}'.format(f1_score_1), '{0:.4f}'.format(mcc), '{0:.4f}'.format(auc), '{0:.4f}'.format(auPR)])
        elif cmd == '1':
            writer.writerow(['StackCBEmbed', '{0:.4f}'.format(sensitivity), '{0:.4f}'.format(specificity), '{0:.4f}'.format(bal_acc), '{0:.4f}'.format(acc), '{0:.4f}'.format(f1_score_1), '{0:.4f}'.format(mcc), '{0:.4f}'.format(auc), '{0:.4f}'.format(auPR)])
        else:
            writer.writerow(['StackCBEmbed (PSSM)', '{0:.4f}'.format(sensitivity), '{0:.4f}'.format(specificity), '{0:.4f}'.format(bal_acc), '{0:.4f}'.format(acc), '{0:.4f}'.format(f1_score_1), '{0:.4f}'.format(mcc), '{0:.4f}'.format(auc), '{0:.4f}'.format(auPR)])
