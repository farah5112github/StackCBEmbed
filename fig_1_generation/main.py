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
import matplotlib.pyplot as plt


def preprocess_the_dataset(feature_X):

    pt = PowerTransformer()
    pt.fit(feature_X)
    feature_X = pt.transform(feature_X)

    return feature_X


def model_fit(model_name, X_train, y_train):
    if model_name == 'RF':
        model = RandomForestClassifier(random_state=1)
    elif model_name == 'ET':
        model = ExtraTreesClassifier(random_state=1)
    elif model_name == 'DT':
        model = DecisionTreeClassifier(random_state=1)
    elif model_name == 'MLP':
        model = MLPClassifier(random_state=1, max_iter=4000)
    elif model_name == 'LR':
        model = LogisticRegression(class_weight='balanced', random_state=1, max_iter=1000)
    elif model_name == 'SVM':
        model = SVC(kernel='rbf', random_state=1, probability=True)
    elif model_name == 'NB':
        model = GaussianNB()
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
    elif model_name == 'XGB':
        model = XGBClassifier(random_state=1)
    elif model_name == 'PLS':
        model = PLSRegression(n_components=2)
    else:
        print('Wrong model name')
        return

    try:
        model.fit(X_train, y_train)
    except:
        model = PLSRegression(n_components=1)
        model.fit(X_train, y_train)

    return model


def make_string(s):
    str = ''
    for i in s:
        str += i + ", "
    return str[:-2]


def create_subsample(X, y, percentage):
    X = np.concatenate((y.reshape(-1, 1), X), axis=1)

    feature_X_positive = X[X[:, 0] == 1, 1:]
    feature_y_positive = X[X[:, 0] == 1, 0].astype('int')

    feature_X_negative = X[X[:, 0] == 0, 1:]
    feature_y_negative = X[X[:, 0] == 0, 0].astype('int')

    # print(feature_X_positive.shape)
    # print(feature_y_positive.shape)
    #
    # print(feature_X_negative.shape)
    # print(feature_y_negative.shape)

    feature_X_positive_test = feature_X_positive
    feature_y_positive_test = feature_y_positive

    feature_X_negative_train, feature_X_negative_test, feature_y_negative_train, feature_y_negative_test = train_test_split(
        feature_X_negative, feature_y_negative, test_size=percentage, random_state=1)

    # print(feature_X_positive_train.shape)
    # print(feature_X_positive_test.shape)
    #
    # print(feature_X_negative_train.shape)
    # print(feature_X_negative_test.shape)

    feature_X_test = np.concatenate((feature_X_positive_test, feature_X_negative_test), axis=0)
    feature_y_test = np.concatenate((feature_y_positive_test, feature_y_negative_test), axis=0)

    print("subsample shape:",)
    print(feature_X_test.shape)
    print(feature_y_test.shape)

    return feature_X_test, feature_y_test


def create_base_layer(X, y):
    X = preprocess_the_dataset(X)
    percentages = [0.002, 0.002, 0.002, 0.002, 0.02, 0.02, 0.02, 0.2, 0.2, 0.2]
    model_names = ['SVM', 'MLP', 'ET', 'PLS']

    i = 0
    for percentage in percentages:
        trainX, trainy = create_subsample(X, y, percentage)
        for model_name in model_names:
            if model_name == 'ET':
                model = ExtraTreesClassifier(random_state=1, bootstrap=False, criterion='gini', max_depth=20,
                                             max_features='sqrt', n_estimators=200)
            elif model_name == 'MLP':
                model = MLPClassifier(random_state=1, max_iter=4000, activation='relu', alpha=0.05,
                                      hidden_layer_sizes=(50, 50, 50), learning_rate='constant', solver='adam')
            elif model_name == 'SVM':
                model = SVC(kernel='rbf', random_state=1, probability=True, C=10, degree=3, gamma='auto')
            elif model_name == 'PLS':
                model = PLSRegression(n_components=2)
            else:
                exit(0)
            model = model.fit(trainX, trainy)

            filename = f'./base_layer_pickle_files_with_PSSM/{model_name}_base_layer_{i}.sav'
            with open(filename, 'wb') as f:
                pickle.dump(model, f)

        i += 1


def load_model_and_get_BLP(X):
    X = preprocess_the_dataset(X)
    model_names = ['SVM', 'MLP', 'ET', 'PLS']

    prob = np.zeros((X.shape[0], 1), dtype=float)
    for itr in range(10):
        for model_name in model_names:
            filename = f'./base_layer_pickle_files_with_PSSM/{model_name}_base_layer_{itr}.sav'
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                if model_name == 'PLS':
                    p_all = []
                    p_all.append([1 - np.abs(item[0]) for item in model.predict(X)])
                    p_all.append([np.abs(item[0]) for item in model.predict(X)])
                    y_proba = np.transpose(np.array(p_all))[:, 1].reshape(-1, 1)
                else:
                    y_proba = model.predict_proba(X)[:, 1].reshape(-1, 1)
                prob = np.concatenate((prob, y_proba), axis=1)
    prob = np.delete(prob, 0, axis=1)

    return prob


def find_metrics(model_name, y_test):
    if model_name == 'RF':
        model = RandomForestClassifier(random_state=1)
    elif model_name == 'ET':
        model = ExtraTreesClassifier(random_state=1)
    elif model_name == 'DT':
        model = DecisionTreeClassifier(random_state=1)
    elif model_name == 'MLP':
        model = MLPClassifier(random_state=1, max_iter=4000)
    elif model_name == 'LR':
        model = LogisticRegression(class_weight='balanced', random_state=1, max_iter=1000)
    elif model_name == 'SVM':
        model = SVC(kernel='rbf', random_state=1, probability=True)
    elif model_name == 'NB':
        model = GaussianNB()
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
    elif model_name == 'XGB':
        model = XGBClassifier(random_state=1)
    elif model_name == 'PLS':
        model = PLSRegression(n_components=2)
    else:
        print('Wrong model name')
        return

    try:
        model.fit(X_train, y_train)
    except:
        model = PLSRegression(n_components=1)
        model.fit(X_train, y_train)

    if model_name == 'PLS':
        y_predict = []
        for item in model.predict(X_test):
            if (item < 1.5):
                y_predict.append(np.round(np.abs(item[0])))
            else:
                y_predict.append(1)

        p_all = []
        p_all.append([1 - np.abs(item[0]) for item in model.predict(X_test)])
        p_all.append([np.abs(item[0]) for item in model.predict(X_test)])
        y_proba = np.transpose(np.array(p_all))
    else:
        y_predict = model.predict(X_test)  # predicted labels
        y_proba = model.predict_proba(X_test)

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


def plot_bar_graph(f1_scores, f2_scores, f3_scores, f4_scores, f5_scores, f6_scores):
    bar_width = 0.1
    r1 = np.arange(len(f1_scores))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]

    xticks = [x + bar_width/2 for x in r3]

    plt.bar(r1, f1_scores, color='magenta', width=bar_width, label='FC-1', edgecolor='gray')
    plt.bar(r2, f2_scores, color='lightgray', width=bar_width, label='FC-2', edgecolor='gray')
    plt.bar(r3, f3_scores, color='pink', width=bar_width, label='FC-3', edgecolor='gray')
    plt.bar(r4, f4_scores, color='cyan', width=bar_width, label='FC-4', edgecolor='gray')
    plt.bar(r5, f5_scores, color='thistle', width=bar_width, label='FC-5', edgecolor='gray')
    plt.bar(r6, f6_scores, color='orange', width=bar_width, label='FC-6', edgecolor='gray')
    plt.xticks(xticks, ('SN', 'SP', 'BACC', 'ACC', 'Prec', 'F1', 'MCC', 'AUC', 'auPR'))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6)
    plt.ylim(0.5, 0.9)
    plt.show()


feature_paths = {
    'PSSM': './all_required_csvs/Benchmark_with_PSSM.csv',
    'MonoGram': './all_required_csvs/Benchmark_with_monogram.csv',
    'DPC': './all_required_csvs/Benchmark_with_DPC.csv',
    'ASA': './all_required_csvs/Benchmark_with_ASA.csv',
    'torsion_angles': './all_required_csvs/Benchmark_with_torsion_angles.csv',
    'Physicochemical': './all_required_csvs/Benchmark_with_physicochemical.csv',
    'ProtT5-XL-UniRef50': './all_required_csvs/benchmark_embeddings.csv',
}

feature_combination = [['ProtT5-XL-UniRef50', 'PSSM'],
                       ['ProtT5-XL-UniRef50', 'PSSM', 'torsion_angles', 'MonoGram', 'DPC', 'ASA'],
                       ['ProtT5-XL-UniRef50', 'PSSM', 'torsion_angles', 'MonoGram', 'DPC'],
                       ['ProtT5-XL-UniRef50', 'PSSM', 'torsion_angles'],
                       ['ProtT5-XL-UniRef50', 'PSSM', 'torsion_angles', 'MonoGram', 'DPC', 'Physicochemical'],
                       ['ProtT5-XL-UniRef50', 'PSSM', 'BLP']]

pssm = True

file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-UniRef50']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
feature_y_Benchmark_embeddings = D_feature.iloc[:, 0].values

BLP = False

all_metrics = []
for features in feature_combination:
    print(make_string(features))
    feature_X_Benchmark_embeddings = np.zeros((feature_y_Benchmark_embeddings.shape[0], 1), dtype=float)

    for feature in features:
        if feature == 'ProtT5-XL-UniRef50':
            file_path_Benchmark_embeddings = feature_paths[feature]
            D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
            feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 0:].values), axis=1)
        elif feature == 'BLP':
            BLP = True
        else:
            file_path_Benchmark_embeddings = feature_paths[feature]
            D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
            feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[1:, 2:].values), axis=1)

    feature_X_Benchmark_embeddings = np.delete(feature_X_Benchmark_embeddings, 0, axis=1)

    feature_X_Benchmark_embeddings_positive = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 1, 1:]
    feature_y_Benchmark_embeddings_positive = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 1, 0].astype('int')

    feature_X_Benchmark_embeddings_negative = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 0, 1:]
    feature_y_Benchmark_embeddings_negative = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 0, 0].astype('int')
    # print(feature_X_Benchmark_embeddings_positive.shape)
    # print(feature_y_Benchmark_embeddings_positive.shape)
    #
    # print(feature_X_Benchmark_embeddings_negative.shape)
    # print(feature_y_Benchmark_embeddings_negative.shape)
    feature_X_Benchmark_embeddings_positive_train, feature_X_Benchmark_embeddings_positive_test, feature_y_Benchmark_embeddings_positive_train, feature_y_Benchmark_embeddings_positive_test = train_test_split(feature_X_Benchmark_embeddings_positive, feature_y_Benchmark_embeddings_positive, test_size=275, random_state=1)
    feature_X_Benchmark_embeddings_negative_train, feature_X_Benchmark_embeddings_negative_test, feature_y_Benchmark_embeddings_negative_train, feature_y_Benchmark_embeddings_negative_test = train_test_split(feature_X_Benchmark_embeddings_negative, feature_y_Benchmark_embeddings_negative, test_size=7741, random_state=1)
    # print(feature_X_Benchmark_embeddings_positive_train.shape)
    # print(feature_X_Benchmark_embeddings_positive_test.shape)
    #
    # print(feature_X_Benchmark_embeddings_negative_train.shape)
    # print(feature_X_Benchmark_embeddings_negative_test.shape)
    feature_X_Benchmark_embeddings_train = np.concatenate((feature_X_Benchmark_embeddings_positive_train, feature_X_Benchmark_embeddings_negative_train), axis=0)
    feature_y_Benchmark_embeddings_train = np.concatenate((feature_y_Benchmark_embeddings_positive_train, feature_y_Benchmark_embeddings_negative_train), axis=0)
    feature_X_Benchmark_embeddings_test = np.concatenate((feature_X_Benchmark_embeddings_positive_test, feature_X_Benchmark_embeddings_negative_test), axis=0)
    feature_y_Benchmark_embeddings_test = np.concatenate((feature_y_Benchmark_embeddings_positive_test, feature_y_Benchmark_embeddings_negative_test), axis=0)

    # print(feature_X_Benchmark_embeddings_train.shape)
    # print(feature_y_Benchmark_embeddings_train.shape)
    print(feature_X_Benchmark_embeddings_test.shape)
    print(feature_y_Benchmark_embeddings_test.shape)

    if BLP:
        X = feature_X_Benchmark_embeddings_train.copy()
        y = feature_y_Benchmark_embeddings_train.copy()
        create_base_layer(X, y)

        X = feature_X_Benchmark_embeddings_test.copy()
        y = feature_y_Benchmark_embeddings_test.copy()
        X = np.concatenate((X, load_model_and_get_BLP(X)), axis=1)
    else:
        X = feature_X_Benchmark_embeddings_test.copy()
        y = feature_y_Benchmark_embeddings_test.copy()

    X = preprocess_the_dataset(X)

    # Step 06 : Spliting with 10-FCV :
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    # balance the dataset :
    rus = RandomUnderSampler(random_state=1)
    X, y = rus.fit_resample(X, y)

    c = Counter(y)
    print(c)

    local_Sensitivity = []
    local_Specificity = []
    local_Balanced_acc = []
    local_Accuracy = []
    local_Precision = []
    local_AUPR = []
    local_F1 = []
    local_MCC = []
    local_AUC = []

    i = 1
    for train_index, test_index in cv.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics('SVM', y_test)

        local_Sensitivity.append(sensitivity)
        local_Specificity.append(specificity)
        local_Balanced_acc.append(bal_acc)
        local_Accuracy.append(acc)
        local_Precision.append(prec)
        local_F1.append(f1_score_1)
        local_MCC.append(mcc)
        local_AUC.append(auc)
        local_AUPR.append(auPR)

        print(i, 'th iteration done')
        i = i + 1
        print('___________________________________________________________________________________________________________')

    print('Combination : ', make_string(features))
    print('Sensitivity : {0:.4f}'.format(np.mean(local_Sensitivity)))
    print('Specificity : {0:.4f}'.format(np.mean(local_Specificity)))
    print('Balanced_acc : {0:.4f}'.format(np.mean(local_Balanced_acc)))
    print('Accuracy : {0:.4f}'.format(np.mean(local_Accuracy)))
    print('Precision : {0:.4f}'.format(np.mean(local_Precision)))
    print('F1-score: {0:.4f}'.format(np.mean(local_F1)))
    print('MCC: {0:.4f}'.format(np.mean(local_MCC)))
    print('auROC: {0:.4f}'.format(np.mean(local_AUC)))
    print('auPR: {0:.4f}'.format(np.mean(local_AUPR)))
    print('___________________________________________________________________________________________________________')

    all_metrics.append([np.mean(local_Sensitivity), np.mean(local_Specificity), np.mean(local_Balanced_acc), np.mean(local_Accuracy), np.mean(local_Precision), np.mean(local_F1), np.mean(local_MCC), np.mean(local_AUC), np.mean(local_AUPR)])

plot_bar_graph(all_metrics[0], all_metrics[1], all_metrics[2], all_metrics[3], all_metrics[4], all_metrics[5])


