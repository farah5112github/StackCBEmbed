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

from sklearn.model_selection import train_test_split, GridSearchCV

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


def find_metrics(model_name, y_test, n_components=2, scale=True, max_iter=500, tol=1e-06):
    if model_name == 'PLS':
        model = PLSRegression(n_components=n_components, scale=scale, max_iter=max_iter, tol=tol)
    else:
        print('Wrong model name')
        return

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


feature_paths = {
    'PSSM': './all_required_csvs/Benchmark_with_PSSM.csv',
    'MonoGram': './all_required_csvs/Benchmark_with_monogram.csv',
    'DPC': './all_required_csvs/Benchmark_with_DPC.csv',
    'ASA': './all_required_csvs/Benchmark_with_ASA.csv',
    'HSE': './all_required_csvs/Benchmark_with_HSE.csv',
    'torsion_angles': './all_required_csvs/Benchmark_with_torsion_angles.csv',
    'Physicochemical': './all_required_csvs/Benchmark_with_physicochemical.csv',
    'ProtT5-XL-UniRef50': './all_required_csvs/benchmark_embeddings.csv',
}


file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-UniRef50']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
feature_y_Benchmark_embeddings = D_feature.iloc[:, 0].values

feature_X_Benchmark_embeddings = np.zeros((feature_y_Benchmark_embeddings.shape[0], 1), dtype=float)

file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-UniRef50']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 0:].values), axis=1)

file_path_Benchmark_embeddings = feature_paths['PSSM']
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

X = feature_X_Benchmark_embeddings_train.copy()
y = feature_y_Benchmark_embeddings_train.copy()

X = preprocess_the_dataset(X)

# balance the dataset :
rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(X, y)

c = Counter(y)
print(c)

#n_components=2, scale=True, max_iter=500, tol=1e-06

grid = {
    'n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'scale': [True, False],
    'max_iter': [500, 600],
    'tol': [1e-6, 1e-7],
}

with open("./output_csvs/PLS_grid_search.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["n_components, scale, max_iter, tol", "Sensitivity", "Specificity", "Balanced_acc", "Accuracy", "Precision", "F1-score", "MCC", "AUC", "auPR"])
    for n_components in grid['n_components']:
        for scale in grid['scale']:
            for max_itr in grid['max_iter']:
                for tol in grid['tol']:
                    random.seed(1)

                    # Step 06 : Spliting with 10-FCV :
                    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

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

                        sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics('PLS', y_test, n_components=n_components, scale=scale, max_iter=max_itr, tol=tol)

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
                        print(
                            '___________________________________________________________________________________________________________')

                    print(n_components, scale, max_itr, tol)
                    print('Sensitivity : {0:.3f}'.format(np.mean(local_Sensitivity)))
                    print('Specificity : {0:.3f}'.format(np.mean(local_Specificity)))
                    print('Balanced_acc : {0:.3f}'.format(np.mean(local_Balanced_acc)))
                    print('Accuracy : {0:.3f}'.format(np.mean(local_Accuracy)))
                    print('Precision : {0:.3f}'.format(np.mean(local_Precision)))
                    print('F1-score: {0:.3f}'.format(np.mean(local_F1)))
                    print('MCC: {0:.3f}'.format(np.mean(local_MCC)))
                    print('AUC: {0:.3f}'.format(np.mean(local_AUC)))
                    print('auPR: {0:.3f}'.format(np.mean(local_AUPR)))

                    writer.writerow([f"'n_components':{n_components}, 'scale':{scale}, 'max_itr':{max_itr},'tol': {tol}", '{0:.3f}'.format(np.mean(local_Sensitivity)),
                                     '{0:.3f}'.format(np.mean(local_Specificity)),
                                     '{0:.3f}'.format(np.mean(local_Balanced_acc)),
                                     '{0:.3f}'.format(np.mean(local_Accuracy)),
                                     '{0:.3f}'.format(np.mean(local_Precision)), '{0:.3f}'.format(np.mean(local_F1)),
                                     '{0:.3f}'.format(np.mean(local_MCC)), '{0:.3f}'.format(np.mean(local_AUC)),
                                     '{0:.3f}'.format(np.mean(local_AUPR))])
                    print('___________________________________________________________________________________________________________')
