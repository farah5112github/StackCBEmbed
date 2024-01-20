import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,f1_score,matthews_corrcoef,average_precision_score

# import the classifiers :
import math
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import TweedieRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.under_sampling import RandomUnderSampler
from random import sample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from collections import Counter
from sklearn.metrics import balanced_accuracy_score

from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler

import random
import pickle
import csv

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.model_selection import StratifiedKFold


def preprocess_the_dataset(feature_X):

    pt = PowerTransformer()
    pt.fit(feature_X)
    feature_X = pt.transform(feature_X)

    return feature_X


def find_metrics(model_name, y_test):
    if model_name == 'RF':
        model = RandomForestClassifier(random_state=1)
    elif model_name == 'ET':
        model = ExtraTreesClassifier(random_state=1)
    elif model_name == 'DT':
        model = DecisionTreeClassifier(random_state=1)
    elif model_name == 'MLP':
        model = MLPClassifier(random_state=1)
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
        if model_name == 'PLS':
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
    prec = tp / (tp + fp)
    f1_score_1 = 2 * prec * sensitivity / (prec + sensitivity)
    mcc = matthews_corrcoef(y_test, y_predict)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    auPR = average_precision_score(y_test, y_proba[:, 1])  # auPR

    return sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR


embeddings = {
    'ProtT5-XL-UniRef50': './all_required_csvs/benchmark_embeddings.csv',
    'Prot-BERT': './all_required_csvs/Benchmark_Prot_BERT_embeddings.csv',
    'ProtBERT-BFD': './all_required_csvs/Benchmark_ProtBERT_BFD_embeddings.csv',
    'ProtT5-XL-BFD': './all_required_csvs/Benchmark_ProtT5_XL_BFD_embeddings.csv',
    'ProtT5-XL-Net': './all_required_csvs/Benchmark_Prot_XL_Net_embeddings.csv',
    'Prot-Albert': './all_required_csvs/Benchmark_Prot_Albert_embeddings.csv',
}

with open('./output_csvs/language_model_wise_benchmark_results.csv', 'w') as f1:
    language_model_wise_benchmark_results_csv = csv.writer(f1)

    language_model_wise_benchmark_results_csv.writerow(['Language Model', 'Sensitivity', 'Specificity', 'Balanced Accuracy', 'Accuracy', 'Precision', 'F1-score', 'MCC', 'AUC', 'auPR'])

    for embedding in embeddings:
        print(embedding)
        random.seed(1)

        file_path_Benchmark_embeddings = embeddings[embedding]
        D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None)
        feature_X_Benchmark_embeddings = D_feature.iloc[:, 1:].values
        feature_y_Benchmark_embeddings = D_feature.iloc[:, 0].values
        print(feature_X_Benchmark_embeddings.shape)
        print(feature_y_Benchmark_embeddings.shape)

        feature_X_Benchmark = feature_X_Benchmark_embeddings.copy()

        preprocessed_feature_X_Benchmark = preprocess_the_dataset(feature_X_Benchmark)
        feature_y_Benchmark = feature_y_Benchmark_embeddings.copy()

        # Step 06 : Spliting with 10-FCV :
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

        # balance the dataset :
        rus = RandomUnderSampler(random_state=1)
        X, y = rus.fit_resample(preprocessed_feature_X_Benchmark, feature_y_Benchmark)

        c = Counter(y)
        print(c)

        # other classifiers except PLS :

        global_Sensitivity = []
        global_Specificity = []
        global_Balanced_acc = []
        global_Accuracy = []
        global_Precision = []
        global_AUPR = []
        global_F1 = []
        global_MCC = []
        global_AUC = []

        all_model_name = ['RF', 'ET', 'DT', 'MLP', 'LR', 'SVM', 'NB', 'KNN', 'XGB', 'PLS']

        for model_name in all_model_name:
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

                sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(model_name, y_test)

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

            print('classifier : ', model_name)
            print('Sensitivity : {0:.3f}'.format(np.mean(local_Sensitivity)))
            print('Specificity : {0:.3f}'.format(np.mean(local_Specificity)))
            print('Balanced_acc : {0:.3f}'.format(np.mean(local_Balanced_acc)))
            print('Accuracy : {0:.3f}'.format(np.mean(local_Accuracy)))
            print('Precision : {0:.3f}'.format(np.mean(local_Precision)))
            print('F1-score: {0:.3f}'.format(np.mean(local_F1)))
            print('MCC: {0:.3f}'.format(np.mean(local_MCC)))
            print('AUC: {0:.3f}'.format(np.mean(local_AUC)))
            print('auPR: {0:.3f}'.format(np.mean(local_AUPR)))

            global_Sensitivity.append(np.mean(local_Sensitivity))
            global_Specificity.append(np.mean(local_Specificity))
            global_Balanced_acc.append(np.mean(local_Balanced_acc))
            global_Accuracy.append(np.mean(local_Accuracy))
            global_Precision.append(np.mean(local_Precision))
            global_AUPR.append(np.mean(local_AUPR))
            global_F1.append(np.mean(local_F1))
            global_MCC.append(np.mean(local_MCC))
            global_AUC.append(np.mean(local_AUC))

            print('___________________________________________________________________________________________________________')
            print('___________________________________________________________________________________________________________')

        language_model_wise_benchmark_results_csv.writerow([embedding, '{0:.3f}'.format(np.mean(global_Sensitivity)),'{0:.3f}'.format(np.mean(global_Specificity)),'{0:.3f}'.format(np.mean(global_Balanced_acc)), '{0:.3f}'.format(np.mean(global_Accuracy)), '{0:.3f}'.format(np.mean(global_Precision)), '{0:.3f}'.format(np.mean(global_F1)), '{0:.3f}'.format(np.mean(global_MCC)), '{0:.3f}'.format(np.mean(global_AUC)), '{0:.3f}'.format(np.mean(global_AUPR))])
        print('___________________________________________________________________________________________________________')
        print('___________________________________________________________________________________________________________')
        print('___________________________________________________________________________________________________________')