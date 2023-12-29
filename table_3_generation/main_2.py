import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef, average_precision_score

# import the classifiers :
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from collections import Counter
from sklearn.metrics import balanced_accuracy_score

from xgboost import XGBClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from imblearn.under_sampling import RandomUnderSampler

import random
import csv
from sklearn.preprocessing import PowerTransformer
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
        model = PLSRegression(n_components=1)
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
    prec = tp / (tp + fp)
    f1_score_1 = 2 * prec * sensitivity / (prec + sensitivity)
    mcc = matthews_corrcoef(y_test, y_predict)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    auPR = average_precision_score(y_test, y_proba[:, 1])  # auPR

    return sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR


class Metrics:
    def __init__(self):
        self.sensitivity = []
        self.specificity = []
        self.bal_acc = []
        self.acc = []
        self.prec = []
        self.f1_score_1 = []
        self.mcc = []
        self.auc = []
        self.auPR = []


features = {
    'PSSM': './all_required_csvs/Benchmark_with_PSSM.csv',
    'MonoGram': './all_required_csvs/Benchmark_with_monogram.csv',
    'DPC': './all_required_csvs/Benchmark_with_DPC.csv',
    'ASA': './all_required_csvs/Benchmark_with_ASA.csv',
    'HSE': './all_required_csvs/Benchmark_with_HSE.csv',
    'torsion_angles': './all_required_csvs/Benchmark_with_torsion_angles.csv',
    'Physicochemical': './all_required_csvs/Benchmark_with_physicochemical.csv',
    'ProtT5-XL-UniRef50': './all_required_csvs/benchmark_embeddings.csv',
}

model_metrics = {
    'NB': Metrics(),
    'KNN': Metrics(),
    'LR': Metrics(),
    'PLS': Metrics(),
    'SVM': Metrics(),
    'RF': Metrics(),
    'MLP': Metrics(),
    'DT': Metrics(),
    'ET': Metrics(),
    'XGB': Metrics()
}

with open('./output_csvs/model_wise_benchmark_results.csv', 'w') as f1:
    model_wise_benchmark_results_csv = csv.writer(f1)

    model_wise_benchmark_results_csv.writerow(['Learner', 'Sensitivity', 'Specificity', 'Balanced Accuracy', 'Accuracy', 'Precision', 'F1-score', 'MCC', 'auROC', 'auPR'])

    for feature in features:
        print(feature)
        random.seed(1)

        file_path_Benchmark_embeddings = features[feature]
        D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None,low_memory=False)
        if feature == 'ProtT5-XL-UniRef50':
            feature_X_Benchmark_embeddings = D_feature.iloc[:, 1:].values
            feature_y_Benchmark_embeddings = D_feature.iloc[:, 0].values
        else:
            feature_X_Benchmark_embeddings = D_feature.iloc[1:, 2:].values
            feature_y_Benchmark_embeddings = D_feature.iloc[1:, 0].values.astype(int)
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

        for model_name in model_metrics:
            print(model_name)
            i = 1
            for train_index, test_index in cv.split(X, y):
                X_train = X[train_index]
                X_test = X[test_index]

                y_train = y[train_index]
                y_test = y[test_index]

                sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(model_name, y_test)

                model_metrics[model_name].sensitivity.append(sensitivity)
                model_metrics[model_name].specificity.append(specificity)
                model_metrics[model_name].bal_acc.append(bal_acc)
                model_metrics[model_name].acc.append(acc)
                model_metrics[model_name].prec.append(prec)
                model_metrics[model_name].f1_score_1.append(f1_score_1)
                model_metrics[model_name].mcc.append(mcc)
                model_metrics[model_name].auc.append(auc)
                model_metrics[model_name].auPR.append(auPR)

                print(i, 'th iteration done')
                i = i + 1
                print('___________________________________________________________________________________________________________')

            print('classifier : ', model_name)
            print('Sensitivity : {0:.4f}'.format(np.mean(model_metrics[model_name].sensitivity)))
            print('Specificity : {0:.4f}'.format(np.mean(model_metrics[model_name].specificity)))
            print('Balanced_acc : {0:.4f}'.format(np.mean(model_metrics[model_name].bal_acc)))
            print('Accuracy : {0:.4f}'.format(np.mean(model_metrics[model_name].acc)))
            print('Precision : {0:.4f}'.format(np.mean(model_metrics[model_name].prec)))
            print('F1-score: {0:.4f}'.format(np.mean(model_metrics[model_name].f1_score_1)))
            print('MCC: {0:.4f}'.format(np.mean(model_metrics[model_name].mcc)))
            print('auROC: {0:.4f}'.format(np.mean(model_metrics[model_name].auc)))
            print('auPR: {0:.4f}'.format(np.mean(model_metrics[model_name].auPR)))

            print('___________________________________________________________________________________________________________')
            print('___________________________________________________________________________________________________________')



        print('___________________________________________________________________________________________________________')
        print('___________________________________________________________________________________________________________')
        print('___________________________________________________________________________________________________________')


    for model_name in model_metrics:
        model_wise_benchmark_results_csv.writerow(
            [model_name, '{0:.4f}'.format(np.mean(model_metrics[model_name].sensitivity)), '{0:.4f}'.format(np.mean(model_metrics[model_name].specificity)), '{0:.4f}'.format(np.mean(model_metrics[model_name].bal_acc)),
             '{0:.4f}'.format(np.mean(model_metrics[model_name].acc)), '{0:.4f}'.format(np.mean(model_metrics[model_name].prec)), '{0:.4f}'.format(np.mean(model_metrics[model_name].f1_score_1)),
             '{0:.4f}'.format(np.mean(model_metrics[model_name].mcc)), '{0:.4f}'.format(np.mean(model_metrics[model_name].auc)), '{0:.4f}'.format(np.mean(model_metrics[model_name].auPR))])
