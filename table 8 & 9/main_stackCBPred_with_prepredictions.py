import pandas as pd
import numpy as np
import csv
import pickle as pk
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC


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


test_set = ['TS49', 'TS88']

with open('./output_csvs/StackCBPred.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Dataset', 'Sensitivity', 'Specificity', 'Balanced_acc', 'Accuracy', 'F1-score', 'MCC', 'AUC', 'auPR'])
    for t in test_set:
        file_path = f'./StackCBPred/pred_file_test_{t}.csv'
        df = pd.read_csv(file_path, header=None)
        arr = df.values

        sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(arr[:, 0], arr[:, 1: 3], arr[:, 3])

        print('Sensitivity : {0:.4f}'.format(sensitivity))
        print('Specificity : {0:.4f}'.format(specificity))
        print('Balanced_acc : {0:.4f}'.format(bal_acc))
        print('Accuracy : {0:.4f}'.format(acc))
        print('F1-score: {0:.4f}'.format(f1_score_1))
        print('MCC: {0:.4f}'.format(mcc))
        print('auROC: {0:.4f}'.format(auc))
        print('auPR: {0:.4f}'.format(auPR))
        print('-------------------------------------')

        writer.writerow([t, '{0:.4f}'.format(sensitivity), '{0:.4f}'.format(specificity), '{0:.4f}'.format(bal_acc), '{0:.4f}'.format(acc), '{0:.4f}'.format(f1_score_1), '{0:.4f}'.format(mcc), '{0:.4f}'.format(auc), '{0:.4f}'.format(auPR)])
