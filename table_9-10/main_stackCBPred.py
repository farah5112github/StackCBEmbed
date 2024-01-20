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

with open('./output_csvs/metrics_StackCBPred.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Dataset', 'Sensitivity', 'Specificity', 'Balanced_acc', 'Accuracy', 'F1-score', 'MCC', 'AUC', 'auPR'])
    for t in test_set:
        file_path = str('./StackCBPred/feature_file_train_ws1.csv')
        train_df_1 = pd.read_csv(file_path, header=None)
        train_1 = train_df_1.values
        y_1 = train_1[:, 0]
        X_1 = train_1[:, 1:]
        scaler = StandardScaler()
        X_scale_1 = scaler.fit_transform(X_1)
        print(X_scale_1.shape)

        # read the testing data file for window size 1
        file_path = f'./StackCBPred/feature_file_test_ws1_{t}.csv'
        test_df_1 = pd.read_csv(file_path, header=None)

        test_1 = test_df_1.values
        X_test_1 = test_1[:, 1:]
        X_test_scale_1 = scaler.transform(X_test_1)
        print(X_test_scale_1.shape)

        # read the training data file for window size 5
        file_path = str('./StackCBPred/feature_file_train_ws5.csv')
        train_df_5 = pd.read_csv(file_path, header=None)
        train_5 = train_df_5.values
        y_5 = train_5[:, 0]
        X_5 = train_5[:, 1:]
        scaler = StandardScaler()
        X_scale_5 = scaler.fit_transform(X_5)
        print(X_scale_5.shape)

        # read the testing data file for window size 5
        file_path = f'./StackCBPred/feature_file_test_ws5_{t}.csv'
        test_df_5 = pd.read_csv(file_path, header=None)

        test_5 = test_df_5.values
        X_test_5 = test_5[:, 1:]
        y_test_5 = test_5[:, 0]
        X_test_scale_5 = scaler.transform(X_test_5)
        print(X_test_scale_5.shape)

        ################################ First base layer-> LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=1)
        model.fit(X_scale_5, y_5)
        y_pred_log_prob = model.predict_proba(X_test_scale_5)
        #print("logreg_predicted")

        ########################## Second Base layer is ExtraTree##########################################
        model = ExtraTreesClassifier(n_estimators=1000, random_state=1)
        model.fit(X_scale_5, y_5)
        y_pred_etc_prob = model.predict_proba(X_test_scale_5)
        #print("etc_predicted")


        ########################### Third base layer ML method is knn ###############################
        model = KNeighborsClassifier(n_neighbors=7)
        model.fit(X_scale_1, y_1)
        y_pred_knn_prob = model.predict_proba(X_test_scale_1)
        #print("knn_predicted")

        #################################### Fourth base layer is BaggingClassifier ###################
        model = SVC(C=2.378414230005442, kernel='rbf', gamma=0.013139006488339289, probability=True, random_state=1)
        model.fit(X_scale_5, y_5)
        y_pred_svm_prob = model.predict_proba(X_test_scale_5)

        ############################### Combine probabilities of base layer to the original features and run SVM ##############################
        train_df = pd.read_csv('./StackCBPred/Model_9_stacking_train_SVM_meta.csv', header=None)
        train = train_df.values
        y = train[:, 0]
        X = train[:, 1:]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_meta_test = np.column_stack([X_test_5, y_pred_log_prob, y_pred_knn_prob, y_pred_etc_prob, y_pred_svm_prob])
        X_scale_test_SVM = scaler.transform(X_meta_test)

        # clf = SVC(C=grid_fine.best_params_['C'],kernel='rbf',gamma=grid_fine.best_params_['gamma'])
        model = SVC(C=0.21022410381342863, gamma=0.0011613350732448448, probability=True, random_state=1)
        model.fit(X, y)
        y_pred = model.predict(X_scale_test_SVM)
        y_proba = model.predict_proba(X_scale_test_SVM)

        sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(y_pred, y_proba, y_test_5)


        print('Sensitivity : {0:.3f}'.format(sensitivity))
        print('Specificity : {0:.3f}'.format(specificity))
        print('Balanced_acc : {0:.3f}'.format(bal_acc))
        print('Accuracy : {0:.3f}'.format(acc))
        print('F1-score: {0:.3f}'.format(f1_score_1))
        print('MCC: {0:.3f}'.format(mcc))
        print('AUC: {0:.3f}'.format(auc))
        print('auPR: {0:.3f}'.format(auPR))

        writer.writerow([t, '{0:.3f}'.format(sensitivity), '{0:.3f}'.format(specificity), '{0:.3f}'.format(bal_acc),
                         '{0:.3f}'.format(acc), '{0:.3f}'.format(f1_score_1), '{0:.3f}'.format(mcc),
                         '{0:.3f}'.format(auc), '{0:.3f}'.format(auPR)])

