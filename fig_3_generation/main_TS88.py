import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from matplotlib import pyplot as plt
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

from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef, average_precision_score, roc_curve
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

    base_classifiers = ['SVM', 'MLP', 'ET', 'PLS']

    for i in range(0, 10):
        for base_classifier in base_classifiers:
            pickle_file_path = str(pickle_folder_path + base_classifier + '_base_layer_' + str(i) + '.sav')
            # print(pickle_file_path)

            outfile = open(pickle_file_path, 'rb')
            model = pickle.load(outfile)
            outfile.close()

            if base_classifier == 'PLS':
                p_all = []
                p_all.append([1 - np.abs(item[0]) for item in model.predict(test_X)])
                p_all.append([np.abs(item[0]) for item in model.predict(test_X)])
                y_proba = np.transpose(np.array(p_all))[:, 1].reshape(-1, 1)
            else:
                y_proba = model.predict_proba(test_X)[:, 1].reshape(-1, 1)
            test_base_output_total = np.concatenate((test_base_output_total, y_proba), axis=1)

    test_base_output_total = np.delete(test_base_output_total, 0, axis=1)

    return test_base_output_total


def load_the_pickle_files_meta_layer(converted_all_features_with_output):
    test_X = preprocess_the_dataset(converted_all_features_with_output)

    if cmd == '1':
        pickle_file_path = str('./base_layer_pickle_files_with_PSSM/SVM_meta_layer.sav')
    elif cmd == '2':
        pickle_file_path = str('./base_layer_pickle_files_without_embedding/SVM_meta_layer.sav')
    else:
        pickle_file_path = str('./base_layer_pickle_files_without_PSSM/SVM_meta_layer.sav')

    outfile = open(pickle_file_path, 'rb')
    clf = pickle.load(outfile)
    outfile.close()

    y_pred = clf.predict(test_X)
    y_prob = clf.predict_proba(test_X)

    return y_pred, y_prob


feature_paths = {
    'PSSM': './all_required_csvs/TS88_with_PSSM.csv',
    'ProtT5-XL-UniRef50': './all_required_csvs/TS88_embeddings.csv',
}

cmds = ['2', '0', '1']
fprs = []
tprs = []
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

    #roc curve
    fpr, tpr, thresholds = roc_curve(y, y_proba[:, 1])
    print('fpr : ', fpr)
    print('tpr : ', tpr)
    fprs.append(fpr)
    tprs.append(tpr)

file_path = str('./StackCBPred/feature_file_train_ws1.csv')
train_df_1 = pd.read_csv(file_path, header=None)
train_1 = train_df_1.values
y_1 = train_1[:, 0]
X_1 = train_1[:, 1:]
scaler = StandardScaler()
X_scale_1 = scaler.fit_transform(X_1)
print(X_scale_1.shape)

# read the testing data file for window size 1
file_path = f'./StackCBPred/feature_file_test_ws1_TS88.csv'
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
file_path = f'./StackCBPred/feature_file_test_ws5_TS88.csv'
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

fpr, tpr, thresholds = roc_curve(y_test_5, y_proba[:, 1])
print('fpr : ', fpr)
print('tpr : ', tpr)
fprs.append(fpr)
tprs.append(tpr)

print(fprs)
print(tprs)
plt.plot([0.0, 1.0], [0.0, 1.0], color="darkblue", linestyle='--')
plt.plot(fprs[0], tprs[0], color="cornflowerblue", label='StackCBEmbed (PSSM)')
plt.plot(fprs[1], tprs[1], color="orange", label='StackCBEmbed (Embedding)')
plt.plot(fprs[2], tprs[2], color="limegreen", label='StackCBEmbed')
plt.plot(fprs[3], tprs[3], color="red", label='StackCBPred')
plt.title('Receiver operating characteristic curve (TS-88)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

