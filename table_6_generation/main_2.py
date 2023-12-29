import pandas as pd
import numpy as np

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
        model = PLSRegression(n_components=1)
    else:
        print('Wrong model name')
        return

    model.fit(X_train, y_train)

    return model


def make_string(s):
    str = ''
    for i in s:
        str += i + ", "
    return str[:-2]


feature_paths = {
    'PSSM': './all_required_csvs/Benchmark_with_PSSM.csv',
    'ProtT5-XL-UniRef50': './all_required_csvs/benchmark_embeddings.csv',
}

learner_combination = [['XGB'],
                       ['XGB', 'NB'], ['XGB', 'KNN'], ['XGB', 'LR'], ['XGB', 'PLS'], ['XGB', 'SVM'], ['XGB', 'MLP'], ['XGB', 'DT'], ['XGB', 'RF'], ['XGB', 'ET'],
                       ['XGB', 'SVM', 'NB'], ['XGB', 'SVM', 'KNN'], ['XGB', 'SVM', 'LR'], ['XGB', 'SVM', 'PLS'], ['XGB', 'SVM', 'MLP'], ['XGB', 'SVM', 'DT'], ['XGB', 'SVM', 'RF'], ['XGB', 'SVM', 'ET'],
                        ['XGB', 'SVM', 'ET','NB'], ['XGB', 'SVM', 'ET', 'KNN'], ['XGB', 'SVM', 'ET', 'LR'], ['XGB', 'SVM', 'ET', 'PLS'], ['XGB', 'SVM', 'ET', 'MLP'], ['XGB', 'SVM', 'ET', 'DT'], ['XGB', 'SVM', 'ET', 'RF'],
                       ]
pssm = True
file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-UniRef50']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
feature_X_Benchmark_embeddings_positive = D_feature.loc[D_feature[D_feature.columns[0]] == 1, 1:].values
feature_y_Benchmark_embeddings_positive = D_feature.loc[D_feature[D_feature.columns[0]] == 1, 0].values

feature_X_Benchmark_embeddings_negative = D_feature.loc[D_feature[D_feature.columns[0]] == 0, 1:].values
feature_y_Benchmark_embeddings_negative = D_feature.loc[D_feature[D_feature.columns[0]] == 0, 0].values

if pssm:
    file_path_Benchmark_embeddings = feature_paths['PSSM']
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
    feature_X_Benchmark_embeddings_positive = np.concatenate((feature_X_Benchmark_embeddings_positive, D_feature.loc[D_feature[D_feature.columns[0]] == '1', 2:].values), axis=1)
    feature_X_Benchmark_embeddings_negative = np.concatenate((feature_X_Benchmark_embeddings_negative, D_feature.loc[D_feature[D_feature.columns[0]] == '0', 2:].values), axis=1)

print(feature_X_Benchmark_embeddings_positive.shape)
print(feature_y_Benchmark_embeddings_positive.shape)

print(feature_X_Benchmark_embeddings_negative.shape)
print(feature_y_Benchmark_embeddings_negative.shape)

feature_X_Benchmark_embeddings_positive_train, feature_X_Benchmark_embeddings_positive_test, feature_y_Benchmark_embeddings_positive_train, feature_y_Benchmark_embeddings_positive_test = train_test_split(feature_X_Benchmark_embeddings_positive, feature_y_Benchmark_embeddings_positive, test_size=275, random_state=1)
feature_X_Benchmark_embeddings_negative_train, feature_X_Benchmark_embeddings_negative_test, feature_y_Benchmark_embeddings_negative_train, feature_y_Benchmark_embeddings_negative_test = train_test_split(feature_X_Benchmark_embeddings_negative, feature_y_Benchmark_embeddings_negative, test_size=7741, random_state=1)


print(feature_X_Benchmark_embeddings_positive_train.shape)
print(feature_X_Benchmark_embeddings_positive_test.shape)

print(feature_X_Benchmark_embeddings_negative_train.shape)
print(feature_X_Benchmark_embeddings_negative_test.shape)

feature_X_Benchmark_embeddings_train = np.concatenate((feature_X_Benchmark_embeddings_positive_train, feature_X_Benchmark_embeddings_negative_train), axis=0)
feature_y_Benchmark_embeddings_train = np.concatenate((feature_y_Benchmark_embeddings_positive_train, feature_y_Benchmark_embeddings_negative_train), axis=0)
feature_X_Benchmark_embeddings_test = np.concatenate((feature_X_Benchmark_embeddings_positive_test, feature_X_Benchmark_embeddings_negative_test), axis=0)
feature_y_Benchmark_embeddings_test = np.concatenate((feature_y_Benchmark_embeddings_positive_test, feature_y_Benchmark_embeddings_negative_test), axis=0)

print(feature_X_Benchmark_embeddings_train.shape)
print(feature_y_Benchmark_embeddings_train.shape)

print(feature_X_Benchmark_embeddings_test.shape)
print(feature_y_Benchmark_embeddings_test.shape)

X = feature_X_Benchmark_embeddings_train.copy()
y = feature_y_Benchmark_embeddings_train.copy()

X = preprocess_the_dataset(X)

probabilities = {}
model_names = ['XGB', 'NB', 'KNN', 'LR', 'PLS', 'SVM', 'MLP', 'DT', 'RF', 'ET']
for model_name in model_names:
    print(model_name+" model fit is running")
    if model_name == 'PLS':
        model = model_fit(model_name, X, y)
        p_all = []
        p_all.append([1 - np.abs(item[0]) for item in model.predict(feature_X_Benchmark_embeddings_test)])
        p_all.append([np.abs(item[0]) for item in model.predict(feature_X_Benchmark_embeddings_test)])
        probabilities[model_name] = np.transpose(np.array(p_all))[:, 1].reshape(-1, 1)
    else:
        model = model_fit(model_name, X, y)
        probabilities[model_name] = model.predict_proba(feature_X_Benchmark_embeddings_test)[:, 1].reshape(-1, 1)

with open('./output_csvs/model_wise_MI_2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Learner Combination', 'MI'])
    for learners in learner_combination:
        print(make_string(learners))
        global_MI = []
        for learner in learners:
            MI = mutual_info_classif(probabilities[learner], feature_y_Benchmark_embeddings_test)
            global_MI.append(np.mean(MI))
        print('{0:.4f}'.format(np.mean(global_MI)))
        writer.writerow([make_string(learners), '{0:.4f}'.format(np.mean(global_MI))])
