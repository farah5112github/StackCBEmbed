import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

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
from collections import Counter


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

learner_combination = [
    ['XGB'], ['NB'], ['KNN'], ['LR'], ['PLS'], ['SVM'], ['MLP'], ['DT'], ['RF'], ['ET'],
    ['SVM', 'NB'], ['SVM', 'KNN'], ['SVM', 'LR'], ['SVM', 'PLS'], ['SVM', 'MLP'], ['SVM', 'DT'], ['SVM', 'RF'], ['SVM', 'ET'], ['SVM', 'XGB'],
    ['SVM', 'MLP', 'NB'], ['SVM', 'MLP', 'KNN'], ['SVM', 'MLP', 'LR'], ['SVM', 'MLP', 'PLS'], ['SVM', 'MLP', 'DT'], ['SVM', 'MLP', 'RF'], ['SVM', 'MLP', 'ET'], ['SVM', 'MLP', 'XGB'],
    ['SVM', 'MLP', 'RF', 'NB'], ['SVM', 'MLP', 'RF', 'KNN'], ['SVM', 'MLP', 'RF', 'LR'], ['SVM', 'MLP', 'RF', 'PLS'], ['SVM', 'MLP', 'RF', 'DT'], ['SVM', 'MLP', 'RF', 'ET'], ['SVM', 'MLP', 'RF', 'XGB'],
    ['SVM', 'MLP', 'ET', 'NB'], ['SVM', 'MLP', 'ET', 'KNN'], ['SVM', 'MLP', 'ET', 'LR'], ['SVM', 'MLP', 'ET', 'PLS'], ['SVM', 'MLP', 'ET', 'DT'], ['SVM', 'MLP', 'ET', 'RF'], ['SVM', 'MLP', 'ET', 'XGB'],
    ['SVM', 'MLP', 'ET', 'PLS', 'NB'], ['SVM', 'MLP', 'ET', 'PLS', 'KNN'], ['SVM', 'MLP', 'ET', 'PLS', 'LR'], ['SVM', 'MLP', 'ET', 'PLS', 'DT'], ['SVM', 'MLP', 'ET', 'PLS', 'RF'], ['SVM', 'MLP', 'ET', 'PLS', 'XGB'],
    ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'NB'], ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'KNN'], ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'LR'], ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'DT'], ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'XGB'],
    ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'KNN', 'NB'], ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'KNN', 'LR'], ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'KNN', 'DT'], ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'KNN', 'XGB'],
    ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'KNN', 'LR', 'NB'], ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'KNN', 'LR', 'DT'], ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'KNN', 'LR', 'XGB'],
    ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'KNN', 'LR', 'NB', 'DT'], ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'KNN', 'LR', 'NB', 'XGB'],
    ['SVM', 'MLP', 'ET', 'PLS', 'RF', 'KNN', 'LR', 'NB', 'DT', 'XGB'],
    ['SVM', 'MLP', 'RF', 'KNN', 'NB'], ['SVM', 'MLP', 'RF', 'KNN', 'LR'], ['SVM', 'MLP', 'RF', 'KNN', 'PLS'], ['SVM', 'MLP', 'RF', 'KNN', 'DT'],['SVM', 'MLP', 'RF', 'KNN', 'ET'], ['SVM', 'MLP', 'RF', 'KNN', 'XGB'],
    ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'NB'], ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'LR'], ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'PLS'], ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'DT'], ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'ET'],
    ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'ET', 'NB'], ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'ET', 'LR'], ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'ET', 'PLS'], ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'ET', 'DT'],
    ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'ET', 'NB', 'LR'], ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'ET', 'NB', 'PLS'], ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'ET', 'NB', 'DT'],
    ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'ET', 'NB', 'LR', 'PLS'], ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'ET', 'NB', 'LR', 'DT'],
    ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'ET', 'NB', 'PLS', 'LR'], ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'ET', 'NB', 'PLS', 'DT'],
    ['SVM', 'MLP', 'RF', 'KNN', 'XGB', 'ET', 'NB', 'PLS', 'LR', 'DT'],
]
others = ['PSSM']
file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-UniRef50']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
feature_y_Benchmark_embeddings = D_feature.iloc[:, 0].values

feature_X_Benchmark_embeddings = np.zeros((feature_y_Benchmark_embeddings.shape[0], 1), dtype=float)

file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-UniRef50']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 0:].values), axis=1)

for other in others:
    file_path_Benchmark_embeddings = feature_paths[other]
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
    feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[1:, 2:].values), axis=1)

feature_X_Benchmark_embeddings = np.delete(feature_X_Benchmark_embeddings, 0, axis=1)

feature_X_Benchmark_embeddings_positive = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 1, 1:]
feature_y_Benchmark_embeddings_positive = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 1, 0].astype('int')

feature_X_Benchmark_embeddings_negative = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 0, 1:]
feature_y_Benchmark_embeddings_negative = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 0, 0].astype('int')

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

feature_X_Benchmark_embeddings_train = preprocess_the_dataset(feature_X_Benchmark_embeddings_train)
feature_X_Benchmark_embeddings_test = preprocess_the_dataset(feature_X_Benchmark_embeddings_test)

X = feature_X_Benchmark_embeddings_train.copy()
y = feature_y_Benchmark_embeddings_train.copy()

rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(X, y)

c = Counter(y)
print(c)

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


with open('./output_csvs/model_wise_MI.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Learner Combination", "MI"])
    for learners in learner_combination:
        print(make_string(learners))
        prob = np.zeros((feature_X_Benchmark_embeddings_test.shape[0], 1), dtype=float)
        for learner in learners:
            prob = np.concatenate((prob, probabilities[learner]), axis=1)
        prob = np.delete(prob, 0, axis=1)

        realProb = np.zeros(prob.shape[0], dtype=float)
        for i in range(prob.shape[0]):
            realProb[i] = np.mean(prob[i])
        MI = mutual_info_classif(realProb.reshape(-1, 1), feature_y_Benchmark_embeddings_test, random_state=1)
        print('{0:.4f}'.format(np.mean(MI)))
        writer.writerow([make_string(learners), '{0:.4f}'.format(np.mean(MI))])
