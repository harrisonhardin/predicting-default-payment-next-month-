"""
The modeling strategy is for this script to first take a numeric arguement
corresponding the model that will be used.  Then the modeling config will be
read and random values from the model config will be reading as hyper parameters
for the given model.  The model will then append its score to an inline dictionary
and save a line to the model_manifest.txt.  The model will then be pickled and
saved with the filename #####model.pkl.

These trainings will multiprocessed with the multipython go program and each
python script will be given a timeout of 2 min each to complete training less
they be terminated.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
from numba import njit, jit
import toml
import os
import random
import json
import pickle

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# this list is treated like an enumerable in this program and is used to pick
# the model used to classify the data.  The model picked is based of the args
models_list = [
    {"name": "LogisticRegression", "model": LogisticRegression},
    {"name": "GaussianMixture", "model": GaussianMixture},
    {"name": "DecisionTreeClassifier", "model": DecisionTreeClassifier},
    {"name": "RandomForestClassifier", "model": RandomForestClassifier},
    {"name": "AdaBoostClassifier", "model": AdaBoostClassifier},
    {"name": "GradientBoostingClassifier", "model": GradientBoostingClassifier},
    {"name": "KNeighborsClassifier", "model": KNeighborsClassifier},
]

# get the script id that will be used to pick the model
model_type_id = int(sys.argv[1])

# get the number of models created and save as the model id number
model_id = None
with open("models_created.txt", "r", os.O_NONBLOCK) as f:
    model_id = int(f.read())
model_id += 1
# save the incremented model id to the file
with open("models_created.txt", "w", os.O_NONBLOCK) as f:
    f.write(str(model_id))

# load in the data
print("loading data...", end="", flush=True)
df = None
with open("selected_features.csv", "r", os.O_NONBLOCK) as f:
    df = pd.read_csv(f)
    df = df.dropna()

print("done.")
X = df.drop(columns=["default_payment_next_month"])
y = df["default_payment_next_month"]

# do SMOTE resampling
print("smoteing the data...", end="", flush=True)
categories = []
for i, c in enumerate(X.columns):
    u = pd.unique(X[c])
    if len(u) < 20:
        categories.append(i)
X, y = SMOTE().fit_resample(X, y)
print("done.")


# load in the model config
model_config = None
with open("model_config.toml", "r", os.O_NONBLOCK) as f:
    model_config = toml.load(f, _dict=dict)


# fit_predict_score will give the results of the models predicctions on the data
# and will return a dictionary containing the id of the model and various scores
def fit_predict_score(clf, X_train, y_train, _id):
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    prob_preds = clf.predict(X_test)
    return {
        "model": "%05d" % _id + model["name"],
        "avg_accuracy": accuracy_score(y_test, preds),
        "avg_precision": f1_score(y_test, preds),
        "avg_f1": precision_score(y_test, preds),
        "avg_recall": recall_score(y_test, preds),
    }


# chose the model to be using
model = models_list[model_type_id % len(models_list)]
params = model_config[model["name"]]

# get random parameters from the configuration file
selected_params = {}
for param_name in params:
    param_value = params[param_name]
    if type(param_value) == list:
        param_value = param_value[random.randint(0, len(param_value) - 1)]
    selected_params[param_name] = param_value

# create a model of the name found in the config with the chosen params
clf = model["model"](**selected_params)

# train the model on the data with test train split evaluation
cv = KFold(n_splits=3, shuffle=True, random_state=0)
results = []
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    results.append(fit_predict_score(clf, X_train, y_train, model_id))

# append the results to the model manifest file
with open("model_manifest.txt", "a", os.O_NONBLOCK) as f:
    cv_json_line = json.dumps(results) + "\n"
    f.write(cv_json_line)

# save the model in a pickle file in the correct model directory
with open(model["name"] + "/%05dmodel.pkl" % model_id, "wb", os.O_NONBLOCK) as f:
    pickle.dump(model["model"], f)
