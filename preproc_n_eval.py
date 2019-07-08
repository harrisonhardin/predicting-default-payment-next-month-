print("loading libraries...", end="", flush=True)
import pandas as pd
import numpy as np
from numba import njit, jit
import json
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTENC, SMOTE
print("done.")


def rename_cols(df):
    new_cols = list(df.iloc[0])
    new_cols = list(map(lambda x: re.sub(r"\s", "_", x).lower(), new_cols))
    df = df.iloc[1:-1]
    df.columns = new_cols
    return df


def drop_id(df):
    return df.drop(columns=["id"])


def df_to_numeric(df):
    def my_numeric(x):
        try:
            result = np.int32(x)
            return result
        except:
            pass
        try:
            result = np.float32(x)
            return result
        except:
            return np.nan

    # set numberic data to be the best numeric data they can even be :')
    return df.apply(my_numeric)


def remove_edu_and_mar_errata(df):
    # filter out unreasonable variables and unknowns
    df["education"] = df["education"].apply(lambda x: x if 1 <= x and x <= 4 else 4)
    df["marriage"] = df["marriage"].apply(lambda x: x if 1 <= x and x <= 3 else 3)
    return df


def get_sex_dummies(df):
    # get dummy variables and appropriately label
    sex_dummies = pd.get_dummies(df["sex"])
    sex_dummies.columns = ["male", "female"]
    #     return pd.concat([df, sex_dummies], sort=False)
    return pd.concat([df, sex_dummies], axis=1)


def get_edu_dummies(df):
    edu_dummies = pd.get_dummies(df["education"])
    edu_dummies.columns = ["graduate", "university", "high_school", "other_edu"]
    #     return pd.concat([df, edu_dummies], sort=False)
    return pd.concat([df, edu_dummies], axis=1)


def get_mar_dummies(df):
    mar_dummies = pd.get_dummies(df["marriage"])
    mar_dummies.columns = ["married", "single", "other_mar"]
    #     return pd.concat([df, mar_dummies], sort=False)
    return pd.concat([df, mar_dummies], axis=1)


def duly_payments_bool(df):
    # create booleans for duly payments for each month
    pay_columns = df.columns[5:11]
    duly_payments = []
    for c in pay_columns:
        temp = df[c].apply(lambda x: 0 if x > 0 else 1)
        duly_payments.append(temp)
    duly_payments_df = pd.concat(duly_payments, axis=1)
    duly_payments_df.columns = list(
        map(lambda x: x + "_duly", duly_payments_df.columns)
    )
    #     return pd.concat([df, duly_payments_df], sort=False)
    return pd.concat([df, duly_payments_df], axis=1)


def prepocess_df(df, preprocessors):
    for pp in preprocessors:
        df = pp(df)
    return df


# just run these functions in this lists passed to prepocess_df function
preprocessors = [
    rename_cols,
    drop_id,
    df_to_numeric,
    remove_edu_and_mar_errata,
    get_sex_dummies,
    get_edu_dummies,
    get_mar_dummies,
    duly_payments_bool,
]

print("preprocessing df...",end="", flush=True)
df = pd.read_csv("default_of_credit_card_clients.csv")
df = prepocess_df(df, preprocessors)
X = df.drop(columns="default_payment_next_month")
y = df["default_payment_next_month"]
print("done.")


print("SMOTE resampling...",end="", flush=True)
# do SMOTE resampling
categories = []
for i, c in enumerate(X.columns):
    u = pd.unique(X[c])
    if len(u) < 20:
        categories.append(i)
# smote_nc = SMOTENC(categorical_features=categories, random_state=0)
# X, y = smote_nc.fit_resample(X, y)
X, y = SMOTE().fit_resample(X, y)
print("done.")

def get_interaction_df(df, depth=2):
    from itertools import combinations as iter_combinations

    if depth < 2:
        return None
    combinations = list(iter_combinations(df.columns, depth))
    interactions = []
    for combo in combinations:
        interaction = df[combo[0]]
        for i in range(1, len(combo)):
            try:
                interaction = interaction * df[combo[i]]
            except:
                #                 print(interaction, df[combo[i]])
                print(combo[i])
        #                 printf("interaction type: %s\t df[combo] type: %s\n", interaction.dtype, df[combo[i]].dtypes)
        interactions.append(interaction)
    int_df = pd.concat(interactions, axis=1)
    int_df.columns = ["_".join(combo) for combo in combinations]
    return int_df


print("getting interaction terms...",end="", flush=True)
interactions_df = get_interaction_df(
    pd.DataFrame(
        X, columns=list(filter(lambda x: x != "default_payment_next_month", df.columns))
    )
)
print("done.")


@jit
def simple_feature_evaluation(X, y, n_splits=10):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    y = np.array(y)
    result = []
    for c in X.columns:

        x = np.array(X[c]).reshape(-1, 1)
        avg_f1 = 0
        avg_precision = 0
        avg_recall = 0

        clf = LogisticRegression(solver="lbfgs")

        avg_accuracy = 0
        avg_f1 = 0
        avg_precision = 0
        avg_recall = 0
        for train_index, test_index in cv.split(x):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            prob_preds = clf.predict_proba(X_test)

            if preds.sum() == 0:
                continue
            avg_accuracy += accuracy_score(y_test, preds)
            avg_f1 += f1_score(y_test, preds)
            avg_precision += precision_score(y_test, preds)
            avg_recall += recall_score(y_test, preds)

        avg_accuracy /= n_splits
        avg_f1 /= n_splits
        avg_precision /= n_splits
        avg_recall /= n_splits
        result.append(
            {
                "field": c,
                "avg_accuracy": avg_accuracy,
                "avg_precision": avg_precision,
                "avg_f1": avg_f1,
                "avg_recall": avg_recall,
            }
        )
    return result


# get the evaluations for interaction df and normal dfs
print("evaluating interaction terms data...",end="", flush=True)
interaction_evaluation = simple_feature_evaluation(interactions_df, y)
print("done.")
print("evaluating data...",end="", flush=True)
df_evaluation = simple_feature_evaluation(df, y)
print("done.")


# save both evaluations
print("saving evaluated data...",end="", flush=True)
with open("interaction_evaluation.json", "w") as f:
    f.write(json.dumps(interaction_evaluation))
with open("df_evaluation.json", "w") as f:
    f.write(json.dumps(df_evaluation))
print("done.")


print("writing interaction and data frame to csv...",end="", flush=True)
interactions_df.to_csv("interactions_df.csv")
df.to_csv("dataframe.csv")
print("done.")
