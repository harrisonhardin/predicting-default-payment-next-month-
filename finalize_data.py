print("loading liraries...", end="", flush=True)
import pandas as pd
import numpy as np
from numba import njit, jit
import json
from sklearn.preprocessing import MinMaxScaler
print("done.")


def sprintf(fmt, *args):
    return fmt % (args)


def printf(fmt, *args):
    print(sprintf(fmt, *args), end="")


evaluation_keys = ["field", "avg_accuracy", "avg_precision", "avg_f1", "avg_recall"]

def json_from_file(name):
    result = None
    with open(name, "r") as f:
        result = json.loads(f.read())
        return result


def get_top_percent_features(data, percent):
    if not (0 < percent and percent < 100):
        return None

    global evaluation_keys
    m = int(percent * len(data) / 100)
    result = []
    for k in evaluation_keys:
        if k == "field":
            continue
        metric_sorted = sorted(data, key=lambda x: x[k], reverse=True)
        fields_sorted = list(map(lambda x: x["field"], metric_sorted))
        for i in range(m):
            result.append(fields_sorted[i])
    return list(set(result))


print("get top 10 percents terms from interaction data...", end="", flush=True)
interaction_evaluation_saved = json_from_file("interaction_evaluation.json")
final_interaction_labels = get_top_percent_features(interaction_evaluation_saved, 10)
print("done.")

print("get top 50 percents terms from df data...", end="", flush=True)
df_evaluation_saved = json_from_file("df_evaluation.json")
final_df_labels = get_top_percent_features(df_evaluation_saved, 50)
print("done.")

print("read in data frame...",end="", flush=True)
df = pd.read_csv("dataframe.csv")
print("done.")

print("read in interaction data frame...",end="", flush=True)
interactions_df = pd.read_csv("interactions_df.csv")
print("done.")

print("concatenate dataframes...",end="", flush=True)
final_df = pd.concat(
    [df[final_df_labels], interactions_df[final_interaction_labels]], axis=1
)
print("done.")

print("min max scale data...",end="", flush=True)
scaler = MinMaxScaler()
scaler.fit(final_df)
temp = final_df.columns
norm_data = scaler.transform(final_df)
final_df = pd.DataFrame(norm_data, columns=temp)
print("done.")


print("save selected features to csv...", end="", flush=True)
final_df.to_csv("selected_features.csv")
print("done.")
