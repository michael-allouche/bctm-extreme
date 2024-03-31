import pandas as pd
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import yaml

# import sys
# sys.path.append("'/Users/michaelallouche/PhD/repos/nn-ES/nn-risk-measures'")



def get_best_models(source):
    """returns the best models with gamma=0.5 and rho=-1 for each order condition"""
    df_summary = load_summary(source)
    dict_best_models = defaultdict(list)

    order_conditions = np.sort(df_summary["n_orders"].unique())
    for order in order_conditions:
        df_order_summary = df_summary[df_summary["n_orders"]==order]
        filename, epoch, loss = get_best_order_model(df_order_summary)
        # save it
        dict_best_models["order"].append(order)
        dict_best_models["filename"].append(filename)
        dict_best_models["epoch"].append(epoch)
        dict_best_models["loss"].append(loss)
    return pd.DataFrame.from_dict(dict_best_models)


def load_summary(source):
    df_summary = pd.read_csv(Path("ckpt", source, "_config_summary.csv"), sep=";")
    df_summary.drop_duplicates(subset=["model_filename"], inplace=True, keep="last")
    df_summary.dropna(axis=0, how="all", inplace=True)
    df_summary.to_csv(Path("ckpt", source, "_config_summary.csv"), header=True, index=False, sep=";")  # save it again
    # ----
    return df_summary

def load_summary_file(filename, source):
    df_summary = load_summary(source)
    file_summary = df_summary[df_summary["model_filename"] == filename].to_dict('records')[0]
    if source == "sim":
        file_summary["params"] = yaml.safe_load(file_summary["params"])
    return file_summary

def get_config(source):
    """
    load the .yaml config file
    Returns
    -------
    dict:
        config file
    """

    with(open(Path("configs", source, "config_file.yaml"))) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader) 
    return config



def flatten_dict(nested_dict):
    """flatten a 2 level nested dict"""
    out = {}
    list_subdict = nested_dict.values()
    for subdict in list_subdict:
        for key, val in subdict.items():
            out[key] = [val] #str(val)
    return out

def nested_dict_to_df(nested_dict):
    """convert a nested dict to a multi index Dataframe"""
    new_dict = {}
    for outerKey, innerDict in nested_dict.items():
        for innerKey, values in innerDict.items():
            new_dict[(outerKey, innerKey)] = values
    return pd.DataFrame(new_dict)
