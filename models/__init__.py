import pandas as pd

from utils import load_summary, load_summary_file, nested_dict_to_df
from extreme.data_management import load_quantiles, DataSampler, load_real_data
from extreme.estimators import random_forest_k
from models.metrics import compute_criteria
import numpy as np
from pathlib import Path

# we considered several criterias. For the paper, we selected the MAD
list_criterias = ["variance", "r_variance", "mad", "r_mad", "aad", "r_aad"]

def get_best_order_model(df_order_summary, criteria, condition):
    """returns the best model for a given order condition"""
    best_metric = None  # init the best metric
    for file in df_order_summary.iterrows():
        filename = file[1]["model_filename"]
        pt_ckpt = torch.load(Path("ckpt",  condition, "{}.pt".format(filename)), map_location="cpu")
        metric = pt_ckpt["eval"][criteria]["value"]

        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_epoch = pt_ckpt["eval"][criteria]["epoch"]
            best_filename = filename
    return best_filename, best_epoch, best_metric

def get_best_crit(filename, distribution=None, source='sim'):
    """returns the best epoch and value of a specific filename"""
    df = pd.DataFrame(columns=list_criterias, index=["epoch", "value"])
    if distribution is None:
        config_file = load_summary_file(filename=filename, source=source)
        if source == "sim":
            distribution=config_file["distribution"]
    for criteria in list_criterias:
        if source == "sim":
            pt_ckpt = torch.load(Path("ckpt", source, distribution, "training", "{}.pt".format(filename)), map_location="cpu")
        else:
            pt_ckpt = torch.load(Path("ckpt", source, "training", "{}.pt".format(filename)), map_location="cpu")
        df.loc["epoch", criteria] = pt_ckpt["eval"][criteria]["epoch"]
        df.loc["value", criteria] = pt_ckpt["eval"][criteria]["value"]
    return df


def load_model(filename, epoch, distribution=None, from_config=False, config_file=None, source="sim"):
    # dict_models = {"sim": ExtrapolateNN, "LDreal": LDExtrapolateNN, "Creal": CExtrapolateNN}
    # path =  Path('/Users', 'model_selectionche', 'PhD', 'repos', 'nn-ES', 'nn-risk-measures', 'ckpt', distribution, "training", "{}.pt".format(filename))
    if not from_config:
        config_file = load_summary_file(filename=filename, source=source)
    model = ExpectedShortfallNN(**config_file)
    if source == "sim":
        pt_ckpt = torch.load(Path("ckpt", source, distribution, "training", "{}.pt".format(filename)), map_location="cpu")
    else:
        pt_ckpt = torch.load(Path("ckpt", source, "training", "{}.pt".format(filename)),
                             map_location="cpu")
    # pt_ckpt = torch.load(path, map_location="cpu")
    model.net.load_state_dict(pt_ckpt["epoch{}".format(epoch)]["params"])
    model.optimizer.load_state_dict(pt_ckpt["epoch{}".format(epoch)]["optimizer"])

    return model

def model_selection(distribution, params, n_replications, metric="median", **kwargs):
    """
    returns the best models for each NN order condition and eval criteria, given a specific parametrization
    Parameters
    ----------
    distribution : str
    params : dict
    metric : str
    n_replications : int
    kwargs :

    Returns
    -------

    """
    # pathdir = Path('/Users', 'michaelallouche', 'PhD', 'repos', 'nn-ES', 'nn-risk-measures', 'ckpt', distribution,
    #                "extrapolation", str(params))

    pathdir = Path("ckpt", "sim", distribution, "extrapolation", str(params))
    pathdir.mkdir(parents=True, exist_ok=True)

    df_summary = load_summary()
    
    df_summary = df_summary[df_summary["replications"] == n_replications]  # filter on the number of replications
    
    df_summary = df_summary[df_summary["distribution"] == distribution]  # filter on the distributions
    
    # filer on a specific parametrization
    def params_filtering(x):
        x_dict = eval(x)
        return x_dict == params

    df_summary = df_summary[df_summary['params'].apply(params_filtering)] # filter on the parameters
   
    try:
        nn_order_trunc = np.sort(df_summary["trunc"].unique())  # list the different neural network truncation
    except KeyError:
        raise KeyError("No such parametrization is listed !")
       # print("No such parametrization is listed !")

    dict_best_nn = {"NN_{}".format(order): {crit: {"filename": None, "value": None, "rmse_bestK": None} for crit in list_criterias} for order in
                 nn_order_trunc}  # save the best extrapolation model for each criteria and orderNN
    
    for order in nn_order_trunc: # for each NN order condition
        df_order_summary = df_summary[(df_summary["trunc"] == order)]
        models_filenames = df_order_summary["model_filename"].str.replace(r"-rep*[0-9]+", "", regex=True).unique()  # unique model filenames

        for model_filename in models_filenames:  # for each model
            print(model_filename)
            try:
                dict_nn = model_evaluation(model_filename)

                for criteria in list_criterias:
                    # condition to find the best model given each NN order and criteria (just windowed)
                    best_metric = dict_best_nn["NN_{}".format(order)][criteria]["value"]
                    if best_metric is None or dict_nn[criteria][metric]["value"] < best_metric:
                        dict_best_nn["NN_{}".format(order)][criteria]["value"] = dict_nn[criteria][metric]["value"]
                        dict_best_nn["NN_{}".format(order)][criteria]["filename"] = model_filename
                        dict_best_nn["NN_{}".format(order)][criteria]["rmse_bestK"] = dict_nn[criteria][metric]["rmse_bestK"]
            except Exception as e:
                print(e)
    return nested_dict_to_df(dict_best_nn).T


def model_evaluation(model_filename):
    df_summary = load_summary(source="sim")
    config_model = df_summary[df_summary["model_filename"] == model_filename + "-rep1"]
    n_replications = int(config_model["replications"].values[0])
    # ----------------------------------
    # Check Missing rows in config_file
    for i in range(1, n_replications+1):
        if df_summary[df_summary["model_filename"] == model_filename + "-rep{}".format(i)].shape[0] == 0:
            print(i)
    # ----------------------------------
    nrep_files = df_summary[df_summary["model_filename"].str.contains(model_filename)].shape[0]  # number of rep files
    if nrep_files == n_replications: # only if all replications are saved
    #if 1 == 1: # pass
        print("running {} ...".format(model_filename))
        
        # initialize utilities
        n_data = int(config_model["n_data"].values[0])
        zeta = config_model["zeta"].values[0]
        distribution = config_model["distribution"].values[0]
        params = eval(config_model["params"].values[0])

        pathdir = Path("ckpt", "sim", distribution, "extrapolation", str(params))
        pathdir.mkdir(parents=True, exist_ok=True)
        pathfile = Path(pathdir, "{}.npy".format(model_filename))
        try:
            dict_nn = np.load(pathfile, allow_pickle=True)[()]
        except FileNotFoundError:
            # save all info in a new dictionary
            dict_nn = {criteria: {_metric:{"value": [], "series": [],
                            "var": None, "rmse": None, "rmse_bestK": None, "q_bestK": [],
                            "bestK": []} for _metric in ["mean", "median"]} for criteria in list_criterias}

            # data:  model filenames may not have the same number of data

            anchor_points = np.arange(2 + int(zeta*n_data), n_data)  # k = 2, ..., n-1
            EXTREME_ALPHA = 1/(2*n_data)  # extreme alpha
            summary_file = load_summary_file(model_filename + "-rep1", source='sim')
            data_sampler = DataSampler(**summary_file)
            real_quantile = data_sampler.ht_dist.tail_ppf(1/EXTREME_ALPHA)  # real extreme quantile
            real_es = data_sampler.ht_dist.expected_shortfall(1 - EXTREME_ALPHA)  # real extreme quantile

            for replication in range(1, n_replications + 1):  # for each replication
                print("rep: ", replication)
                # ------------------------
                # Neural Network estimator
                # ------------------------
                model_filename_rep = model_filename + "-rep{}".format(replication)
                # best epoch and value at each replication
                best_parametrization = get_best_crit(filename=model_filename_rep, distribution=distribution)
                # print("ok")
                # Order statistics
                X_order = load_quantiles(distribution, params, n_data, rep=replication)  # replication order statistics X_1,n, ..., X_n,n
                # print("ok")
                for criteria in list_criterias:  # for each criteria
                    
                    best_epoch_rep = int(best_parametrization.loc["epoch", criteria])
                    # NN extrapolation
                    model = load_model(model_filename_rep, best_epoch_rep, distribution)
                    # print("ok")

                    # extrapolation
                    q_nn = model.extrapolate(alpha=EXTREME_ALPHA, k_anchor=anchor_points, X_order=X_order).ravel()
                    # print("ok")
                    # find the best k
                    bestK_nn = random_forest_k(q_nn, 10000)  # for k=15,...,375 (i=13,...,373)
                    # MEAN (RMSE)
                    dict_nn[criteria]["mean"]["value"].append(best_parametrization.loc["value", criteria])
                    dict_nn[criteria]["mean"]["series"].append(q_nn)
                    dict_nn[criteria]["mean"]["q_bestK"].append(q_nn[int(bestK_nn)])
                    dict_nn[criteria]["mean"]["bestK"].append(bestK_nn+2)  # k = i +2, with Python index i=(0,...,497)

                    # MEDIAN (RMedSE)
                    dict_nn[criteria]["median"]["value"].append(best_parametrization.loc["value", criteria])
                    dict_nn[criteria]["median"]["series"].append(q_nn)
                    dict_nn[criteria]["median"]["q_bestK"].append(q_nn[int(bestK_nn)])
                    dict_nn[criteria]["median"]["bestK"].append(bestK_nn+2)  # k = i +2, with Python index i=(0,...,497)
                # ----------------------

            for criteria in list_criterias:
                # MEAN (RMSE)
                q_nn_mean_series = np.array(dict_nn[criteria]["mean"]["series"])
                dict_nn[criteria]["mean"]["value"] = np.mean(dict_nn[criteria]["mean"]["value"])  # mean of all best values in order to compare NN models
                dict_nn[criteria]["mean"]["var"] = q_nn_mean_series.var(axis=0)  # variance between the replications
                dict_nn[criteria]["mean"]["series"] = q_nn_mean_series.mean(axis=0)  # mean between the replications
                dict_nn[criteria]["mean"]["rmse"] = ((q_nn_mean_series / real_es - 1) ** 2).mean(axis=0)  # rmse of the series
                dict_nn[criteria]["mean"]["rmse_bestK"] = ((np.array(dict_nn[criteria]["mean"]["q_bestK"]) / real_es - 1) ** 2).mean()  # rmse metric on all the selected  q(k^star)

                # MEDIAN (RMedSE)
                q_nn_median_series = np.array(dict_nn[criteria]["median"]["series"])
                dict_nn[criteria]["median"]["value"] = np.median(dict_nn[criteria]["median"]["value"])  # median of all best values in order to compare NN models
                dict_nn[criteria]["median"]["var"] = q_nn_median_series.var(axis=0)  # variance between the replications
                dict_nn[criteria]["median"]["series"] = np.median(q_nn_median_series, axis=0)  # median between the replications
                dict_nn[criteria]["median"]["rmse"] = np.median((q_nn_median_series / real_es - 1) ** 2, axis=0)  # rmedse between the replications
                dict_nn[criteria]["median"]["rmse_bestK"] = np.median((np.array(dict_nn[criteria]["median"]["q_bestK"]) / real_es - 1) ** 2)  # rmsede metric on all the selected  q(k^star)

            np.save(pathfile, dict_nn)
            # ----------------------
        finally:
            return dict_nn
    else:
        print(model_filename, "not complet: ", nrep_files, ' files')
    return


def model_evaluation_real(model_filename):
    """For real data"""
    df_summary = load_summary(source="real")
    config_model = df_summary[df_summary["model_filename"] == model_filename]
    # ----------------------------------
    print("running {} ...".format(model_filename))

    # initialize utilities
    zeta = config_model["zeta"].values[0]
    year = config_model["year"].values[0]
    xi = config_model["xi"].values[0]
    percentile = config_model["percentile"].values[0]
    X_train, order_train, order_test = load_real_data(year=year, xi=xi, zeta=zeta, percentile=percentile)
    n_train = order_train.shape[0]

    pathdir = Path("ckpt", "real", "extrapolation")
    pathdir.mkdir(parents=True, exist_ok=True)
    pathfile = Path(pathdir, "{}.npy".format(model_filename))
    try:
        dict_nn = np.load(pathfile, allow_pickle=True)[()]
    except FileNotFoundError:
        # save all info in a new dictionary
        dict_nn = {criteria: {_metric: {"value": [], "series": [],
                                        "var": None, "rmse": None, "rmse_bestK": None, "q_bestK": [],
                                        "bestK": []} for _metric in ["mean", "median"]} for criteria in
                   list_criterias}

        # data:  model filenames may not have the same number of data

        anchor_points = np.arange(2 + int(zeta * n_train), n_train)  # k = 2, ..., n-1
        EXTREME_ALPHA = 1 / (n_train)  # extreme alpha
        summary_file = load_summary_file(model_filename, source="real")
        real_es = np.mean(order_test)


        # ------------------------
        # Neural Network estimator
        # ------------------------
        # best epoch and value at each replication
        best_parametrization = get_best_crit(filename=model_filename, source="real")

        for criteria in list_criterias:  # for each criteria

            best_epoch_rep = int(best_parametrization.loc["epoch", criteria])
            # NN extrapolation
            model = load_model(model_filename, best_epoch_rep, source='real')
            q_nn = model.extrapolate(alpha=EXTREME_ALPHA, k_anchor=anchor_points, X_order=order_train).ravel()
            # find the best k
            bestK_nn = random_forest_k(q_nn, 10000)  # for k=15,...,375 (i=13,...,373)
            # MEAN (RMSE)
            dict_nn[criteria]["mean"]["value"].append(best_parametrization.loc["value", criteria])
            dict_nn[criteria]["mean"]["series"].append(q_nn)
            dict_nn[criteria]["mean"]["q_bestK"].append(q_nn[int(bestK_nn)])
            dict_nn[criteria]["mean"]["bestK"].append(bestK_nn + 2)  # k = i +2, with Python index i=(0,...,497)

            # MEDIAN (RMedSE)
            dict_nn[criteria]["median"]["value"].append(best_parametrization.loc["value", criteria])
            dict_nn[criteria]["median"]["series"].append(q_nn)
            dict_nn[criteria]["median"]["q_bestK"].append(q_nn[int(bestK_nn)])
            dict_nn[criteria]["median"]["bestK"].append(
                bestK_nn + 2)  # k = i +2, with Python index i=(0,...,497)
        # ----------------------

        for criteria in list_criterias:
            # MEAN (RMSE)
            q_nn_mean_series = np.array(dict_nn[criteria]["mean"]["series"])
            dict_nn[criteria]["mean"]["value"] = np.mean(
                dict_nn[criteria]["mean"]["value"])  # mean of all best values in order to compare NN models
            dict_nn[criteria]["mean"]["var"] = q_nn_mean_series.var(axis=0)  # variance between the replications
            dict_nn[criteria]["mean"]["series"] = q_nn_mean_series.mean(axis=0)  # mean between the replications
            dict_nn[criteria]["mean"]["rmse"] = ((q_nn_mean_series / real_es - 1) ** 2).mean(
                axis=0)  # rmse of the series
            dict_nn[criteria]["mean"]["rmse_bestK"] = ((np.array(dict_nn[criteria]["mean"][
                                                                     "q_bestK"]) / real_es - 1) ** 2).mean()  # rmse metric on all the selected  q(k^star)

            # MEDIAN (RMedSE)
            q_nn_median_series = np.array(dict_nn[criteria]["median"]["series"])
            dict_nn[criteria]["median"]["value"] = np.median(
                dict_nn[criteria]["median"]["value"])  # median of all best values in order to compare NN models
            dict_nn[criteria]["median"]["var"] = q_nn_median_series.var(axis=0)  # variance between the replications
            dict_nn[criteria]["median"]["series"] = np.median(q_nn_median_series,
                                                              axis=0)  # median between the replications
            dict_nn[criteria]["median"]["rmse"] = np.median((q_nn_median_series / real_es - 1) ** 2,
                                                            axis=0)  # rmedse between the replications
            dict_nn[criteria]["median"]["rmse_bestK"] = np.median((np.array(dict_nn[criteria]["median"][
                                                                                "q_bestK"]) / real_es - 1) ** 2)  # rmsede metric on all the selected  q(k^star)

        np.save(pathfile, dict_nn)
        # ----------------------
    finally:
        return dict_nn
