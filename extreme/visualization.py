from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
import re
from pathlib import Path
import collections
from utils import load_summary_file
from extreme.data_management import DataSampler, load_quantiles, load_real_data, box_cox
from extreme.estimators import get_gamma_hill, best_gamma_hill, get_gamma_Rhill, random_forest_k, ExtremeBCTM, real_estimators, sim_estimators, hill, TailIndexEstimator
from models import load_model, model_evaluation, model_evaluation_real
from scipy.special import hyp2f1
import scipy.special as sp
import sys

# sns.set_style("whitegrid", {'grid.linestyle': '--'})
#
# sys.path.append("..")

def quantile_plot(distribution, params, n_data, alpha, rep, zeta=0):
    """Quantile plot"""
    fig, axes = plt.subplots(1, 1, figsize=(12, 7), sharex=False, squeeze=False)
    data_sampler = DataSampler(distribution=distribution, params=params)
    data, X_order = data_sampler.load_simulated_data(n_data=n_data, rep=rep)

    i_indices = np.arange(1, int(alpha*n_data))[::-1]
    # x = np.log(k_anchor / i_indices).reshape(-1, 1)
    real_quantiles = [data_sampler.ht_dist.tail_ppf(n_data / _i) for _i in np.arange(1, int(alpha*n_data))[::-1]]  # simulate the real quantile

    # Real function
    # -----------
    plt.plot(1-i_indices/n_data, data_sampler.ht_dist.ppf(1-i_indices/n_data), color='black', label="real function")

    sns.scatterplot(x=1-i_indices/n_data, y=[np.mean(X_order[-(i-1)]) for i in range(int(alpha*n_data), 1, -1) ], marker="o", color='C2', s=50, label="Order Stat")

    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    fig.tight_layout()
    sns.despine()
    plt.legend()
    return


def bctm_plot(distribution, params, n_data, alpha, a, rep, zeta=0.):
    """Box Cox CTM plot"""
    fig, axes = plt.subplots(1, 1, figsize=(12, 7), sharex=False, squeeze=False)
    data_sampler = DataSampler(distribution=distribution, params=params)
    X_order = data_sampler.simulate_quantiles(n_data=n_data, seed=rep)
    k_anchors = np.arange(1 + int(zeta * n_data), int(alpha * n_data))[::-1]

    # Real function
    # -----------
    plt.plot(1 - k_anchors / n_data, data_sampler.ht_dist.box_conditional_tail_moment(k_anchors / n_data, a),
             color='black', label="Real function", linewidth=2)

    # Empirical function
    # -----------
    sns.scatterplot(x=1 - k_anchors / n_data, y=[box_cox(np.mean(X_order[-j:]), a) for j in k_anchors], marker="o",
                    color='C0', s=50, label="Empirical")

    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    fig.tight_layout()
    sns.despine()
    plt.legend()
    return


##################################################
##################################################


def xes_mse_paper_plot(criteria="mad", metric="median", graph="bias", saved=False, source="sim", **model_filenames):
    """extreme ES plot at level 1/2n for different replications"""

    # take the first one to load the config summary and extract data infos
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    colors = plt.cm.rainbow(np.linspace(0, 1, len(model_filenames)))

    _, model_filename = list(model_filenames.items())[0]
    summary_file = load_summary_file(filename=model_filename+"-rep1", source=source)

    # assume all models have the same number of data and replications
    n_data = int(summary_file["n_data"])
    n_replications = int(summary_file["replications"])
    zeta = summary_file["zeta"]

    #pathdir = Path("ckpt", summary_file["distribution"], "extrapolation", str(summary_file["params"]))
    pathdir = Path('/Users', 'michaelallouche', 'PhD', 'repos', 'nn-ES', 'nn-risk-measures', "ckpt", source, summary_file["distribution"], "extrapolation", str(summary_file["params"]))
    pathdir.mkdir(parents=True, exist_ok=True)

    EXTREME_ALPHA = 1/(2*n_data)  # pick the extreme alpha
    anchor_points = np.arange(2 + int(zeta*n_data), n_data)  # 2, ..., n-1

    # real data
    data_sampler = DataSampler(**summary_file)
    # real_quantile = data_sampler.ht_dist.tail_ppf(1/EXTREME_ALPHA)  # real extreme quantile
    real_es = data_sampler.ht_dist.expected_shortfall(1 - EXTREME_ALPHA)

    fig, axes = plt.subplots(1, 1, figsize=(12, 7), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse


    # # EVT ESTIMATORS
    # ---------------
    try:
        print(Path(pathdir, "evt_estimators_rep{}_ndata{}_zeta{}.npy".format(n_replications, n_data, zeta)))
        dict_evt = np.load(Path(pathdir, "evt_estimators_rep{}_ndata{}_zeta{}.npy".format(n_replications, n_data, zeta)), allow_pickle=True)[()]
    except FileNotFoundError:
        print("Training EVT estimators ...")
        dict_evt = evt_estimators(n_replications, n_data, zeta, summary_file["distribution"], summary_file["params"], return_full=True,
                                  metric=metric)
    # ---------------

    for idx_model, (trunc_condition, model_filename) in enumerate(model_filenames.items()):
        pathfile = Path(pathdir, "{}.npy".format(model_filename))
        try:
            dict_nn = np.load(pathfile, allow_pickle=True)[()]
        except FileNotFoundError:
            print("Model Selection ...")
            dict_nn = model_evaluation(model_filename)

        # for replication in range(1, n_replications + 1):
        model_mean = dict_nn[criteria][metric]["series"]
        model_rmse = dict_nn[criteria][metric]["rmse"]  # series for different k
        model_rmse_bestK = dict_nn[criteria][metric]["rmse_bestK"]

        # plot NN
        if graph == "bias":
            axes[0, 0].plot(anchor_points, model_mean,  label="{}: {:.4f}".format(trunc_condition, model_rmse_bestK), color=colors[idx_model])
        elif graph == "rmse":
            axes[0, 0].plot(anchor_points, model_rmse, label="{}: {:.4f}".format(trunc_condition, model_rmse_bestK), color=colors[idx_model])

    # # EVT ESTIMATORS
    # # ---------------
    for estimator in dict_evt.keys():
        if graph == "bias":
            axes[0, 0].plot(anchor_points, dict_evt[estimator][metric]["series"],
                             linestyle="-.")
        elif graph == "rmse":
            axes[0, 0].plot(anchor_points, dict_evt[estimator][metric]["rmse"],
                             linestyle="-.")
    # -----------------

    axes[0, 0].hlines(y=real_es, xmin=0., xmax=n_data,  color="black", linestyle="--")
    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")


    # y_lim
    if graph == "bias":
        axes[0, 0].set_ylim(real_es*0.8, real_es*1.1)  # ES
    elif graph == "rmse":
        axes[0, 0].set_ylim(0, 1)  # RMedSE

    fig.tight_layout()
    #fig.suptitle("Estimator plot \n{}: {}".format(summary_file["distribution"].capitalize(), str(summary_file["params"]).upper()), fontweight="bold", y=1.04)
    sns.despine()
    if saved:
        # plt.savefig("imgs/{}-{}-{}.eps".format(graph, summary_file["distribution"], str(summary_file["params"])), format="eps")
        plt.savefig("imgs/{}-{}.eps".format(graph, model_filename), format="eps")
    return

def xbctm_evt_mse_plot(distribution, params, n_data, zeta, a, risk_level, n_replications, criteria="mad", metric="median", source='sim'):
    """EVT estimators extreme BCTM plot at level 1/2n for different replications"""

    # take the first one to load the config summary and extract data infos
    sns.set_style("whitegrid", {'grid.linestyle': '--'})

    pathdir = Path( "ckpt", source, distribution, str(params))
    pathdir.mkdir(parents=True, exist_ok=True)

    EXTREME_ALPHA = risk_level #1/(2*n_data)  # pick the extreme alpha
    anchor_points = np.arange(2 + int(zeta*n_data), n_data)  # 2, ..., n-1

    # real data
    data_sampler = DataSampler(distribution=distribution, params=params)
    real_bctm = data_sampler.ht_dist.box_conditional_tail_moment(EXTREME_ALPHA, a)
    fig, axes = plt.subplots(2, 1, figsize=(15, 2 * 5), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    # # EVT ESTIMATORS
    # ---------------
    try:
        dict_evt = np.load(Path(pathdir, "sim_estimators_rep{}_ndata{}_rlevel{}_zeta{}_a{}.npy".format(n_replications, n_data, risk_level, zeta, a)), allow_pickle=True)[()]
    except FileNotFoundError:
        print("Training EVT estimators ...")
        dict_evt = sim_estimators(n_replications=n_replications, params=params,
                           distribution=distribution, n_data=n_data, risk_level=risk_level, a=a, zeta=zeta, metric="median")

    # ---------------
    # # EVT ESTIMATORS
    # # ---------------
    for estimator in dict_evt.keys():
        axes[0, 0].plot(anchor_points, dict_evt[estimator][metric]["series"],
                        label="{}: {:.4f}".format(estimator, dict_evt[estimator][metric]["rmse_bestK"]), linestyle="-.")
        axes[1, 0].plot(anchor_points, dict_evt[estimator][metric]["rmse"],
                        label="{}: {:.4f}".format(estimator, dict_evt[estimator][metric]["rmse_bestK"]), linestyle="-.")
    # -----------------

    axes[0, 0].hlines(y=real_bctm, xmin=0., xmax=n_data, label="reference line", color="black", linestyle="--")
    axes[0, 0].legend(fontsize=15)
    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    # title / axis
    axes[0, 0].set_xlabel(r"anchor point $k$")
    axes[0, 0].set_ylabel("quantile")
    axes[0, 0].set_title("Median estimator", size=15)

    axes[1, 0].set_xlabel(r"anchor point $k$")
    axes[1, 0].set_ylabel("RMedSE")
    axes[1, 0].set_title("RMedSE", size=15)
    axes[1, 0].spines["left"].set_color("black")
    axes[1, 0].spines["bottom"].set_color("black")

    # y_lim
    axes[0, 0].set_ylim(real_bctm*0.5, real_bctm*2)  # bias
    axes[1, 0].set_ylim(0, 1)  # RMedSE

    fig.tight_layout()
    fig.suptitle("Estimator plot \n{}: {}, a={}, alpha={}".format(distribution.capitalize(), str(params), a, risk_level, fontweight="bold"), size=20, y=1.1)
    sns.despine()
    return

def xbctm_evt_mse_paper_plot(distribution, params, n_data, zeta, a, risk_level, n_replications, graph="bias", saved=False, criteria="mad", metric="median", source='sim'):
    """EVT estimators extreme ES plot at level 1/2n for different replications (no NN estimator)"""

    # take the first one to load the config summary and extract data infos
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(model_filenames)))

    # _, model_filename = list(model_filenames.items())[0]
    # summary_file = load_summary_file(filename=model_filename+"-rep1")

    # assume all models have the same number of data and replications
    # n_data = summary_file["n_data"]
    # n_replications = int(summary_file["replications"])
    # zeta = summary_file["zeta"]

    #pathdir = Path("ckpt", summary_file["distribution"], "extrapolation", str(summary_file["params"]))
    pathdir = Path('/Users', 'michaelallouche', 'PhD', 'repos', 'bctm', "ckpt", source, distribution, str(params))
    pathdir.mkdir(parents=True, exist_ok=True)

    EXTREME_ALPHA = risk_level #1/(2*n_data)  # pick the extreme alpha
    anchor_points = np.arange(2 + int(zeta*n_data), n_data)  # 2, ..., n-1

    # real data
    data_sampler = DataSampler(distribution=distribution, params=params)
    real_bctm = data_sampler.ht_dist.box_conditional_tail_moment(EXTREME_ALPHA, a)
    fig, axes = plt.subplots(1, 1, figsize=(12, 7), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    # # EVT ESTIMATORS
    # ---------------
    try:
        dict_evt = np.load(Path(pathdir, "sim_estimators_rep{}_ndata{}_rlevel{}_zeta{}_a{}.npy".format(n_replications, n_data, risk_level, zeta, a)), allow_pickle=True)[()]
    except FileNotFoundError:
        print("Training EVT estimators ...")
        dict_evt = sim_estimators(n_replications=n_replications, params=params,
                           distribution=distribution, n_data=n_data, risk_level=risk_level, a=a, zeta=zeta, metric="median")

    # ---------------
    # # EVT ESTIMATORS
    # # ---------------
    for estimator in dict_evt.keys():
        if graph=="bias":
            bias_values = dict_evt[estimator][metric]["series"]
            bias_values[bias_values<0] = np.nan
            axes[0, 0].plot(anchor_points, bias_values,
                            label="{}: {:.4f}".format(estimator, dict_evt[estimator][metric]["rmse_bestK"]), linestyle="-.", linewidth=2)
        elif graph == "rmedse":
            axes[0, 0].plot(anchor_points, dict_evt[estimator][metric]["rmse"],
                            label="{}: {:.4f}".format(estimator, dict_evt[estimator][metric]["rmse_bestK"]), linestyle="-.", linewidth=2)
    # -----------------

    axes[0, 0].hlines(y=real_bctm, xmin=0., xmax=n_data, label="reference line", color="black", linestyle="--", linewidth=2)
    # axes[0, 0].legend(fontsize=15)
    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")


    # y_lim
    if graph == "bias":
        axes[0, 0].set_ylim(real_bctm*0.5, real_bctm*1.69)
    elif graph=="rmedse":
        axes[0, 0].set_ylim(0, 0.1)  # RMedSE

    fig.tight_layout()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # fig.suptitle("Estimator plot \n{}: {}, a={}, alpha={}".format(distribution.capitalize(), str(params), a, risk_level, fontweight="bold"), size=20, y=1.1)
    sns.despine()
    if saved:
        plt.savefig("imgs/sim/{}-{}-{}-{}.eps".format(graph, distribution, list(params.values())[0], a), format="eps")
    return



# =====================================
#              REAL PLOTS
# =====================================

def real_hill_plot(saved=False):
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    # X = pd.read_csv(Path(os.getcwd(), 'data', "besecura.txt"), sep='\t').loc[:, 'Loss'].to_numpy()  # read data
    X = pd.read_csv("data/real/norwegian90.csv")
    # X_order = np.sort(X)
    # print(X.to_numpy())
    n_data = len(X)
    # EXTREME_ALPHA = 1 / n_data
    # evt_estimators = ExtremeQuantileEstimator(X=X_order, alpha=EXTREME_ALPHA)
    anchor_points = np.arange(2, n_data)  # 2, ..., n-1

    hill_gammas = [hill(X.to_numpy(), k_anchor) for k_anchor in anchor_points]

    # k_prime = evt_estimators.get_kprime_rw(n_data-1)[0]
    # anchor_points_prime = np.arange(2, int(k_prime)+1)
    # hill_gammas_prime = [evt_estimators.hill(k_anchor) for k_anchor in anchor_points_prime]
    bestK = random_forest_k(np.array(hill_gammas), n_forests=10000, seed=42)

    axes[0, 0].plot(anchor_points, hill_gammas, color="black")
    axes[0, 0].scatter(bestK, hill_gammas[bestK], s=200, color="red", marker="^", edgecolor="black")
    print("\hat\gamma(k^\star={})={}".format(bestK+anchor_points[0], hill_gammas[bestK]))
    # axes[0, 0].plot(anchor_points_prime, hill_gammas_prime, color="red")


    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    fig.tight_layout()
    sns.despine()

    if saved:
        pathdir = Path("imgs/real")
        pathdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(pathdir / "hill_plot_real.eps", format="eps")
    return hill_gammas[bestK]



def real_loglog_plot(percentile, gamma_hill=None, saved=False):
    fig, ax = plt.subplots(figsize=(15, 7))

    X = pd.read_csv("data/real/norwegian90.csv")

    n_data = X.shape[0]

    order_stats = np.sort(X.fire)
    quantile_levels = np.linspace(1, n_data, n_data) / (n_data+1)
    extreme_data = np.round((1-percentile) * n_data).astype(int)
    x = -np.log(1 - quantile_levels)[-extreme_data:]
    y = np.log(order_stats)[-extreme_data:]

    par = np.polyfit(x, y, 1, full=True)
    intercept = par[0][1]

    if gamma_hill is None:
        slope = par[0][0]
    else:
        slope = gamma_hill

    # sns.regplot(x=x, y=y,
    #             ci=None, marker="o",
    #             #label=r"$\gamma$({}): {:.3f}".format(year, slope), ci=None, marker="o",
    #             scatter_kws={'s':100, "edgecolor":"black"}, color = 'red')
    
    plt.scatter(x, y, s=100, edgecolor="black")
    plt.plot(x, x*slope + intercept, color="red", linewidth=2)




    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    # plt.legend() # bbox_to_anchor=(1.02, 1), , borderaxespad=0, loc='bottom left'
    # plt.setp(ax.get_legend().get_texts(), fontsize='20') # for legend text
    print("$\hat\gamma(k^\star)=$", slope)
    if saved:
        plt.savefig("imgs/real/loglog.eps", format="eps")
    return



def xes_real_bias_plot(a, xi, zeta=0, metric="median", saved=False):
    """Bias plot estimators extreme ES plot real data at level 1/n"""

    # take the first one to load the config summary and extract data infos
    sns.set_style("whitegrid", {'grid.linestyle': '--'})

    X = pd.read_csv("data/real/norwegian90.csv")
    n = X.shape[0]
    Xtrain = X[:int(np.floor(xi * n))].to_numpy()
    Xtest = X[-int(np.ceil((1 - xi) * n)):].to_numpy()
    n_train = Xtrain.shape[0]
    n_test = Xtest.shape[0]

    anchor_points = np.arange(2 + int(zeta*n_train), n_train)  # 2, ..., n-1
    EXTREME_ALPHA = 1 / (n_train)  # extreme alpha
    real_bctm = np.mean(box_cox(Xtest, a))

    fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    # # EVT ESTIMATORS
    # ---------------
    dict_evt = real_estimators(a=a, xi=xi, zeta=zeta, return_full=True)
    # ---------------

    # # EVT ESTIMATORS
    # # ---------------
    for estimator in dict_evt.keys():
        es_evt = dict_evt[estimator][metric]["series"]
        bestK = dict_evt[estimator]["median"]["bestK"][0]
        bctm_bestK = np.round(dict_evt[estimator]["median"]["q_bestK"], 4)
        axes[0, 0].plot(anchor_points, es_evt,
                        label="{}: {}".format(estimator, bctm_bestK), linestyle="-.")
        axes[0, 0].scatter(int(bestK), bctm_bestK, s=200, marker="^", edgecolor="black")
    # -----------------

# # -----------------
    print(real_bctm)
    axes[0, 0].hlines(y=real_bctm, xmin=0., xmax=n_train, label="reference line", color="black", linestyle="--")
#     # axes[0, 0].legend()
#     axes[0, 0].spines["left"].set_color("black")
#     axes[0, 0].spines["bottom"].set_color("black")

    # title / axis
    # axes[0, 0].set_xlabel(r"anchor point $k$")
    # axes[0, 0].set_ylabel("ES")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # y_lim
    axes[0, 0].set_ylim(real_bctm*0.5, real_bctm*2)
    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    fig.tight_layout()
    # plt.title("Bias plot \n{}".format(year), size=20)
    sns.despine()

    plt.legend()

    if saved:
        plt.savefig("imgs/real/bias-a{}_xi{}_zeta{}.eps".format(a, xi, zeta), format="eps")
    return

def xes_real_cte_half(xi, zeta=0, gamma_estimator="hill", metric="median"):
    """Bias plot estimators extreme ES plot real data at level 1/n"""
    a = 0.5
    X = pd.read_csv("data/real/norwegian90.csv")
    n = X.shape[0]


    Xtrain = X[:int(np.floor(xi * n))].to_numpy()
    Xtest = X[-int(np.ceil((1 - xi) * n)):].to_numpy()
    n_train = Xtrain.shape[0]
    n_test = Xtest.shape[0]
    # tail_estimator = TailIndexEstimator(X.to_numpy())
    tail_estimator = TailIndexEstimator(Xtrain)

    # anchor_points = np.arange(2, n)  # 2, ..., n-1
    anchor_points = np.arange(2, n_train)  # 2, ..., n-1

    if gamma_estimator == "hill":
        gammas = [tail_estimator.hill(k_anchor) for k_anchor in anchor_points]
    elif gamma_estimator == "hill_RB":
        gammas = [tail_estimator.corrected_hill(k_anchor) for k_anchor in anchor_points]

    bestK = random_forest_k(np.array(gammas), n_forests=10000, seed=42)
    gamma = gammas[bestK]
    print("Gamma estimation:", gamma)

    real_cte = np.mean(Xtest)

    # # EVT ESTIMATORS
    # ---------------
    dict_evt = real_estimators(a=a, xi=xi, zeta=zeta, return_full=True)
    # ---------------

    # # EVT ESTIMATORS
    # # ---------------
    dict_cte = {}
    for estimator in dict_evt.keys():
        bctm_bestK = dict_evt[estimator][metric]["q_bestK"]
        # print(bctm_bestK)
        dict_cte[estimator] = ((1-gamma/2) * (1 + np.array(bctm_bestK)/2))**2 / (1-gamma)
    dict_cte["emp"] = real_cte
    return dict_cte

# ==================================================





def scatter_real(criteria="mad", metric="median", saved=False, **model_filenames):
    """scatter of ES estimations: {year: model_filename}"""
    fig, ax = plt.subplots(figsize=(15, 7))  # 3 plots: quantile, var, mse

    dict_scatter = collections.defaultdict(list)
    # take the first one to load the config summary and extract data infos
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    pathdir = Path('/Users', 'michaelallouche', 'PhD', 'repos', 'nn-ES', 'nn-risk-measures', "ckpt", "real")
    pathdir.mkdir(parents=True, exist_ok=True)
    for idx_model, (_, model_filename) in enumerate(model_filenames.items()):
        summary_file = load_summary_file(model_filename, "real")
        year = summary_file["year"]
        xi = summary_file["xi"]
        zeta = summary_file["zeta"]
        percentile = summary_file["percentile"]
        dict_scatter["year"].append(year)
        X_train, order_train, order_test = load_real_data(year=year, xi=xi)
        n_train = order_train.shape[0]
        real_es = np.mean(order_test)
        # dict_scatter["Empirical"].append(real_es)
        dict_scatter["values"].append(real_es)
        dict_scatter["method"].append("empirical")
        pathfile = Path(pathdir, "{}.npy".format(model_filename))

        try:
            dict_nn = np.load(pathfile, allow_pickle=True)[()]
        except FileNotFoundError:
            print("Model Selection ...")
            dict_nn = model_evaluation_real(model_filename)
        model_es_bestK = np.round(dict_nn[criteria][metric]["q_bestK"], 4)
        # dict_scatter["NN"].append(model_es_bestK)
        dict_scatter["year"].append(year)
        dict_scatter["values"].append(model_es_bestK[0])
        dict_scatter["method"].append("NN")

        # Other Estimators
        # ----------------
        dict_evt = real_estimators(year=year, xi=xi, zeta=zeta, percentile=percentile, return_full=True)
        for estimator in dict_evt.keys():
            es_bestK = np.round(dict_evt[estimator]["median"]["q_bestK"], 4)
            # dict_scatter[estimator].append(es_bestK)
            dict_scatter["year"].append(year)
            dict_scatter["values"].append(es_bestK[0])
            dict_scatter["method"].append(estimator)

    df_scatter = pd.DataFrame(dict_scatter)
    palette = {'empirical': 'black', 'NN': plt.cm.rainbow(np.linspace(0, 1, len(model_filenames)))[0], "D": "C0", "D_CH": "C1", "I": "C2", "I_CH": "red", "I_CW": "plum"}
    b = sns.scatterplot(data=df_scatter, x="year", y="values", hue="method", s=200, edgecolor="black", palette=palette, legend=False)
    plt.ylim(0, 1)
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    # plt.setp(ax.get_legend().get_texts(), fontsize='22')  # for legend text
    # plt.setp(ax.get_legend().get_title(), fontsize='32')  # for legend title
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    b.set_xlabel("")
    b.set_ylabel("")
    sns.despine()
    if saved:
        plt.savefig("imgs/real/bias-scatter.eps", format="eps")
    return
