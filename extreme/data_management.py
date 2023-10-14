from .distributions import Burr, Frechet, InverseGamma, Fisher, GPD, Student, Pareto
import numpy as np
from pathlib import Path
import itertools
import pandas as pd

dict_distributions = {"burr": Burr, "invgamma": InverseGamma, "frechet": Frechet, "fisher": Fisher, "gpd": GPD,
                      "student": Student,  "pareto": Pareto}

def load_distribution(name_distribution):
    """load a distribution"""
    return dict_distributions[name_distribution]

def load_quantiles(distribution, params, n_data, rep=32, **kwargs):
    file = Path("data", distribution, "Xorder_{}_{}_ndata{}-rep{}.npy".format(distribution, params, n_data, rep))
    try:
        if file.is_file():  # load if exists
            return np.load(file, allow_pickle=True)
        else:  # else simulate them
            data_sampler = DataSampler(distribution, params)
            return data_sampler.simulate_quantiles(n_data, seed=rep, random=True).reshape(-1, 1)
    except OSError:  # if file not properly saved, delete it and save it again
        print("file", file, " removed")
        file.unlink()  # remove the file
        data_sampler = DataSampler(distribution, params)
        return data_sampler.simulate_quantiles(n_data, seed=rep, random=True).reshape(-1, 1)



def load_real_data(year, xi, zeta=0, percentile=0, saved=True):
    pathdir = Path(f"data/real/{year}")
    pathfile_train = Path(f"traindata_xi{xi}_zeta{zeta}_perc{percentile}.npy")
    pathfile_test = Path(f"testdata_xi{xi}_zeta{zeta}_perc{percentile}.npy")
    data_filename = f"xyz_xi{xi}_zeta{zeta}_perc{percentile}.npz"
    if ((pathdir/pathfile_train).exists()) and ((pathdir/pathfile_test).exists()) and ((pathdir/data_filename).exists()):
        # print("Loading...")
        data = np.load(pathdir/data_filename)
        order_train = np.load(pathdir/pathfile_train)
        order_test = np.load(pathdir/pathfile_test)
        return np.concatenate([data["x1"], data["x2"], data["x3"], data["x4"], data["y"]], axis=1), order_train, order_test
    else:
        # print("Building...")
        Path(pathdir).mkdir(exist_ok=True, parents=True)
        return build_real_data(year, xi, zeta, percentile, saved)

def build_real_data(year, xi, zeta=0, percentile=0, saved=True):
    df_real = pd.read_csv("data/real/btc-usd-2015-2023.csv")
    pathdir = Path(f"data/real/{year}")
    pathdir.mkdir(exist_ok=True, parents=True)
    pathfile_train = Path(f"traindata_xi{xi}_zeta{zeta}_perc{percentile}.npy")
    pathfile_test = Path(f"testdata_xi{xi}_zeta{zeta}_perc{percentile}.npy")
    data_filename = f"xyz_xi{xi}_zeta{zeta}_perc{percentile}.npz"
    # -----------
    df_real["returns"] = np.log(df_real.price) - np.log(df_real.price.shift(1))
    df_real.timestamp = pd.to_datetime(df_real.timestamp)
    df_real.dropna(inplace=True)
    df_stats = df_real[df_real["returns"] <= 0]
    df_stats.loc[:, "returns"] = -df_stats["returns"]
    df_stats.loc[:, "year"] = df_stats.apply(lambda x: x.timestamp.year, axis=1)
    df_data = df_stats[df_stats["year"] == year]
    n_data = df_data.shape[0]

    trainset = np.sort(df_data["returns"]).astype("float32")[:-int((1 - xi) * n_data)].reshape(-1,1)
    testset = np.sort(df_data["returns"]).astype("float32")[-int((1 - xi) * n_data):].reshape(-1,1)
    # -----------
    list_indices = []
    threshold_K = int(n_data - (n_data * percentile)) - 1  # threshold K < n (integer)
    for k in range(2 + int(zeta * n_data), threshold_K + 1):
        for i in range(1 + int(zeta * n_data), k):
            list_indices.append([i, k])

    i_idx = np.array(list_indices)[:, 0]
    k_idx = np.array(list_indices)[:, 1]

    def log_spacings(idx):
        """log X_{n-i+1, n} - log X_{n-k+1, n}"""
        return np.log(trainset[-idx[0]:].mean()) - np.log(trainset[-idx[1]:].mean())

    y = np.apply_along_axis(log_spacings, axis=1, arr=list_indices).reshape(-1, 1)
    x1 = np.float32(np.log(k_idx / i_idx).reshape(-1, 1))
    x2 = np.float32(np.log(n_data / k_idx).reshape(-1, 1))
    x3 = np.float32((i_idx / n_data).reshape(-1, 1))
    x4 = np.float32((k_idx / n_data).reshape(-1, 1))

    if saved:
        np.save(pathdir / pathfile_train, trainset)
        np.save(pathdir / pathfile_test, testset)
        np.savez(pathdir / data_filename, x1=x1, x2=x2, x3=x3, x4=x4, y=y)
    return np.concatenate([x1, x2, x3, x4, y], axis=1), trainset, testset



class DataSampler():
    def __init__(self, distribution, params, percentile=0, **kwargs):
        self.distribution = distribution
        self.params = params
        self.ht_dist = load_distribution(distribution)(**params)  # heavy-tailed distribution
        self.percentile = percentile

        self.pathdir_data = Path("data", self.distribution)
        if not self.pathdir_data.is_dir():
            self.pathdir_data.mkdir(parents=True)

        return

    def load_simulated_data(self, n_data, rep=0, zeta=0., saved=False):
        """
        Simulate dataset (x,y,z) and the associated order statistics (quantiles)

        Parameters
        ----------
        n_data :int
            number of simulations
        rep :  int
            replication to fix the seed
        saved : str
            if True, save the the data

        Returns
        -------
        ndarray, ndarray
            [x1, x2, y], X
        """
        threshold_K = int(n_data - (n_data * self.percentile)) - 1  # threshold K < n (integer)

        data_filename = "xyz_{}_{}_ndata{}_zeta{}-rep{}.npz".format(self.distribution, self.params, n_data, zeta, rep)
        Xorder_filename = "Xorder_{}_{}_ndata{}-rep{}.npy".format(self.distribution, self.params, n_data, rep)

        pathfile_data = Path(self.pathdir_data , data_filename)
        pathfile_Xorder = Path(self.pathdir_data , Xorder_filename)

        if pathfile_data.is_file() and pathfile_Xorder.is_file():  # if file exists, load existing data
            data = np.load(pathfile_data)
            X_order = np.load(pathfile_Xorder)
            return np.concatenate([data["x1"], data["x2"], data["x3"], data["x4"], data["y"]], axis=1), X_order
        else:
            Path(self.pathdir_data).mkdir(exist_ok=True)
            return self.build_simulated_data(n_data, threshold_K, rep, saved, pathfile_data, pathfile_Xorder, zeta=zeta)

    def build_simulated_data(self, n_data, threshold_K, rep=0, saved=False, pathfile_data=None, pathfile_Xorder=None, zeta=0):
        """Compute ES-log spacings"""
        X_order = self.simulate_quantiles(n_data, seed=rep, random=True).reshape(-1,1)  # X_{1,n}, ..., X_{n,n} with u~U(0,1)

        list_indices = []
        for k in range(2+int(zeta*n_data), threshold_K+1):
            for i in range(1+int(zeta*n_data), k):
                list_indices.append([i,k])

        i_idx = np.array(list_indices)[:, 0]
        k_idx = np.array(list_indices)[:, 1]

        def log_spacings(idx):
            """log X_{n-i+1, n} - log X_{n-k+1, n}"""
            return np.log(X_order[-idx[0]:].mean()) - np.log(X_order[-idx[1]:].mean())

        y = np.apply_along_axis(log_spacings, axis=1, arr=list_indices).reshape(-1,1)
        x1 = np.float32(np.log(k_idx / i_idx).reshape(-1, 1))
        x2 = np.float32(np.log(n_data / k_idx).reshape(-1, 1))
        x3 = np.float32((i_idx / n_data).reshape(-1, 1))
        x4 = np.float32((k_idx / n_data).reshape(-1, 1))

        if saved:
            np.save(pathfile_Xorder, X_order)
            np.savez(pathfile_data, x1=x1, x2=x2, x3=x3, x4=x4, y=y)
        return np.concatenate([x1, x2, x3, x4, y], axis=1), X_order

    def simulate_quantiles(self, n_data, low_bound=0., high_bound=1., random=True, seed=32, **kwargs):
        """
        simulate from quantile function  q
        quantiles(random=False) or order statistics (random=True) from heavy-tailed distribution
        Parameters
        ----------
        n_data :
        low_bound :
        up_bound :
        random : bool
            if true: drawn u values from a uniform distribution, else from a linear grid
        kwargs :

        Returns
        -------

        """
        if random:
            np.random.seed(seed)
            u_values = np.random.uniform(low_bound, high_bound, size=(int(n_data), 1))  # sample from U( [0, 1) )
            quantiles = np.float32(self.ht_dist.ppf(u_values))
            return np.sort(quantiles, axis=0)  # sort the order statistics
        else:
            u_values = np.linspace(low_bound, high_bound, int(n_data)).reshape(-1,1)  # endpoint=False
            return np.float32(self.ht_dist.ppf(u_values))

    def simulate_bctm(self, n_data, a, seed):
        X_simulated = self.simulate_quantiles(n_data=n_data, rep=seed)
        empirical_bctm = box_cox(X_simulated, a)
        return empirical_bctm.ravel()


    @staticmethod
    def split(arr, cond):
        """split an array given a condition"""
        return [arr[cond], arr[~cond]]

def box_cox(x, a):
    if a > 0:
        return (x**a - 1) / a
    elif a == 0:
        return np.log(x)
    else:
        raise ValueError("Please enter a positive value for a.")





