import pandas as pd
import scipy.stats
from rpy2 import robjects as ro
import rpy2.robjects.numpy2ri

from extreme.data_management import load_quantiles, DataSampler, load_real_data, box_cox
import numpy as np
from pathlib import Path

list_estimators = ["D", "D_GRB","D_RB", "I", "I_RB"]

# ==================================================
#                  Tail index estimator
# ==================================================

def hill(X, k_anchor):
    """

    Parameters
    ----------
    X : ndarray
        order statistics
    k : threshold
        int
        greater than 1

    Returns
    -------

    """
    X_in = X[-k_anchor:]
    X_kn = X[-(k_anchor+1)] * np.ones_like(X_in)

    return np.mean(np.log(X_in) - np.log(X_kn))

def get_gamma_hill(X):
    anchor_points = np.arange(2, X.shape[0])  # 2, ..., n-1
    hill_gammas = [hill(np.sort(X), k_anchor) for k_anchor in anchor_points]
    return hill_gammas

def best_gamma_hill(hill_gammas, n_forests=10000):
    bestK = random_forest_k(np.array(hill_gammas), n_forests=n_forests, seed=42)
    return hill_gammas[bestK-1]

def get_gammaRB(X_order, k):
    tail_estimator = TailIndexEstimator(X_order)
    return tail_estimator.corrected_hill(k)


def get_gamma_Rhill(hill_gammas, rho, n_data):
    EXTREME_ALPHA = 1 / n_data
    k_prime = get_kprime_rw(n_data-1, rho, EXTREME_ALPHA, n_data)
    return hill_gammas[:(int(k_prime)+1)]
    

def get_kprime_rw(k_anchor, rho, alpha, n_data):
    """
    Compute the intermediate sequence to plug in the Hill estimator.

    Parameters
    ----------
    k_anchor : int
        Intermediate sequence of the quantile estimator

    Returns
    -------
    k: int
    """
    extrapolation_ratio = k_anchor / (alpha * n_data)
    k_prime = k_anchor * np.power((-rho * np.log(extrapolation_ratio)) / ((1-rho) * (1 - np.power(extrapolation_ratio, rho))), 1/rho)
    return k_prime


class TailIndexEstimator():
    def __init__(self, X_order):
        """
        Tail index estimators

        The class contains:
        - Hill (H) [1]
        - Corrected Hill (CH) [2]
        - (H_p) [3]
        - (CH_p) [4]
        - (CH_{p^star}) [5]
        - (PRB_P) [6]
        - (PRB_{p^star}) [7]

        Parameters
        ----------
        X_order : ndarray
            Order statistics X_{1,n} \leq ... \leq X_{n,n}

        References
        ----------

        Examples
        --------
        """
        # R settings
        rpy2.robjects.numpy2ri.activate()
        r = ro.r
        r['source']('extreme/revt.R')
        self.get_rho_beta = r.get_rho_beta
        # self.l_run = r.lrun

        self.X_order = X_order
        self.n_data = X_order.shape[0]

        self.rho, self.beta = self.get_rho_beta(X_order)  # rho and beta estimated
        self.varphi = 1 - self.rho/2 - np.sqrt(np.square(1 - self.rho/2) - 0.5)
        self.k0 = self.get_k0()
        self.p_star = self.varphi / self.corrected_hill(self.k0)
        return

    def hill(self, k_anchor):
        """
        Hill estimator

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        gamma: float
            Tail index estimator
        """
        X_in = self.X_order[-k_anchor:]
        X_kn = self.X_order[-(k_anchor + 1)] * np.ones_like(X_in)  # take also the last point X_n,n
        return np.mean(np.log(X_in) - np.log(X_kn))

    def corrected_hill(self, k_anchor):
        """
        Corrected Hill estimator

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        gamma: float
            Tail index estimator
        """
        gamma_hill = self.hill(k_anchor)
        return gamma_hill * (1 - (self.beta / (1 - self.rho) * np.power(self.n_data / k_anchor, self.rho)))

    def hill_p(self, k_anchor, p):
        """
        Redcued-bias H_p

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        p: float
            Tuning parameter

        Returns
        -------
        gamma: float
            Tail index estimator
        """
        if p == 0.:
            gamma = self.hill(k_anchor)
        else:
            X_in = self.X_order[-k_anchor:]
            X_kn = self.X_order[-(k_anchor + 1)] * np.ones_like(X_in)
            gamma = (1 - np.power(np.mean(np.power(X_in / X_kn, p)), -1)) / p
        return gamma


    def corrected_hill_p(self, k_anchor, p=None):
        """
        Reduced-bias mean of order (CH_p)

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        gamma: float
            Tail index estimator
        """
        if p is None:
            p = self.p_CH
        gamma = self.hill_p(k_anchor, p)
        return gamma * (1 - ((self.beta * (1 - p * gamma)) / (1 - self.rho - p * gamma) * np.power(self.n_data / k_anchor, self.rho)))

    def corrected_hill_ps(self, k_anchor):
        """
        Corrected Hill estimator with p^*

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        gamma: float
            Tail index estimator
        """
        gamma = self.hill_p(k_anchor, self.p_star)
        return gamma * (1 - ((self.beta * (1 - self.p_star * gamma)) / (1 - self.rho - self.p_star * gamma) * np.power(self.n_data / k_anchor, self.rho)))

    def partially_reduced_bias_p(self, k_anchor, p=None):
        """
        Partially reduced bias estimator

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        p: float or None (default None)
            Tuning parameter

        Returns
        -------
        gamma: float
            PRB_p estimator
        """
        if p is None:
            p = self.p_PRB
        gamma = self.hill_p(k_anchor, p)
        return gamma * (1 - ((self.beta * (1 - self.varphi)) / (1 - self.rho - self.varphi) * np.power(self.n_data / k_anchor, self.rho)))

    def partially_reduced_bias_ps(self, k_anchor):
        """
        Partially reduced bias estimator with optimal p^*

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        gamma: float
            Tail index estimator
        """
        gamma = self.hill_p(k_anchor, self.p_star)
        return gamma * (1 - ((self.beta * (1 - self.varphi)) / (1 - self.rho - self.varphi) * np.power(self.n_data / k_anchor, self.rho)))

    def get_k0(self):
        """
        Estimated best intermediate sequence to choose the optimal value of p in PRB_{p^\star} and PRB_{p^\star}
        Returns
        -------

        """
        term1 = self.n_data - 1
        term2 = np.power(np.square(1 - self.rho) * np.power(self.n_data, -2*self.rho) / (-2*self.rho*np.square(self.beta)), 1/(1-2*self.rho))
        return int(np.minimum(term1, np.floor(term2) + 1))


# ==================================================
#                  Extreme quantile estimator
# ==================================================


def weissman(X_order, alpha, k_anchor):
    """
    Parameters
    ----------
    X_orders : order statistics
    alpha : extreme order
    k_anchor : anchor point

    Returns
    -------

    Maths
    ----
    X_{n-k, n}(k/np)^gamma_hill(k) with 0<p<1 and k\in{1,...,n-1}

    """
    gamma_hill = hill(X_order, k_anchor)
    n_data = X_order.shape[0]
    X_anchor = X_order[-k_anchor]
    return X_anchor * np.power(k_anchor/(alpha * n_data), gamma_hill)


class ExtremeQuantileEstimator(TailIndexEstimator):
    def __init__(self, X, alpha):
        """
        Extreme quantile estimators

        The class contains:
        - Weissman (H) [1]
        - Refined Weissman (RW) [2]
        - Corrected Weissman (CW) [3]
        - (CH) [4]
        - (CH_{p^star}) [5]
        - (PRB_P) [6]
        - (PRB_{p^star}) [7]

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Data X_1, ..., X_n
        alpha : float
            extreme quantile level
        """
        super(ExtremeQuantileEstimator, self).__init__(X)
        self.alpha = alpha
        self.dict_q_estimators = {"W": self.weissman, "RW": self.r_weissman, "CW": self.c_weissman,
                                  "CH": self.ch_weissman, "CHps": self.chps_weissman,
                                   "PRBps": self.prbps_weissman}
        self.dict_qp_estimators = {"CHp": self.chp_weissman, "PRBp": self.prbp_weissman}
        self.dict_quantile_estimators = {**self.dict_q_estimators, **self.dict_qp_estimators}

        self.p_CH = self.get_p(method="CHp")
        self.p_PRB = self.get_p(method="PRBp")
        return

    def weissman(self, k_anchor):
        """
        Weissman estimator (W)
        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        Quantile estimator: float
        """
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.hill(k_anchor))

    def r_weissman(self, k_anchor):
        """Refined Weissman (RW)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        k_prime = k_anchor * np.power((-self.rho * np.log(extrapolation_ratio)) / ((1-self.rho) * (1 - np.power(extrapolation_ratio, self.rho))), 1/self.rho)
        return X_anchor * np.power(extrapolation_ratio, self.hill(int(np.ceil(k_prime))))

    def c_weissman(self, k_anchor):
        """Corrected Weissman (CW)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio *
                                        np.exp(self.beta * np.power(self.n_data/k_anchor, self.rho)
                                               * (np.power(extrapolation_ratio, self.rho) - 1) / self.rho), self.corrected_hill(k_anchor))

    def ch_weissman(self, k_anchor):
        """Corrected-Hill Weissman (CH)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.corrected_hill(k_anchor))

    def chp_weissman(self, k_anchor, p=None):
        """Corrected-Hill with Mean-of-order-p Weissman (CHp)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.corrected_hill_p(k_anchor, p))

    def chps_weissman(self, k_anchor):
        """Corrected-Hill with Mean-of-order-p star (optimal) Weissman (CHps)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.corrected_hill_ps(k_anchor))

    def prbp_weissman(self, k_anchor, p=None):
        """Partially Reduced-Bias mean-of-order-p Weissman (PRBp)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.partially_reduced_bias_p(k_anchor, p))

    def prbps_weissman(self, k_anchor):
        """Partially Reduced-Bias mean-of-order-p star (optimal) Weissman (PRBPs)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.partially_reduced_bias_ps(k_anchor))

    def quantile_estimator(self, method, k_anchor):
        return self.dict_quantile_estimators[method](k_anchor)

    def get_k(self, x):
        """
        best k based on Algo 1 from Gomes, 2018
        Parameters
        ----------
        x : ndarray
            estimator (gamma or quantiles)

        Returns
        -------

        """
        x = np.log(x)  # convert to log as proposed in the article
        #STEP 2
        list_runsize = []
        k_minmax_list = []
        j = 0
        optimal=False
        while not optimal:
            x_rounded = np.around(x, j)
            if np.unique(x_rounded).shape[0] == x_rounded.shape[0]:  # if all uniques
                optimal = True
            else:
                j += 1
        # STEP 3
        k_min, k_max = self.run_size(x, j)
        k_minmax_list.append((k_min, k_max))
        list_runsize.append(k_max - k_min)
        largest_k_min, largest_kmax = k_minmax_list[np.argmax(list_runsize)]
        # STEP 4
        selected_x = x[int(largest_k_min) : int(largest_kmax+1)]
        new_q_rounded = np.around(selected_x, j+1)
        K_T = np.where(new_q_rounded == scipy.stats.mode(new_q_rounded)[0])
        # STEP 4.2
        bestK = int(np.median(K_T)+largest_k_min)  # \geq 0
        return bestK


    def get_p(self, method):
        """
        get best p and k based on Algo 2 from Gomes, 2018
        Parameters
        ----------
        method :

        Returns
        -------

        """
        # STEP 1
        xi_star = self.corrected_hill(self.k0)
        p_ell = np.arange(16)/(16*xi_star)

        #STEP 2
        list_runsize = []
        for ell in range(16):
            # STEP 2.1:  applied on log(.) as suggested in the paper
            quantiles = np.log([self.dict_qp_estimators[method](k_anchor=k_anchor, p=p_ell[ell])[0] for k_anchor in range(2, self.n_data)])
            # STEP 2.2: find the minimum j>=0 s.t all q_rounded are distinct
            j = 0
            optimal = False
            while not optimal:
                q_rounded = np.around(quantiles, j)
                if np.unique(q_rounded).shape[0] == q_rounded.shape[0]:  # if all uniques
                    optimal = True
                else:
                    j += 1
                if j > 15:  # exit
                    optimal = True

            # STEP 2.3
            k_min, k_max = self.longest_run(q_rounded, j)
            list_runsize.append(k_max - k_min)
        largest_runsize_idx = np.argmax(list_runsize)
        # STEP 3
        p = largest_runsize_idx / (16*xi_star)
        return p[0]

    # 
    
    @staticmethod
    def longest_run(x, j):
        """
        Compute the run size k_min and k_max

        Parameters
        ----------
        x : ndarray
        j: int
            decimal point + 1

        Returns
        -------
        k_min, k_max: int, int
        """
        x = x[~np.isnan(x)]  # remove nans
        x = x[~np.isinf(x)]  # remove inf
        mat = np.zeros(shape=(len(x), j + 1))
        for idx in range(len(x)):
            for val in range(j):
                # split the integer into array. Add "1"*(j+1) to avoid problem with numbers starting by 0
                mat[idx, val] = int(str(int(float('% .{}f'.format(j)%np.abs(x[idx]))*10**j) + + int("1"*(j+1)))[val])

        diff_mat = np.diff(mat, axis=1)  # diff in between columns
        list_k = np.count_nonzero(diff_mat == 0., axis=1)  # count number of zeros in columns
        return np.min(list_k), np.max(list_k)

# =======================================================
#                  Extreme Expected Shortfall estimator
# =======================================================

class ExtremeBCTM(ExtremeQuantileEstimator):
    def __init__(self, X, a, alpha):
        """
        Extreme BCTM estimators

        The class contains:
        - Weissman (W) [1] : First Order approximation + Weissman + Hill
        - Refined Weissman (RW) [2]
        - Corrected Weissman (CW) [3]
        - (CH) [4]
        - (CH_{p^star}) [5]
        - (PRB_P) [6]
        - (PRB_{p^star}) [7]

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Data X_1, ..., X_n
        alpha : float
            extreme quantile level
        """
        super(ExtremeBCTM, self).__init__(X, alpha)
        self.alpha = alpha

        if (a>=0) and (a <=1):
            self.a = a
        else:
            raise ValueError("a must be between 0 and 1")
        self.dict_bctm_estimators = {"D": self.bctm_direct, "D_GRB": self.bctm_direct, "D_RB": self.bctm_direct_RB,
                                   "I": self.bctm_indirect, "I_RB": self.bctm_indirect_RB}
        return

    def bctm_estimator(self, method, k_anchor):
        if (method == "D") or (method == "I"):
            return self.dict_bctm_estimators[method](k_anchor, gamma="hill")
        elif method == "D_GRB":
            return self.dict_bctm_estimators[method](k_anchor, gamma="hill_RB")
        else:
            return self.dict_bctm_estimators[method](k_anchor)


    def bctm_direct(self, k_anchor, gamma="hill"):
        """
        Empirical Estimator Direct (D)
        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        Quantile estimator: float
        """
        X_anchor = self.X_order[-k_anchor:]
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        if gamma == "hill":
            tail_index = self.hill(k_anchor)
        elif gamma == "hill_RB":
            tail_index = self.corrected_hill(k_anchor)
        return box_cox(X_anchor * np.power(extrapolation_ratio, tail_index), self.a).mean()


    def bctm_indirect(self, k_anchor, gamma="hill"):
        """
        First order approximation with Weissman estimator Indicrect (I)
        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        Quantile estimator: float
        """
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        if gamma == "hill":
            tail_index = self.hill(k_anchor)
        elif gamma == "hill_RB":
            tail_index = self.corrected_hill(k_anchor)
        return (box_cox(X_anchor * np.power(extrapolation_ratio, tail_index), self.a) + tail_index) / (1 - (self.a*tail_index))



    def bctm_direct_RB(self, k_anchor):
        """
        Indirect First order approximation with Corrected Hill estimator (I_CH)
        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        Quantile estimator: float
        """
        gamma_hill_RB = self.corrected_hill(k_anchor)
        auxiliary_function = self.beta*gamma_hill_RB*np.power(self.n_data/k_anchor, self.rho)
        num = auxiliary_function * (1 - self.a*self.corrected_hill(k_anchor))
        denum = self.rho * (1 - (self.a * self.corrected_hill(k_anchor)) - self.rho)
        bias_term = num/denum
        term1 = 1 - (self.a*bias_term)
        return self.bctm_direct(k_anchor, gamma="hill_RB") * term1 - bias_term


    def bctm_indirect_RB(self, k_anchor):
        """
        Indirect First order approximation with Corrected Weissman estimator (I_CW)
        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        Quantile estimator: float
        """
        return self.bctm_indirect(k_anchor, gamma="hill_RB")





def tree_k(x, a=None, c=None, return_var=False):
    """
    choice of the best k based on the dyadic decomposition.
    returns the Python index (starts at 0).
    """
    if a is None:
        a = 13
    if c is None:
        c = int(3*x.shape[0] / 4)
    b = int((c+a)/2)

    list_var = []
    finish = False
    while not finish:
        if (b-a) < 2:
            finish = True
        else:
            v1 = np.var(x[a:b+1])
            v2 = np.var(x[b:c+1])
            if v1 < v2:  # left wins
                list_var.append(v1)
                c = b
            else:  # right wins
                list_var.append(v2)
                a = b
            b = int((c + a) / 2)
    if return_var:
        return b, np.mean(list_var)
    return b

def random_forest_k(x, n_forests, a=13/498, c=0.75, seed=42):
    """
    Algorithm to choose the intermediate sequence on a stable region given observations X_1,...,X_n
    Parameters
    ----------
    x : ndarray or list
        Observations
    n_forests : int
        number of forests in the algorithm
    seed : int
        Seed for PRGN

    Returns
    -------
    k : int
        selected anchor point (python indexing)
    """
    np.random.seed(seed)
    # a0 = 13
    # c0 = int(3 * x.shape[0] / 4)
    a0 = int(a * x.shape[0])
    c0 = int(c * x.shape[0])
    b0 = int((a0+c0)/2)
    list_k = []
    # print(a0,b0,c0)

    for i in range(n_forests):
        a = np.random.randint(a0, c0)
        c = np.random.randint(a+1, c0+1)
        # initial search in two parts
        # ---------------------------
        # a = np.random.randint(a0, b0)
        # c = np.random.randint(b0, c0+1)
        # ----------------------------

        list_k.append(tree_k(x, a, c))

    # return list_k[np.argmin(np.array(list_k)[:, 1])][0]
    # return int(np.median(np.array(list_k)[:, 0]))
    return int(np.median(np.array(list_k)))



def sim_estimators(n_replications, n_data, risk_level, distribution, params, a, zeta=0., metric="median", return_full=False):
    """
    Evaluation of CTM estimators based on simulated heavy-tailed data
    """
    dict_evt = {estimator: {_metric: {"series": [], "rmse_bestK": None, "bctm_bestK": [],
                            "bestK": []}for _metric in ["mean", "median"]} for estimator in list_estimators}

    pathdir = Path("ckpt", "sim", distribution, str(params))
    pathdir.mkdir(parents=True, exist_ok=True)
    pathfile = Path(pathdir, "sim_estimators_rep{}_ndata{}_rlevel{}_zeta{}_a{}.npy".format(n_replications, n_data, risk_level, zeta, a))

    anchor_points = np.arange(2 + int(zeta*n_data), n_data)  # 2+[zeta*n], ..., n-1
    data_sampler = DataSampler(distribution=distribution, params=params)
    real_bctm = data_sampler.ht_dist.box_conditional_tail_moment(risk_level, a)  # real CTM

    try:
        dict_evt = np.load(pathfile, allow_pickle=True)[()]
    except FileNotFoundError:
        # Estimators
        # -----------------
        for replication in range(1, n_replications + 1):  # for each replication
            print("rep ", replication)
            X_order = load_quantiles(distribution, params, n_data, rep=replication)  # replication order statistics X_1,n, ..., X_n,n
            dict_bctm = {estimator: [] for estimator in list_estimators}  # dict of bctm
            evt_estimators = ExtremeBCTM(X=X_order, a=a, alpha=risk_level)

            for estimator in list_estimators:
                for anchor_point in anchor_points:  # compute all quantile estimators
                    dict_bctm[estimator].append(evt_estimators.bctm_estimator(k_anchor=int(anchor_point), method=estimator))

                bestK = random_forest_k(np.array(dict_bctm[estimator]), 10000)  # \in [1,n-2] under the python indexing

                # MEAN
                dict_evt[estimator]["mean"]["series"].append(dict_bctm[estimator])
                dict_evt[estimator]["mean"]["bctm_bestK"].append(dict_bctm[estimator][int(bestK)])
                dict_evt[estimator]["mean"]["bestK"].append(bestK + anchor_points[0])  # k \geq 2

                # MEDIAN
                dict_evt[estimator]["median"]["series"].append(dict_bctm[estimator])
                dict_evt[estimator]["median"]["bctm_bestK"].append(dict_bctm[estimator][int(bestK)])
                dict_evt[estimator]["median"]["bestK"].append(bestK + anchor_points[0])  # k \geq 2

        # if not Path(pathdir, "evt_estimators_rep{}.npy".format(n_replications)).is_file():
        for estimator in list_estimators:
            # MEAN
            dict_evt[estimator]["mean"]["var"] = np.array(dict_evt[estimator]["mean"]["series"]).var(axis=0)
            dict_evt[estimator]["mean"]["rmse"] = ((np.array(dict_evt[estimator]["mean"]["series"]) / real_bctm - 1) ** 2).mean(axis=0)
            dict_evt[estimator]["mean"]["series"] = np.array(dict_evt[estimator]["mean"]["series"]).mean(axis=0)
            dict_evt[estimator]["mean"]["rmse_bestK"] = ((np.array(dict_evt[estimator]["mean"]["bctm_bestK"]) / real_bctm - 1) ** 2).mean()

            # MEDIAN
            dict_evt[estimator]["median"]["var"] = np.array(dict_evt[estimator]["median"]["series"]).var(axis=0)
            dict_evt[estimator]["median"]["rmse"] = np.median((np.array(dict_evt[estimator]["median"]["series"]) / real_bctm - 1) ** 2, axis=0)
            dict_evt[estimator]["median"]["series"] = np.median(dict_evt[estimator]["median"]["series"], axis=0)
            dict_evt[estimator]["median"]["rmse_bestK"] = np.median((np.array(dict_evt[estimator]["median"]["bctm_bestK"]) / real_bctm - 1) ** 2)

        np.save(pathfile, dict_evt)

    if return_full:
        return dict_evt
    df = pd.DataFrame(columns=list_estimators, index=["RMSE"])
    for estimator in list_estimators:
        df.loc["RMSE", estimator] = dict_evt[estimator][metric]["rmse_bestK"]
    return df


def sim_coverage_probabilities(n_replications, n_data, risk_level, distribution, params, a, zeta=0., metric="median", return_full=False):
    """Evaluation of the empirical coverage probabilities of confidence interval in (eq 18) associated to either
    the estimator (D,RB) or (I,RB)"""

    dict_res = {estimator: {"Lambda": [], "confidence_interval": [], "coverage_probability": []} for estimator in ["D_RB", "I_RB"]}

    pathdir = Path("ckpt", "sim", distribution, str(params))
    pathdir.mkdir(parents=True, exist_ok=True)
    pathfile_coverage = Path(pathdir, "sim_coverage_rep{}_ndata{}_rlevel{}_zeta{}_a{}.npy".format(n_replications, n_data, risk_level, zeta, a))

    anchor_points = np.arange(2 + int(zeta*n_data), n_data)  # 2+[zeta*n], ..., n-1
    data_sampler = DataSampler(distribution=distribution, params=params)
    real_bctm = data_sampler.ht_dist.box_conditional_tail_moment(risk_level, a)  # real CTM

    try:
        dict_res = np.load(pathfile_coverage, allow_pickle=True)[()]
    except FileNotFoundError:
        dict_estimators = sim_estimators(n_replications, n_data, risk_level, distribution, params, a, zeta=zeta, metric="median", return_full=True)
        norm_dist = scipy.stats.norm()
        q0_norm = norm_dist.ppf(1-0.05/2)  # fix alpha=0.05

        # Estimators
        # -----------------
        for replication in range(1, n_replications + 1):  # for each replication
            X_order = load_quantiles(distribution, params, n_data, rep=replication)  # replication order statistics X_1,n, ..., X_n,n
            tail_estimator = TailIndexEstimator(X_order)

            for estimator in ["D_RB", "I_RB"]:
                bestK = dict_estimators[estimator][metric]['bestK'][replication-1]
                gamma_RB = tail_estimator.corrected_hill(bestK)
                dict_res[estimator]["Lambda"].append(q0_norm* gamma_RB *np.log(2*bestK) * (bestK**-0.5))

                ci_inf, ci_sup = confidence_interval(X=dict_estimators[estimator][metric]["bctm_bestK"][replication-1][0], #[bestK],
                                                     gamma=gamma_RB,
                                                     k=bestK,
                                                     a=a)

                dict_res[estimator]["confidence_interval"].append((ci_inf, ci_sup))
                dict_res[estimator]["coverage_probability"].append((ci_inf<= real_bctm<=ci_sup)[0])

        # np.save(pathfile_coverage, dict_res)

    if return_full:
        return dict_res
    df = pd.DataFrame(columns=["D_RB", "I_RB"], index=["Coverage"])
    for estimator in ["D_RB", "I_RB"]:
        df.loc["Coverage", estimator] = np.mean(dict_res[estimator]["coverage_probability"])
    return df




def real_estimators(a, xi, zeta=0, percentile=0, metric="median", return_full=False):
    """
    Evaluation of extreme BCTM estimators based on real heavy-tailed data

    """
    dict_evt = {estimator: {_metric: {"series": [], "rmse_bestK": None, "q_bestK": [],
                            "bestK": []}for _metric in ["mean", "median"]} for estimator in list_estimators}

    pathdir = Path("ckpt", "real", "extrapolation")
    pathdir.mkdir(parents=True, exist_ok=True)

    # X_train, order_train, order_test = load_real_data(year=year, xi=xi)
    X = pd.read_csv("data/real/norwegian90.csv")
    n = X.shape[0]
    Xtrain = X[:int(np.floor(xi * n))].to_numpy()
    Xtest = X[-int(np.ceil((1 - xi) * n)):].to_numpy()
    n_train = Xtrain.shape[0]
    n_test = Xtest.shape[0]

    anchor_points = np.arange(2 + int(zeta*n_train), n_train)  # 2, ..., n-1
    EXTREME_ALPHA = 1 / (n_train)  # extreme alpha
    real_bctm = np.mean(box_cox(Xtest, a))

    try:
        dict_evt = np.load(Path(pathdir, "evt_estimators_a{}_xi{}_zeta{}_perc{}.npy".format(a, xi, zeta, percentile)), allow_pickle=True)[()]
    except FileNotFoundError:
        # Estimators
        # -----------------
        dict_bctm = {estimator: [] for estimator in list_estimators}  # dict of quantiles
        evt_estimators = ExtremeBCTM(X=Xtrain, a=a, alpha=EXTREME_ALPHA)

        for estimator in list_estimators:
            # if estimator == "I":
            #     print("")
            for anchor_point in anchor_points:  # compute all quantile estimators
                dict_bctm[estimator].append(evt_estimators.bctm_estimator(k_anchor=int(anchor_point), method=estimator))

            bestK = random_forest_k(np.array(dict_bctm[estimator]), 10000)

            # MEAN
            dict_evt[estimator]["mean"]["series"].append(dict_bctm[estimator])
            dict_evt[estimator]["mean"]["q_bestK"].append(dict_bctm[estimator][int(bestK)])
            dict_evt[estimator]["mean"]["bestK"].append(bestK + anchor_points[0])  # k \geq 2

            # MEDIAN
            dict_evt[estimator]["median"]["series"].append(dict_bctm[estimator])
            dict_evt[estimator]["median"]["q_bestK"].append(dict_bctm[estimator][int(bestK)])
            dict_evt[estimator]["median"]["bestK"].append(bestK + anchor_points[0])  # k \geq 2

        for estimator in list_estimators:
            # MEAN
            # dict_evt[estimator]["mean"]["var"] = np.array(dict_evt[estimator]["mean"]["series"]).var(axis=0)
            dict_evt[estimator]["mean"]["rmse"] = ((np.array(dict_evt[estimator]["mean"]["series"]) / real_bctm - 1) ** 2).mean(axis=0)
            dict_evt[estimator]["mean"]["series"] = np.array(dict_evt[estimator]["mean"]["series"]).mean(axis=0)
            dict_evt[estimator]["mean"]["rmse_bestK"] = ((np.array(dict_evt[estimator]["mean"]["q_bestK"]) / real_bctm - 1) ** 2).mean()

            # MEDIAN
            # dict_evt[estimator]["median"]["var"] = np.array(dict_evt[estimator]["median"]["series"]).var(axis=0)
            dict_evt[estimator]["median"]["rmse"] = np.median((np.array(dict_evt[estimator]["median"]["series"]) / real_bctm - 1) ** 2, axis=0)
            dict_evt[estimator]["median"]["series"] = np.median(dict_evt[estimator]["median"]["series"], axis=0)
            dict_evt[estimator]["median"]["rmse_bestK"] = np.median((np.array(dict_evt[estimator]["median"]["q_bestK"]) / real_bctm - 1) ** 2)

        np.save(Path(pathdir, "evt_estimators_a{}_xi{}_zeta{}_perc{}.npy".format(a, xi, zeta, percentile)), dict_evt)

    if return_full:
        return dict_evt
    df = pd.DataFrame(columns=list_estimators, index=["RMSE"])
    for estimator in list_estimators:
        df.loc["RMSE", estimator] = dict_evt[estimator][metric]["rmse_bestK"]
    return df


def confidence_interval(X, gamma, k, a, confidence_level=0.05):
    """Computes the confidence interval around X according to (eq 18)"""
    norm_dist = scipy.stats.norm()
    q0_norm = norm_dist.ppf(1 - confidence_level / 2)  # fix alpha=0.05
    Lamb = q0_norm * gamma * np.log(2 * k) * (k ** -0.5)
    ci_inf = (X - Lamb) / (1 + (a * Lamb))
    ci_sup = (X + Lamb) / (1 - (a * Lamb))
    return ci_inf, ci_sup

def get_real_cte_half(xi, zeta=0, gamma_estimator="hill", metric="median"):
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

    anchor_points = np.arange(2 + int(zeta*n_train), n_train) # 2, ..., n-1
    # anchor_points = np.arange(2, n_train)  # 2, ..., n-1

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
    # return dict_cte, gamma, bestK
    return dict_cte



