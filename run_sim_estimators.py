from extreme.estimators import sim_estimators, sim_coverage_probabilities
from itertools import product

"""
BCTM estimation on simulated data
"""

# ----------- Hyperparameters -------------
LIST_EVI = [0.1, 0.3, 0.5, 0.7, 0.9]  # extreme value indices
LIST_SOP = [-1., -0.75, -0.5, -0.25]  # second order parameters
N_DATA = 500
N_REPLICATIONS = 500
RISK_LEVEL = 1 / (2 * N_DATA)
A = 1  # power level
ZETA = 0.02  # consider the anchor points $k_n\in\{1+[\zeta n], \dots, n-1 \}$ with $\zeta\in[0,1)$
# LIST_DISTRIBUTIONS = ["burr", "gpd", "student", "pareto", "fisher", "frechet", "invgamma"]
LIST_DISTRIBUTIONS = ["burr"]
# ----------------------------------------


dict_runner = {"burr": {"evi": LIST_EVI, "rho": LIST_SOP},
               "gpd": {"evi": LIST_EVI},
               "student": {"evi": LIST_EVI},
                "pareto": {"evi": LIST_EVI},
                "fisher": {"evi": LIST_EVI},
                "frechet": {"evi": LIST_EVI},
                "invgamma": {"evi": LIST_EVI}
               }

if __name__ == "__main__":
    for distribution in LIST_DISTRIBUTIONS:
        keys, values = zip(*dict_runner[distribution].items())
        permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
        for parameters in permutations_dicts:
            print("{}: ".format(distribution),parameters)
            # run estimators
            # --------------
            # sim_estimators(n_replications=N_REPLICATIONS, params=parameters,
            #                distribution=distribution, n_data=N_DATA, risk_level=RISK_LEVEL, a=A, zeta=ZETA, metric="median")

            # run coverage probabilities
            # -------------------------
            sim_coverage_probabilities(n_replications=N_REPLICATIONS, params=parameters,
                           distribution=distribution, n_data=N_DATA, risk_level=RISK_LEVEL, a=A, zeta=ZETA, metric="median")


