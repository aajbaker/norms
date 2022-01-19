"""
Bayesian models of norm updating.

This project is developed as part of the Boston College Cooperation Lab. 
"""
__author__ = "Aaron Baker"
__contact__ = "aaron@aajbaker.com"
__credits__ = [""]
__date__ = "2021/04/29"
__version__ = "0.0.1"

import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import beta

norms2 = pd.read_csv("/Users/aaronbaker/Documents/Cooperation Lab/Norms/norms_s2_clean.csv")
norms3 = pd.read_csv("/Users/aaronbaker/Documents/Cooperation Lab/Norms/norms_s3_clean.csv")

def get_data(df, vign, norm_type, prior_post, strong_weak):
    """ Take in a dataframe DF and vignette (number) VIGN and return a series of 
    norm ratings based on NORM_TYPE. """
    df_vign = df[df["vign.id"] == vign]
    df_prior_post = df_vign[df_vign["desc_manip"] == prior_post]
    df_strong_weak = df_prior_post[df_prior_post["cond"] == strong_weak]
    return df_strong_weak[norm_type]

def collect_vigns(df, norm_type, n_vigns, strong_weak):
    """ From dataframe DF, return a dictionary that contains the prior and posterior 
    Beta distributions for each vignette based on NORM_TYPE (ex. injunctive) and
    number of vignettes N_VIGNS. """
    collection = {}
    for i in range(1, n_vigns+1):
        prior = get_data(df, i, norm_type, "prior", strong_weak)
        post = get_data(df, i, norm_type, "post", strong_weak)
        title = df.loc[df["vign.id"] == i, 'vignette'].iloc[0]
        collection[title] = [Beta(prior, bins, xvalues, title), Beta(post, bins, xvalues, title)]
    return collection

ex_prior = get_data(norms3, 1, "injunctive", "prior", "strong")
ex_post = get_data(norms3, 1, "injunctive", "post", "strong")

descriptive_obs = 0.8

k = 1
c = 0

with pm.Model() as model:
    """Estimating probabilities over means for posterior injunctive ratings separated by domain"""
    # mu_conv = pm.Beta()
    # mu_moral pm.Beta()
    # injunctive_conv = pm.Beta(mu = mu_conv, observed = ex_prior[conv_contexts])
    # injuctive_moral = pm.Beta(mu = mu_moral, observed = ex_prior[moral_contexts])
    
    """Estimating probabilities over transform parameters K and C that interact with the prior incunctive
    rating Beta distribution INJUNCTIVE and the discriptive norm information OBSERVED."""
    #k = pm.Beta('k', mu=0)
    #c = pm.Beta('c')
    #injunctive = pm.Beta('injuctive', mu=np.mean(ex_prior)/100, sigma=np.std(ex_prior)/100)
    #descriptive = pm.Beta('descriptive', mu=k*injunctive + c, sigma=0.5, observed = descriptive_obs)
    
    """Modeling the posterior distribution as a binomial distribution (N observtions with OBS 
    success rate) instead of a beta distribution."""
    # descriptive = pm.Binomial('likelihood', p = pm.math.sigmoid(k * injunctive + C),
    #  n = N, obs = descriptive_obs)

    """Tester of pymc3 mechanics."""
    k = pm.Constant('k', 1)
    c = pm.Constant('c', 0)
    mean = pm.Normal("mean", mu=0, sigma=1)
    desc = pm.Normal("descriptive", mu=k*mean + c, sigma=1)


    trace = pm.sample(5000, tune=1000, progressbar=True, return_inferencedata=True)

posterior_mean = trace["injuctive"].mean(axis=0)

plt.scatter(posterior_mean, ex_post)
plt.show()