"""
Bayesian models of norm updating.

This project is developed as part of the Boston College Cooperation Lab. 
"""
__author__ = "Aaron Baker"
__contact__ = "aaron@aajbaker.com"
__credits__ = [""]
__date__ = "2021/04/29"
__version__ = "0.0.1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import beta

norms2 = pd.read_csv("FILE HERE")
norms3 = pd.read_csv("FILE HERE")

# Parameters: EDIT THESE based on the different results you would like to see.
p_strong = 0.8
p_weak = 0.2
batch_size = 20
num_batches = 1
norm_type = "descriptive"
strong_weak = "strong"

# Parameters: Shape parameters that affect the granularity of the distributions.
xvalues = np.linspace(0, 1.0, 100)
bins = 10

class Beta():
    def __init__(self, prior, bins, xvals, title):
        """ Make a new instance of Beta distribution after the sorting the data into
        BINS number of bins. """
        self.prior = prior
        self.bins = bins
        self.xvals = xvals
        self.title = title
        self.bin_prior = prior.apply(lambda x: x // self.bins)
        self.proportion_prior = prior.apply(lambda x: x / 100)
        self.fit_method_moments()
        
    def fit_method_moments(self):
        """ Fit parameters for beta distribution using method of moments 
        (mean and variance of the data). """
        mean = self.proportion_prior.mean()
        var = self.proportion_prior.var()
        self.a = mean * ((mean * (1-mean) / var) - 1)
        self.b = (1-mean) * ((mean * (1-mean) / var) - 1)
    
    def fit_mle(self):
        """ Set parameters for a beta distribution using SELF.DATA grouped into SELF.BINS."""
        self.a, self.b, self.loc, self.scale = beta.fit(self.bin_prior, loc=0, scale=1)
    
    def pdf(self):
        """ Return a probability density function using the attributes a, b, and xvals."""
        return beta.pdf(self.xvals, self.a, self.b)

    def sf(self):
        """ Return a survival function using attributes, a, b, and xvals."""
        return beta.sf(self.xvals, self.a, self.b)

    def update(self, k, n):
        """ Update parameters alpha and beta based on N observations with K 
        occurrences taking value 1."""
        self.a = self.a + k
        self.b = self.b + (n-k)

    def mean(self):
        """ Calculate the mean of the current beta distribution using attributes a and b."""
        return beta.mean(self.a, self.b)

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

def plot_beliefs(dict, vign, num_batches, batch_size, p_success):
    """ Plot the prior pdf for vignette VIGN from the collection DICT, along with each 
    posterior after each batch update of size BATCH_SIZE with a proportion of 
    P_SUCCESS successes. Then plot the actual posterior belief from the data. 
    Note that P_SUCCESS * BATCH_SIZE must yield a whole number. """
    prior = dict[vign][0]
    post = dict[vign][1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle(vign)
    plt.plot(xvalues, prior.pdf(), label="prior")
    for i in range(num_batches):
        prior.update(p_success*batch_size, batch_size)
        plt.plot(xvalues, prior.pdf(), label="update "+str(i+1))
    plt.plot(xvalues, post.pdf(), label="post")
    plt.legend()
    corr = stats.pearsonr(prior.pdf()[1:-1], post.pdf()[1:-1])
    plt.text(0.5, 1.05, "Pearson corrleation (R, P): " + str(corr), ha='center', va='center', transform=ax.transAxes)
    return fig

def plot_hist(dict, vign):
    """ Plot the histograms for the prior and posterior participant data for 
    each of vignette VIGN from DICT. """
    prior = dict[vign][0].bin_prior
    post = dict[vign][1].bin_prior
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle(vign)
    plt.hist(prior, label="prior", histtype="step")
    plt.hist(post, label="post", histtype="step")
    plt.legend()
    return fig

vignettes = collect_vigns(norms3, norm_type, 12, strong_weak)
vigns2 = collect_vigns(norms3, norm_type, 12, "weak")
for vign in vignettes:    
    plot_beliefs(vignettes, vign, num_batches, batch_size, p_strong if strong_weak=="strong" else p_weak)
    plot_hist(vignettes, vign)
plt.show()




""" OTHER FUNCTIONS """
def calc_likelihood(k, n, x):
    """ Returns normalized probability mass function for a binomial distribution. For each considered
    probability in X, what is the likelihood of seeing K outcomes out of N trials. """
    pmf = stats.binom.pmf(k, n, x)
    normalize = np.linalg.norm(pmf)
    normal_pmf = pmf / normalize
    return normal_pmf