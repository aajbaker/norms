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

norms = pd.read_csv("norms_s2_clean.csv")

# Parameters: standard parameters for the probability density functions.
xvalues = np.linspace(0, 1.0, 100)
p_strong = 0.8
p_weak = 0.2
n = 10
bins = 10
norm_type = "injunctive"

def get_data(df, vign, norm_type, prior_post):
    """ Take in a dataframe DF and vignette (number) VIGN and return a series of 
    norm ratings based on NORM_TYPE. """
    df_prior_post = df[df["desc_manip"] == prior_post]
    df_vign = df_prior_post[df_prior_post["vign.id"] == vign]
    return df_vign[norm_type]

class Beta():
    def __init__(self, prior, bins, xvals):
        """ Make a new instance of Beta distribution after the sorting the data into
        BINS number of bins. """
        self.prior = prior
        self.bins = bins
        self.xvals = xvals
        self.bin_prior = prior.apply(lambda x: x // self.bins)
        self.fit()
        
    def fit(self):
        """ Set parameters for a beta distribution using SELF.DATA grouped into SELF.BINS."""
        self.a, self.b, self.loc, self.scale = beta.fit(self.bin_prior)
    
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

grass_data = get_data(norms, 1, norm_type, "prior")
grass = Beta(grass_data, bins, xvalues)
grass_post_data = get_data(norms, 1, norm_type, "post")
grass_post = Beta(grass_post_data, bins, xvalues)

plt.plot(xvalues, grass.pdf())
for i in range(4):
    grass.update(4, 5)
    plt.plot(xvalues, grass.pdf())
plt.plot(xvalues, grass_post.pdf())

plt.show()

""" Other functions. """
def calc_likelihood(k, n, x):
    """ Returns normalized probability mass function for a binomial distribution. For each considered
    probability in X, what is the likelihood of seeing K outcomes out of N trials. """
    pmf = stats.binom.pmf(k, n, x)
    normalize = np.linalg.norm(pmf)
    normal_pmf = pmf / normalize
    return normal_pmf