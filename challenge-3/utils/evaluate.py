import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

'''
Computing divergence for discrete variables
https://github.com/michaelnowotny/divergence
'''

def compute_probs(data, n=50): 
    h, e = np.histogram(data, n)
    p = h/data.shape[0]
    return e, p

def support_intersection(p, q): 
    sup_int = list(
                filter(
                    lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
                )
    )
    return sup_int

def get_probs(list_of_tuples): 
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q

def kl_divergence(p, q): 
    return np.sum(p*np.log(p/q))

def js_divergence(p, q):
    m = (1./2.)*(p + q)
    return (1./2.)*kl_divergence(p, m) + (1./2.)*kl_divergence(q, m)

def compute_kl_divergence(train_sample, test_sample, n_bins=50): 
    """
    Computes the KL Divergence using the support 
    intersection between two different samples
    """
    # get bins ranging over both distributions 
    e, _ = compute_probs(np.concatenate((train_sample,test_sample), axis=0), n=n_bins)
    
    _, p = compute_probs(train_sample, n=e)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)
    
    return kl_divergence(p, q)

def compute_js_divergence(train_sample, test_sample, n_bins=50): 
    """
    Computes the JS Divergence using the support 
    intersection between two different samples
    """
    
    # get bins ranging over both distributions 
    e, _ = compute_probs(np.concatenate((train_sample,test_sample), axis=0), n=n_bins)
    
    _, p = compute_probs(train_sample, n=e)
    _, q = compute_probs(test_sample, n=e)
    
    list_of_tuples = support_intersection(p,q)
    p, q = get_probs(list_of_tuples)
    
    return js_divergence(p, q)

def compute_ks_statistic(train_sample, test_sample):
    return ks_2samp(test_sample, train_sample)[0]

def compute_wasserstein_distance(train_sample, test_sample):
    return wasserstein_distance(test_sample, train_sample)
