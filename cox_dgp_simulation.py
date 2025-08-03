import numpy as np
 
from numpy.random import normal, uniform, exponential, binomial, beta, weibull
from scipy.stats import norm
from sklearn.metrics.pairwise import pairwise_distances

def simulate_cox_failure_time(log_hr: np.array, S_0_inv = lambda p: -np.log(p), max_out=10000):
    """Inverse transform simulation"""
    U = uniform(0, 1, len(log_hr))
    S_0_inv_input = (1 - U)**np.exp(-log_hr)
    out = S_0_inv((1 - U)**np.exp(-log_hr))
    out[out > max_out] = max_out
    return out

def simulate_cox_cond_indep_censor_time(censor_rate: float, log_hr: np.array, S_0_inv=lambda p: -np.log(p)):
    # print(np.log(censor_rate) - np.log(np.exp(log_hr)-censor_rate) + log_hr)
    return simulate_cox_failure_time(-np.log(1/censor_rate - 1 + 10e-7) + log_hr, S_0_inv)

def simulate_cox_dgp(log_hr, S_0_inv = lambda p: -np.log(p), censor_rate = 0.4):
    """
    Simulate cox failure time, censoring time, event and event time.

    Parameters
    ----------
    log_hr : 1darray
        Log hazard ratios for simulating failure times.
    S_0_inv : function
        Inverse of the baseline survival function.
    censor_rate : float, optional
        The expected proportion of censoring event. Must be between 0 and 1.

    Returns
    -------
    sim_failure_time : 1darray
        Failure time
    sim_censor_time : 1darray
        Censoring time
    sim_event : 1darray
        Boolean array of event indicator (0 = censored, 1 = failure)
    sim_time: 1darray
        Censored failure time
    """
    # sim_X = normal(size=(sim_n, len(beta))) # normally distributed X
    # Simulate failure time under Cox model with exponential S_0(t)
    #sim_failure_time = exponential(scale=S_0_scale, size=sim_n)*np.exp(-log_hr)
    sim_failure_time = simulate_cox_failure_time(log_hr, S_0_inv)

    # Conditional independent censoring
    #sim_censor_time = exponential(scale=S_0_scale*(1/censor_rate - 1), size=sim_n)*np.exp(-np.dot(sim_X, beta))
    sim_censor_time = simulate_cox_cond_indep_censor_time(censor_rate, log_hr, S_0_inv)

    sim_event = (sim_failure_time <= sim_censor_time) # event indicator
    sim_time = sim_failure_time*sim_event + sim_censor_time*(~sim_event) # observed time
    sim_event = np.float64(sim_event)
    return sim_failure_time, sim_censor_time, sim_event, sim_time

def simulate_weibull_failure_cox(log_hr, S0_shape = 1.0, S0_scale=1.0, censor_rate=0.4):
    sim_failure_time = weibull(a=S0_shape, size=len(log_hr))*S0_scale*np.exp(-log_hr/S0_shape)
    # Conditional independent censoring
    #sim_censor_time = exponential(scale=S_0_scale*(1/censor_rate - 1), size=sim_n)*np.exp(-np.dot(sim_X, beta))
    sim_censor_time = weibull(a=S0_shape, size=len(log_hr))*S0_scale*np.exp(-log_hr/S0_shape)*((1-censor_rate)/censor_rate)**(1/S0_shape)

    sim_event = (sim_failure_time <= sim_censor_time) # event indicator
    sim_time = sim_failure_time*sim_event + sim_censor_time*(~sim_event) # observed time
    sim_event = np.float64(sim_event)
    return sim_failure_time, sim_censor_time, sim_event, sim_time

if __name__ == "__main__":
    beta = np.array([3,3])
    n = 100000
    sim_X = normal(size=(n, len(beta)))
    log_hr = np.dot(sim_X, beta)
    censor_rate = 0.2
    sim_failure_time, sim_censor_time, sim_event, sim_time = simulate_weibull_failure_cox(log_hr, S0_shape=0.5, S0_scale=5, censor_rate=censor_rate)
    print(sim_event.mean())