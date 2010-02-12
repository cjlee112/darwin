=============================
Computing Model Information
=============================

    >>> from darwin.entropy import *
    >>> from scipy import stats

Let's set up a simulation model with a normal distribution of mean 10 and deviation 1.

    >>> simulation_model = stats.norm(10,1)

Draw a set of observations.

    >>> n = 20
    >>> obs = simulation_model.rvs(n)

Compute the log-likelihood of the uniformative prior. In this case we use a uniform distribution on the detector range [0,20]:

    >>> prior = stats.uniform(0,20)
    >>> lup = sample_Le(obs, prior)

Suppose our model is a normal with mean 9 and deviation 1.

    >>> model = stats.norm(9,1)
    >>> l_e = sample_Le(obs, model)

Compute the model information as the difference:

    >>> I_m = l_e - lup
    >>> I_m.mean
    1.1755261031761506
    >>> I_m.get_bound()
    0.63228464074704083





