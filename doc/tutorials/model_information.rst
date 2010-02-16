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

===============================
Model Information Drops to Zero
===============================

Let's compute the effect of sample size on model information

    >>> from darwin.robomendel import *
    >>> from darwin.entropy import *
    >>> import math
    >>> from scipy import stats
    >>> import pylab

First construct a simulation distribution and sample data.

    >>> m = 300
    >>> simulation_model = stats.norm(0, 1)
    >>> sample = simulation_model.rvs(m)

Now draw the set of observations.

    >>> n = 300
    >>> obs_list = list(simulation_model.rvs(n))

Compute the model information that comes from i observations as i ranges from 3 to n.

    >>> im_points = []
    >>> for i in range(3, n):
    >>>    obs = numpy.core.array(obs_list[0:i])
    >>>    mean = numpy.average(obs)
    >>>    var = numpy.average(obs * obs) - mean * mean
    >>>    model_obs = stats.norm(mean, math.sqrt(var))
    >>>    l_e = box_entropy(obs, min(len(obs)-1, 7), sample=sample)
    >>>    log_prior = sample_Le(sample, model_obs)
    >>>    i_m = -l_e - log_prior
    >>>    im_points.append((i, i_m.mean))

Finally, plot the model information against the sample size.

    >>> pylab.plot([x for (x,y) in im_points], [y for (x,y) in im_points])
    >>> pylab.xlabel('sample_size')
    >>> pylab.ylabel('Im')
    >>> pylab.grid(True)
    >>> pylab.show()

