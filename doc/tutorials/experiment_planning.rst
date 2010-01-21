=============================
Experiment Planning
=============================

Incorporating Findings into the Model of Models
-----------------------------------------------

Robomendel has discovered that a phenotypic model for the color of the flowers on a pea plant fits his observations well, but has an apparent contradiction his model of models as to how pea plants behave. How is it that there are pea plants of two different colors? Do the different colors indicate that there are two different species of plants in robomendels virtual garden or that there is a further underlying mechanism for the behavior of the observed characteristic?

To test the species model, Robomendel decides to peform two experiments in the next growing season: a Wh x Wh cross and a Wh x Pu cross. Robomendel selects a plant with white flowers and a plant with purple flowers:

   >>> from darwin.robomendel import plantWh, plantPu
   >>> import numpy

Crossing these plants should shed light on the same species versus different species hypothesis. Robomendel checks the results from cross the white plant with itself:
    
    >>> white_crosses = [plantWh * plantWh for i in range(10)]
    >>> for plant in white_crosses:
    ...     obs = plant.get_phenotypes()[0].rvs(10)
    ...     mean = numpy.average(obs)
    ...     var = numpy.average(obs * obs) - mean * mean
    ...     print mean, var
    ...
    -0.471965742027 1.11017135835
    0.119918211243 1.52079207705
    -0.103758390658 0.883358813477
    -0.109359686858 0.270194415697
    0.254738897309 1.83188423526
    0.268314939624 0.767983220431
    -0.309089679625 0.555288698232
    0.206205139635 0.736532155415
    0.319249321975 0.645256877208
    0.0587909903171 0.272983234749

Looks like all white-flowered progeny! His observations are in line with the predictions for the different species model. Robomendel also crosses plantWh and plantPu. Robomendel checks the results.

    >>> hybrid_crosses = [plantWh * plantPu for i in range(10)]
    >>> for plant in hybrid_crosses:
    ...     obs = plant.get_phenotypes()[0].rvs(10)
    ...     mean = numpy.average(obs)
    ...     var = numpy.average(obs * obs) - mean * mean
    ...     print mean, var
    ...
    10.1699200267 1.09444072719
    9.63167641328 0.821067659727
    10.032746143 1.38835868502
    10.0331164469 0.71733403866
    9.71161069374 0.764024319284
    10.2453142774 0.723538153903
    9.53605509441 1.31443966277
    10.0536374559 0.846048471687
    9.78362774467 0.556655591317
    10.280458592 1.13427654906

Not only did the seeds produce plants, all the flowers are purple! This fits the prediction of the same species model! Not only has RoboMendel failed to result the source of the white flowers observation, he has another contradiction!

What's a robotic scientist to do!? Robomendel knows that in some cases infertile offspring can be produced by different species, so he decides to determine if the hybrid plants are sterile. He crosses a large number of hybrid plants and checks the results.

    >>> import random
    >>> hybrid_crosses = [plantWh * plantPu for i in range(100)]
    >>> hybrid_progeny = [random.choice(hybrid_crosses) * random.choice(hybrid_crosses) for i in range(100)]

    >>> hybrid_means = list()
    >>> for plant in hybrid_progeny:
    ...     obs = plant.get_phenotypes()[0].rvs(10)
    ...     mean = numpy.average(obs)
    ...     hybrid_means.append(mean)
    ...
    >>> purple_estimate = len([x for x in hybrid_means if x > 5])
    >>> purple_estimate
    77
    >>> white_estimate = len([x for x in hybrid_means if x < 5])
    >>> white_estimate
    23

Not only are the seeds viable, over 20% of the plants have white flowers and nearly 80% have purple flowers! Repeating the experiment, RoboMendel finds that on average 25% of the plants have white flowers.

RoboMendel's latest experiment deals a decisive blow to the different species model, but the pattern of offspring does not fit the bio-object model RoboMendel has been using for his pea plants.
