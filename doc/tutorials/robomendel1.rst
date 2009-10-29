=============================
Basic RoboMendel Calculations
=============================

Why RoboMendel is Bored
-----------------------

RoboMendel has grown pea plants for 5 years, and has seen many, many
pea flowers, all of them purple.  The poor guy is bored.
This is because he sees no potential information in his observations.
Each year he observed 20 plants, each with 100 flowers::

   >>> from data import *
   >>> import math, numpy
   >>> from scipy import stats
   >>> obsPu = plantPu.get_phenotypes()[0].rvs(5 * 20 * 100)

He constructs a model from his observations::

   >>> mean = numpy.average(obsPu)
   >>> var = numpy.average(obsPu * obsPu) - mean * mean
   >>> modelPu = stats.norm(mean, math.sqrt(var))

He collects a new set of observations this year.
Next he calculates the empirical entropy of the observations,
and the empirical likelihood::

   >>> obsNew = plantPu.get_phenotypes()[0].rvs(20 * 100)
   >>> He = entropy.box_entropy(obsNew, 7)
   >>> Le = entropy.sample_Le(obsNew, modelPu)

From this he calculates the potential information of the observations
relative to his model::

   >>> Ip = -Le - He
   >>> Ip.mean
   -0.003343603207393906
   >>> Ip.get_bound()
   -0.031671209796191716

Evidently, there is no more information to be gained by improving
his model.

A Surprising Observation
------------------------

But while tending his pea plants, something catches RoboMendel's
robotic eye: a *white* pea flower!  Why does it "catch" his eye?
Because it signals potential information::

   >>> obsQuick = plantWh.get_phenotypes()[0].rvs(1)
   >>> obsQuick
   array([-1.41985693])

Pretty darn white.  To calculate *Ip* he combines it with the 
other observations, and redoes the calculation::

   >>> allObs = numpy.concatenate((obsNew,obsQuick))
   >>> He = entropy.box_entropy(allObs, 7)
   >>> Le = entropy.sample_Le(allObs, modelPu)
   >>> Ip = -Le - He
   >>> Ip.mean
   0.02530458797888565

This signals the presence of *positive* potential information.
Of course this can be localized to his new observation::

   >>> allObs[-1]
   -1.4198569271863846
   >>> -Le.sample[-1] - He.sample[-1]
   58.321937043903041

Holy schnikees!  58 nats of *Ip* from one observation?  Compare
that with one of his typical previous observations::

   >>> -Le.sample[0] - He.sample[0]
   -0.84541352546760873

Of course, a single observation doesn't give him confidence that
he has strong potential information, as the 5% confidence lower
bound shows him::

   >>> Ip.get_bound()
   -0.071098984744074889

This indicates strong expectation *Ip* for collecting a large
sample of observations from the "wierd" pea plant: he expects to
produce up to 58 nats of *Ip* by raising the lower bound of
his confidence interval.  So he obtains another 100 observations::

   >>> obsWh = plantWh.get_phenotypes()[0].rvs(100)
   >>> He = entropy.box_entropy(obsWh, 7)
   >>> Le = entropy.sample_Le(obsWh, modelPu)
   >>> Ip = -Le - He
   >>> Ip.mean
   50.203558792565801
   >>> Ip.get_bound()
   47.207416983257303

This tells RoboMendel that he's discovered a new set of 
observations that convincingly do not fit ``modelPu``.
