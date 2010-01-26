=============================
Basic RoboMendel Calculations
=============================

Why RoboMendel is Bored
-----------------------

RoboMendel has grown pea plants for 5 years, and has seen many, many
pea flowers, all of them purple.  The poor guy is bored.
This is because he sees no potential information in his observations.
Each year he observed 20 plants, each with 100 flowers::

   >>> from test.robomendel_data import *
   >>> import math, numpy
   >>> from scipy import stats
   >>> obsPu = plantPu.get_phenotypes()[0].rvs(5 * 20 * 100)

He constructs a model from his observations::

   >>> mean = numpy.average(obsPu)
   >>> var = numpy.average(obsPu * obsPu) - mean * mean
   >>> modelPu = stats.norm(mean, math.sqrt(var))

He collects a new set of observations this year.
Next he calculates the empirical entropy of the observations,
and the empirical log-likelihood::

   >>> obsNew = plantPu.get_phenotypes()[0].rvs(20 * 100)
   >>> from darwin import entropy
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

Testing a Simple Fix
--------------------

RoboMendel always tries the simplest fix first.  In particular,
he constructed his old model by simply training on the past data,
and tested it by measuring how well it predicts new observations.  This
amounts to assuming that all the observations were emitted I.I.D.
from the same distribution.  He can do the same thing with the
new observations.  In his garden, his training data show
that approximately 10% of the flowers are white, vs. 90% are purple.
He trains a new model approximately as follows::

   >>> mean = numpy.average(obsWh)
   >>> var = numpy.average(obsWh * obsWh) - mean * mean
   >>> modelWh = stats.norm(mean, math.sqrt(var))
   >>> import mixture
   >>> modelMix = mixture.Mixture(((0.9, modelPu), (0.1, modelWh)))

To assess whether this model is an improvement over his old
model, he calculates the empirical information gain::

   >>> LeMix = entropy.sample_Le(obsWh, modelMix)
   >>> Ie = LeMix - Le
   >>> Ie.mean
   47.821464796147559
   >>> Ie.get_bound()
   44.806512600644481

This provides a convincing demonstration that RoboMendel should abandon
the old ``modelPu`` (which asserts that no white flowers exist),
in favor of the new mixture model.  One way of describing this is
that the mixture
model has converted approximately 45 nats of *potential information*
into *empirical information*, i.e. a measurable improvement in 
prediction power.

Can RoboMendel rest easy after his success?
He now calculates the potential information for the mixture model
from his "wierd" plant::

   >>> Ip = -LeMix - He
   >>> Ip.mean
   2.3820939964182544
   >>> Ip.get_bound()
   2.2479290384886377

This strong potential information reflects a basic mismatch
versus the model: the flower colors do not appear to be drawn I.I.D.
Instead of each flower having a 10% chance of being white, RoboMendel
sees that on certain plants, *all flowers* are white
(the precise value of *Ip*
indicates that white flowers are occuring about 10 times more frequently
than the model says they should), whereas on the
remaining plants *all flowers* are purple.  Indeed the purple plants
also show strong *Ip* vs. this model::

   >>> He = entropy.box_entropy(obsNew, 7)
   >>> Le = entropy.sample_Le(obsNew, modelMix)
   >>> Ip = -Le - He
   >>> Ip.mean
   0.10201691139077627
   >>> Ip.get_bound()
   0.073689305323023341

Evidently, a more sophisticated model is required.

How to decide what to do next?
------------------------------

Note that RoboMendel has two distinct directions for further work
to choose from:

* he could simply collect more observations from his "white plant"
  to raise the *Ip* lower bound from 2.25 up to as high as 2.38
  (a gain of up to 0.13 nats).

* he could try to convert the 2.25 nats of potential information
  to empirical information.

The latter is clearly a much greater win (simply by total information
magnitude).  RoboMendel can add more decimal points to his
accuracy later -- first he has a major failure of his model to fix!

Defining a Phenotype
--------------------

Since the observations cluster by individual plant, RoboMendel's
next creates a hidden variable :math:`\Theta_i` associated with each plant
*i*, with two possible values {**WH**, **PU**}, which emit
white vs. purple flowers respectively.  Again using the simplest
possible model, RoboMendel assumes that :math:`\Theta_i` is emitted
I.I.D. for each plant with a binomial probability 
:math:`p(\Theta_p=WH)`.  Since he has observed only one white plant
out of the 100 pea plants he's seen over the last five years, he
estimates this binomial probability to be 1%.

To model this, we first create objects representing the two possible
states of this hidden variable::

   >>> from darwin.model import *
   >>> pstate = LinearState('Pu', modelPu)
   >>> wstate = LinearState('Wh', modelWh)

We specify the prior probability for each state as a transition probability
from the initial 'START' state::

   >>> prior = StateGraph({'START':{pstate:0.9, wstate:0.1}})

We also need to specify that each state can "exit" to the terminal 'STOP'
state::

   >>> stop = StopState()
   >>> term = StateGraph({pstate:{stop:1.}, wstate:{stop:1.}})

We assemble these into a simple model graph that shows the structure
of these variables; here we merely draw them as a star-topology from the 
initial 'START' state.  We name our phenotype variable `chi`::

   >>> dg = model.Model(model.NodeGraph({'START':{'chi':prior},
   ...                                   'chi':{'chi':term}}))
   ...

Finally, we package each plant's observations in an *observation graph*
keyed by the possible plant IDs::

   >>> d = {}
   >>> for plant in range(2): # two white plants
   >>>    d[plant] = modelWh.rvs(100)
   ...
   >>> for plant in range(2, 20): # 18 purple plants
   >>>    d[plant] = modelPu.rvs(100)
   ...
   >>> obsGraph = model.ObsGraph({'START':d})

Note that this creates a start topology of 20 independent
observation groups, each connected directly to the START node.

We now compute the model probabilities using the forward-backward
algorithm::

   >>> logPobs = dg.calc_fb((obsGraph,))

This gives us the total log-probability of the entire set of observations.
We can also compute the posterior likelihood of each of the observations::

   >>> llDict = dg.posterior_ll()

This analyzes the likelihood of each observation conditioned on all 
previous observations.  For example, once the model sees several white
flowers from one plant, it will predict that future flowers from that
plant will probably be white as well.

We can use these posterior likelihoods to compute the empirical information
gain versus the previous mixture model::

   >>> for plant in range(20):
   ...     obs = d[plant]
   ...     obsLabel = obsGraph.get_label(plant)
   ...     nodeLabel = dg.graph.get_label('chi', (obsLabel,))
   ...     Le = entropy.LogPVector(numpy.array(llDict[nodeLabel]))
   ...     LeMix = entropy.sample_Le(obs, modelMix)
   ...     Ie = Le - LeMix
   ...     He = entropy.box_entropy(obs, 7)
   ...     Ip = -Le - He
   ...     print 'plant %d, Ie > %1.3f, mean = %1.3f\tIp > %1.3f, mean = %1.3f' \
   ...           % (plant, Ie.get_bound(), Ie.mean, Ip.get_bound(), Ip.mean)
   ...
   plant 0, Ie > 2.207, mean = 2.280	Ip > -0.114, mean = 0.019
   plant 1, Ie > 2.207, mean = 2.280	Ip > -0.114, mean = 0.000
   plant 2, Ie > 0.101, mean = 0.104	Ip > -0.151, mean = -0.048
   plant 3, Ie > 0.101, mean = 0.104	Ip > -0.117, mean = 0.012
   plant 4, Ie > 0.101, mean = 0.104	Ip > -0.142, mean = -0.020
   plant 5, Ie > 0.101, mean = 0.104	Ip > -0.120, mean = 0.053
   plant 6, Ie > 0.101, mean = 0.104	Ip > -0.069, mean = 0.047
   plant 7, Ie > 0.101, mean = 0.104	Ip > -0.097, mean = 0.067
   plant 8, Ie > 0.101, mean = 0.104	Ip > -0.130, mean = -0.006
   plant 9, Ie > 0.101, mean = 0.104	Ip > -0.127, mean = -0.001
   plant 10, Ie > 0.101, mean = 0.104	Ip > -0.107, mean = 0.017
   plant 11, Ie > 0.101, mean = 0.104	Ip > -0.112, mean = -0.025
   plant 12, Ie > 0.101, mean = 0.104	Ip > -0.066, mean = 0.066
   plant 13, Ie > 0.101, mean = 0.104	Ip > -0.110, mean = 0.010
   plant 14, Ie > 0.101, mean = 0.104	Ip > -0.044, mean = 0.115
   plant 15, Ie > 0.101, mean = 0.104	Ip > -0.096, mean = -0.018
   plant 16, Ie > 0.101, mean = 0.104	Ip > -0.061, mean = 0.056
   plant 17, Ie > 0.101, mean = 0.104	Ip > -0.113, mean = 0.052
   plant 18, Ie > 0.101, mean = 0.104	Ip > -0.069, mean = 0.041
   plant 19, Ie > 0.101, mean = 0.104	Ip > -0.115, mean = 0.050

These results show that for the two white-flowered plants, the new
"phenotype model" yields approx. 2.3 nats (i.e. a ten-fold improvement
in the likelihood, by replacing the 10% "cost" per white flower in
the mixture model to a single 10% "cost" for the entire plant).
Note that even the purple-flowered plants yield strong empirical
information gain, reflecting the fact that a plant is either 
all-purple or all-white -- not a mixture as predicted by the mixture model.
The phenotype model has successfully converted all the potential 
information for these observations into empirical information.



