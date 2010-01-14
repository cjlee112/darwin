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

Next we create independent phenotype variables for each of the 20 plants
we are modeling::

   >>> d = {}
   >>> for plant in range(20):
   ...    d[plant] = prior
   ...

We assemble these into the final *dependency graph* that shows the structure
of these variables; here we merely draw them as a star-topology from the 
initial 'START' state::

   >>> dg = DependencyGraph({'START':{0:{0:d}},
   ...                       0:{'STOP':TrivialMap({0:term})}})
   ...

Finally, we package each plant's observations in an *observation dictionary*
keyed by the possible plant IDs::

   >>> obsDict = {}
   >>> for plant in range(2): # two white plants
   >>>    obsDict[(0,plant,0)] = emitWh.rvs(100)
   ...
   >>> for plant in range(2, 20): # 18 purple plants
   >>>    obsDict[(0,plant,0)] = emitPu.rvs(100)
   ...

We now compute the model probabilities using the forward-backward
algorithm::

   >>> f, b, fsub, bsub, ll = dg.calc_fb(obsDict)
   >>> logPobs = b[START]

This gives us the total log-probability of the entire set of observations.
We can also compute the posterior likelihood of each of the observations::

   >>> llDict = posterior_ll(f)

This analyzes the likelihood of each observation conditioned on all 
previous observations.  For example, once the model sees several white
flowers from one plant, it will predict that future flowers from that
plant will probably be white as well.


