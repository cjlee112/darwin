=============================
Basic HMM / Bayesian Modeling
=============================

The darwin :mod:`model` module supports simple HMM and Bayesian modeling
using a convenient class interface.  Let's start with a simple HMM
example, the Occasionally Dishonest Casino (see Durbin & Eddy).

Constructing a Simple HMM
-------------------------

A Hidden Markov Model is represented by its *state graph*.  In darwin,
this is simply modeled as a graph whose nodes are states, and whose
edge values represent the transition probability for going from a source
state to a destination state.  Each state has an associated *emission
dictionary* whose keys are its possible emission values (observations),
and associated values are the likelihood of each possible observation.

The Occasionally Dishonest Casino has just two states:

* **F**: a fair dice that emits all possible rolls with equal probability.

* **L**: a loaded dice that emits a six with 50% probability, and
  the remaining values each with 10% probability.

Let's create them::

    >>> from darwin.model import *
    >>> p6 = 0.5
    >>> p = (1. - p6) / 5.
    >>> L = LinearState('L', EmissionDict({1:p, 2:p, 3:p, 4:p, 5:p, 6:p6}))
    >>> p = 1. / 6.
    >>> F = LinearState('F', EmissionDict({1:p, 2:p, 3:p, 4:p, 5:p, 6:p}))

:class:`model.LinearState` models a process that emits a linear
chain of observations (i.e. a sequence of states each emitting one
observation).  Note that its first argument is just the name assigned
to the state.

Next we just construct the :class:`model.StateGraph` specifying the
allowed transitions::

    >>> sg = StateGraph({F:{F:0.95, L:0.05}, L:{F:0.1, L:0.9}})

i.e. there is a 5% probability of transitioning from F to L, but
a 10% probability of transition back from L to F.

To enable complete control over how the HMM is traversed, it's customary
to specify the *prior* probabilities of the states (i.e. for the very
first hidden state in the HMM) as a "transition probability" from the
"START" state.  In darwin we construct this as another state graph
that provides transitions from the special "START" state, which is just
specified using the string `'START'`::

    >>> prior = StateGraph({'START':{F:2./3., L:1./3.}})

Finally, we specify how the path can exit the HMM, by providing a state
graph with transitions to the special "STOP" state::

    >>> stop = LinearStateStop()
    >>> term = StateGraph({F:{stop:1.}, L:{stop:1.}})

"STOP" transitions generally follow one of two possible patterns:

* for fixed-length sequences: the set of possible observations is simply
  the set of all possible observation sequences of length *n*.  In
  this case, we use the :class:`model.LinearStateStop` end state,
  which will only transition to "STOP" when the observation sequence
  has been exhausted (i.e. after *n* steps).  This transition is 
  assigned 100% probability, i.e. once the observations are exhausted,
  the *only* thing the HMM can do is exit.

* for variable-length sequences: the set of possible observations is
  the set of all possible observation sequences that can be emitted
  by the HMM (possibly including observation sequences of unbounded
  length).  In this case the probability of transitioning to "STOP"
  in any given step should be a small probability, which must be
  included in the normalization of all possible outgoing edges
  from any given state (transition probabilities from any given
  state must sum to one!).

We follow the first model here.

Generating simulated observations
---------------------------------

We first construct our HMM as a composite of these three state graphs::

    >>> dg = BasicHMM(sg, prior, term)

We use its simulate_seq() method to generate a sequence of states
and obserations via simulation::
    >>> n = 10
    >>> s,obs = dg.simulate_seq(n)

Prior to performing Bayesian inference on this set of observations, 
we must index our observations for fast look-up::

    >>> obsDict = obs_sequence(0, obs)

Bayesian inference on the HMM observations
------------------------------------------

Inference on the possible hidden states is computed using the 
*forward-backward algorithm*::

    >>> f, b, fsub, bsub, ll = dg.calc_fb(obsDict)
    >>> logPobs = b[START]

This computes several things (stored in log-probability format,
as a dictionary whose keys are all possible hidden state values in
the HMM):

* **f**: represents the :math:`p(\vec{X}_1^{t-1},\Theta_t=s_i)` for a given
  hidden state :math:`\Theta_t` to be in state :math:`s_i`.  

* **b**: represents the :math:`p(\vec{X}_{t+1}^n|\Theta_t=s_i)`,
  i.e. all observations emitted by "descendants" of this node.  Since
  the special state `START` is at the beginning of the entire HMM,
  `b[START]` simply gives the total probability of the observations
  summed over all possible paths.

* **ll**: stores the log-likelihood for observations from a given hidden
  state, i.e. :math:`p(X_t|\Theta_t=s_i)`

* **fsub** represents the probability of all observations *not* emitted
  by descendants of this node.  Note that **fsub**
  and **b** represent a disjoint division of the set of all
  possible observations, and so can be used to directly calculate
  posterior probabilities for any state given all the observations.

* **bsub** is meaningful mainly for branched (non-linear) 
  model structures, so we will not discuss it further in this example.


Posterior Likelihoods of the Observations
-----------------------------------------

The crucial parameter for assessing the predictive power of a model
is the *posterior likelihood*, which predicts the probability of a
given observation properly taking into account both the model and
all *previous* observations.  Note that this is computed over all
possible hidden states that could have emitted this observation.

.. math:: p(X_t|\vec{X}_1^{t-1})=\frac{\sum_i{p(\vec{X}_1^t,\Theta_t=s_i)}}{\sum_i{p(\vec{X}_1^{t-1},\Theta_t=s_i)}}

We simply compute this from the forward probabilities::

    >>> llDict = posterior_ll(f)

Printing out our results
------------------------

Let's just print out all our results::

    >>> for i in range(n): # print posteriors
    ...    nodeF = Node(F, 0, (i,))
    ...    nodeL = Node(L, 0, (i,))
    ...    print '%s:%0.3f\t%s:%0.3f\tTRUE:%s,%d,%0.3f' % \
    ...          (nodeF, exp(fsub[nodeF] + b[nodeF] - logPobs),
    ...           nodeL, exp(fsub[nodeL] + b[nodeL] - logPobs),
    ...           s[i], obs[i], exp(llDict[nodeF.get_obs_label(i)][0]))
    ...
    <F: (0,) (0, 0)>:0.324	<L: (0,) (0, 0)>:0.676	TRUE:<F: (0,) (0, 0)>,2,0.144
    <F: (1,) (0, 0)>:0.206	<L: (1,) (0, 0)>:0.794	TRUE:<F: (1,) (0, 0)>,6,0.249
    <F: (2,) (0, 0)>:0.178	<L: (2,) (0, 0)>:0.822	TRUE:<F: (2,) (0, 0)>,6,0.324
    <F: (3,) (0, 0)>:0.208	<L: (3,) (0, 0)>:0.792	TRUE:<F: (3,) (0, 0)>,6,0.389
    <F: (4,) (0, 0)>:0.330	<L: (4,) (0, 0)>:0.670	TRUE:<F: (4,) (0, 0)>,5,0.115
    <F: (5,) (0, 0)>:0.384	<L: (5,) (0, 0)>:0.616	TRUE:<F: (5,) (0, 0)>,6,0.376
    <F: (6,) (0, 0)>:0.581	<L: (6,) (0, 0)>:0.419	TRUE:<F: (6,) (0, 0)>,2,0.116
    <F: (7,) (0, 0)>:0.686	<L: (7,) (0, 0)>:0.314	TRUE:<F: (7,) (0, 0)>,3,0.126
    <F: (8,) (0, 0)>:0.737	<L: (8,) (0, 0)>:0.263	TRUE:<F: (8,) (0, 0)>,3,0.136
    <F: (9,) (0, 0)>:0.750	<L: (9,) (0, 0)>:0.250	TRUE:<F: (9,) (0, 0)>,2,0.144
    <F: (10,) (0, 0)>:0.731	<L: (10,) (0, 0)>:0.269	TRUE:<F: (10,) (0, 0)>,5,0.150
    <F: (11,) (0, 0)>:0.674	<L: (11,) (0, 0)>:0.326	TRUE:<F: (11,) (0, 0)>,5,0.154
    <F: (12,) (0, 0)>:0.557	<L: (12,) (0, 0)>:0.443	TRUE:<F: (12,) (0, 0)>,6,0.218
    <F: (13,) (0, 0)>:0.545	<L: (13,) (0, 0)>:0.455	TRUE:<F: (13,) (0, 0)>,6,0.284
    <F: (14,) (0, 0)>:0.624	<L: (14,) (0, 0)>:0.376	TRUE:<L: (14,) (0, 0)>,2,0.128
    <F: (15,) (0, 0)>:0.653	<L: (15,) (0, 0)>:0.347	TRUE:<L: (15,) (0, 0)>,5,0.138
    <F: (16,) (0, 0)>:0.644	<L: (16,) (0, 0)>:0.356	TRUE:<L: (16,) (0, 0)>,3,0.146
    <F: (17,) (0, 0)>:0.593	<L: (17,) (0, 0)>:0.407	TRUE:<L: (17,) (0, 0)>,6,0.245
    <F: (18,) (0, 0)>:0.628	<L: (18,) (0, 0)>:0.372	TRUE:<L: (18,) (0, 0)>,4,0.136
    <F: (19,) (0, 0)>:0.622	<L: (19,) (0, 0)>:0.378	TRUE:<F: (19,) (0, 0)>,2,0.144
    <F: (20,) (0, 0)>:0.573	<L: (20,) (0, 0)>:0.427	TRUE:<L: (20,) (0, 0)>,5,0.150
    <F: (21,) (0, 0)>:0.466	<L: (21,) (0, 0)>:0.534	TRUE:<F: (21,) (0, 0)>,6,0.230
    <F: (22,) (0, 0)>:0.457	<L: (22,) (0, 0)>:0.543	TRUE:<F: (22,) (0, 0)>,6,0.300
    <F: (23,) (0, 0)>:0.539	<L: (23,) (0, 0)>:0.461	TRUE:<F: (23,) (0, 0)>,5,0.125
    <F: (24,) (0, 0)>:0.569	<L: (24,) (0, 0)>:0.431	TRUE:<F: (24,) (0, 0)>,6,0.323
    <F: (25,) (0, 0)>:0.719	<L: (25,) (0, 0)>:0.281	TRUE:<F: (25,) (0, 0)>,4,0.122
    <F: (26,) (0, 0)>:0.800	<L: (26,) (0, 0)>:0.200	TRUE:<F: (26,) (0, 0)>,3,0.132
    <F: (27,) (0, 0)>:0.840	<L: (27,) (0, 0)>:0.160	TRUE:<F: (27,) (0, 0)>,1,0.141
    <F: (28,) (0, 0)>:0.852	<L: (28,) (0, 0)>:0.148	TRUE:<F: (28,) (0, 0)>,4,0.148
    <F: (29,) (0, 0)>:0.842	<L: (29,) (0, 0)>:0.158	TRUE:<F: (29,) (0, 0)>,3,0.153
    <F: (30,) (0, 0)>:0.804	<L: (30,) (0, 0)>:0.196	TRUE:<F: (30,) (0, 0)>,2,0.156

This example illustrates several points:

* To query our results, we construct a :class:`model.Node` representing
  a particular hidden state emitting a specific observation (given by
  the observation index *i*).  Note that its second argument specifies
  the ID of the state graph containing this hidden state (in this case
  just the first state graph, with default index 0).

* The posterior probability for each state is given via the standard
  computation

.. math:: p(\Theta_t=s_i|\vec{X}_1^n) = \frac{p(\vec{X}_1^t,\Theta_t=s_i)p(\vec{X}_{t+1}^n|\Theta_t=s_i)}{p(\vec{X}_1^n)}

* The posterior likelihood of a given observation varies depending
  on what hidden state the model thinks is most likely at that point,
  based on the previous observations.  For example, the posterior
  likelihood of the observed sixes ranges from 0.167 (when the model
  is confident of the F state) to 0.5 (when the model is confident of
  the L state).

