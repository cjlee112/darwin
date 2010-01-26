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

Creating State Objects
^^^^^^^^^^^^^^^^^^^^^^

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

The STOP State
^^^^^^^^^^^^^^

To specify how the path can exit the HMM, we create a special "STOP"
state::

    >>> stop = LinearStateStop()

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

Building the State Graph
^^^^^^^^^^^^^^^^^^^^^^^^

Next we just construct the :class:`model.StateGraph` specifying the
allowed transitions::

    >>> sg = StateGraph({F:{F:0.95, L:0.05, stop:1.}, 
                         L:{F:0.1, L:0.9, stop:1.}})

i.e. there is a 5% probability of transitioning from F to L, but
a 10% probability of transition back from L to F.  Note that the 
transitions to F and L sum to 100%.  Note also that under the 
*fixed-length sequence* model, the probability of transition to
*stop* is 0 until the **end** of the sequence, at which point it
becomes 100%.

Finally, to enable complete control over how the HMM is traversed, 
it's customary
to specify the *prior* probabilities of the states (i.e. for the very
first hidden state in the HMM) as a "transition probability" from the
"START" state.  In darwin we construct this as another state graph
that provides transitions from the special "START" state, which is just
specified using the string `'START'`::

    >>> prior = StateGraph({'START':{F:2./3., L:1./3.}})


Generating simulated observations
---------------------------------

We first construct our HMM as a composite of these three state graphs::

    >>> hmm = Model(NodeGraph({'theta':{'theta':sg}, 'START':{'theta':prior}}))

We have created a single variable labelled `theta` that follows
the states and transitions specified by our state graph `sg`.  
We also specified the `prior` transitions from `START` to `theta`.

We can now use the HMM's simulate_seq() method to generate a 
sequence of 100 states and observations via simulation::
    >>> n = 100
    >>> s,obs = hmm.simulate_seq(n)

Prior to performing Bayesian inference on this set of observations, 
we must transform them into a graph structure (in the standard
Pygr format of a dictionary of the form {srcLabel:{destLabel:obs}}
where srcLabel and destLabel are :class:`model.ObsLabel` and
obs is a tuple of one or more observations.  For a linear sequence
of observations, we can just use the convenience class
:class:`model.ObsSequence` to construct this graph for us::

    >>> obsGraph = ObsSequence(obs)

Bayesian inference on the HMM observations
------------------------------------------

Inference on the possible hidden states is computed using the 
*forward-backward algorithm*::

    >>> logPobs = hmm.calc_fb((obsGraph,))

This computes several things (stored as attributes on the
`hmm` object, in log-probability format,
as a dictionary whose keys are all possible hidden state values in
the HMM):

* **f**: represents the :math:`p(\vec{X}_1^{t-1},\Theta_t=s_i)` for a given
  hidden variable :math:`\Theta_t` to be in state :math:`s_i`.  

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

We simply compute this from our HMM's forward probabilities::

    >>> llDict = hmm.posterior_ll()

Printing out our results
------------------------

Let's just print out all our results::

    >>> for i in range(n): # print posteriors
    ...    obsLabel = obsGraph.get_label(i)
    ...    nodeLabel = hmm.graph.get_label('theta', (obsLabel,))
    ...    nodeF = Node(F, nodeLabel)
    ...    nodeL = Node(L, nodeLabel)
    ...    print '%s:%0.3f\t%s:%0.3f\tTRUE:%s,%d,%0.3f' % \
    ...          (nodeF, hmm.posterior(nodeF),
    ...           nodeL, hmm.posterior(nodeL),
    ...           s[i], obs[i], exp(llDict[nodeLabel][0]))
    ...
    <F: ('theta', (0,))>:0.931	<L: ('theta', (0,))>:0.069	TRUE:<F: ('theta', (0,))>,1,0.144
    <F: ('theta', (1,))>:0.953	<L: ('theta', (1,))>:0.047	TRUE:<F: ('theta', (1,))>,3,0.150
    <F: ('theta', (2,))>:0.965	<L: ('theta', (2,))>:0.035	TRUE:<F: ('theta', (2,))>,4,0.154
    <F: ('theta', (3,))>:0.970	<L: ('theta', (3,))>:0.030	TRUE:<F: ('theta', (3,))>,4,0.156
    <F: ('theta', (4,))>:0.972	<L: ('theta', (4,))>:0.028	TRUE:<F: ('theta', (4,))>,2,0.158
    <F: ('theta', (5,))>:0.970	<L: ('theta', (5,))>:0.030	TRUE:<F: ('theta', (5,))>,5,0.159
    <F: ('theta', (6,))>:0.964	<L: ('theta', (6,))>:0.036	TRUE:<F: ('theta', (6,))>,3,0.159
    <F: ('theta', (7,))>:0.953	<L: ('theta', (7,))>:0.047	TRUE:<F: ('theta', (7,))>,1,0.159
    <F: ('theta', (8,))>:0.930	<L: ('theta', (8,))>:0.070	TRUE:<F: ('theta', (8,))>,5,0.159
    <F: ('theta', (9,))>:0.890	<L: ('theta', (9,))>:0.110	TRUE:<F: ('theta', (9,))>,6,0.203
    <F: ('theta', (10,))>:0.912	<L: ('theta', (10,))>:0.088	TRUE:<F: ('theta', (10,))>,3,0.148
    <F: ('theta', (11,))>:0.918	<L: ('theta', (11,))>:0.082	TRUE:<F: ('theta', (11,))>,3,0.153
    <F: ('theta', (12,))>:0.909	<L: ('theta', (12,))>:0.091	TRUE:<F: ('theta', (12,))>,5,0.156
    <F: ('theta', (13,))>:0.883	<L: ('theta', (13,))>:0.117	TRUE:<F: ('theta', (13,))>,2,0.157
    <F: ('theta', (14,))>:0.831	<L: ('theta', (14,))>:0.169	TRUE:<F: ('theta', (14,))>,2,0.158
    <F: ('theta', (15,))>:0.733	<L: ('theta', (15,))>:0.267	TRUE:<F: ('theta', (15,))>,6,0.206
    <F: ('theta', (16,))>:0.730	<L: ('theta', (16,))>:0.270	TRUE:<F: ('theta', (16,))>,6,0.264
    <F: ('theta', (17,))>:0.817	<L: ('theta', (17,))>:0.183	TRUE:<F: ('theta', (17,))>,1,0.132
    <F: ('theta', (18,))>:0.861	<L: ('theta', (18,))>:0.139	TRUE:<F: ('theta', (18,))>,5,0.141
    <F: ('theta', (19,))>:0.879	<L: ('theta', (19,))>:0.121	TRUE:<F: ('theta', (19,))>,3,0.148
    <F: ('theta', (20,))>:0.877	<L: ('theta', (20,))>:0.123	TRUE:<F: ('theta', (20,))>,4,0.152
    <F: ('theta', (21,))>:0.854	<L: ('theta', (21,))>:0.146	TRUE:<F: ('theta', (21,))>,3,0.155
    <F: ('theta', (22,))>:0.802	<L: ('theta', (22,))>:0.198	TRUE:<F: ('theta', (22,))>,1,0.157
    <F: ('theta', (23,))>:0.703	<L: ('theta', (23,))>:0.297	TRUE:<F: ('theta', (23,))>,2,0.158
    <F: ('theta', (24,))>:0.521	<L: ('theta', (24,))>:0.479	TRUE:<F: ('theta', (24,))>,5,0.159
    <F: ('theta', (25,))>:0.193	<L: ('theta', (25,))>:0.807	TRUE:<L: ('theta', (25,))>,6,0.204
    <F: ('theta', (26,))>:0.079	<L: ('theta', (26,))>:0.921	TRUE:<L: ('theta', (26,))>,6,0.262
    <F: ('theta', (27,))>:0.041	<L: ('theta', (27,))>:0.959	TRUE:<L: ('theta', (27,))>,6,0.338
    <F: ('theta', (28,))>:0.033	<L: ('theta', (28,))>:0.967	TRUE:<L: ('theta', (28,))>,6,0.399
    <F: ('theta', (29,))>:0.043	<L: ('theta', (29,))>:0.957	TRUE:<L: ('theta', (29,))>,3,0.114
    <F: ('theta', (30,))>:0.025	<L: ('theta', (30,))>:0.975	TRUE:<L: ('theta', (30,))>,6,0.380
    <F: ('theta', (31,))>:0.023	<L: ('theta', (31,))>:0.977	TRUE:<L: ('theta', (31,))>,6,0.422
    <F: ('theta', (32,))>:0.036	<L: ('theta', (32,))>:0.964	TRUE:<L: ('theta', (32,))>,5,0.112
    <F: ('theta', (33,))>:0.022	<L: ('theta', (33,))>:0.978	TRUE:<L: ('theta', (33,))>,6,0.391
    <F: ('theta', (34,))>:0.022	<L: ('theta', (34,))>:0.978	TRUE:<L: ('theta', (34,))>,6,0.427
    <F: ('theta', (35,))>:0.036	<L: ('theta', (35,))>:0.964	TRUE:<L: ('theta', (35,))>,1,0.111
    <F: ('theta', (36,))>:0.023	<L: ('theta', (36,))>:0.977	TRUE:<L: ('theta', (36,))>,6,0.394
    <F: ('theta', (37,))>:0.024	<L: ('theta', (37,))>:0.976	TRUE:<L: ('theta', (37,))>,6,0.428
    <F: ('theta', (38,))>:0.041	<L: ('theta', (38,))>:0.959	TRUE:<L: ('theta', (38,))>,6,0.443

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

