===========================================
:mod:`model` --- Information graph modeling
===========================================

.. module:: model
   :synopsis: Information graph modeling
.. moduleauthor:: Christopher Lee <leec@chem.ucla.edu>
.. sectionauthor:: Christopher Lee <leec@chem.ucla.edu>

This module provides a basic framework for modeling general information
graphs, which can have any conditioning structure (not just Markov 
chains).

Definitions
-----------

* A *variable* is a disjoint slicing of probability space into one or
  more distinct *states*.  

* A *dependency graph* is a graph whose nodes are variables, with
  directed edges connecting pairs of nodes.  It shows the dependency
  structure of the variables, as follows:

* A *Markov edge* is a relation between two variables asserting that
  the probability of one variable depends (is conditional) *only*
  on the other variable.  This is represented as a node with a *single*
  incoming edge.

* A variable with multiple incoming edges is said to have
  *multiple-conditions* i.e. its probability depends on multiple variables.

* A variable with multiple outgoing edges (either Markov and / or
  multiple-conditions) is said to have *multiple-dependents*.
  In other words, multiple variables depend on this variable.
  (And if they each depend *only* on this variable, then they are
  conditionally independent given this variable).

* a *state graph* is a graph whose nodes are individual states
  and whose edges are the probabilities of a transition from one state
  to another.  In Darwin you can combine any number of separate state
  graphs to represent the behavior of different sets of variables.

* a *compiled observation-state graph* is a graph whose nodes are
  individual states of a specified variable emitting a particular
  set of observations.  This graph is produced by compiling the basic
  dependency graph combined with all their possible state transitions
  applied to the specified observation set.  All analyses (posterior
  probability of states; posterior likelihood of observations) are
  performed on this graph.

* an *unbranched segment* is a sequence of variables in the
  compiled observation-state graph that form a Markov
  chain (i.e. these variables have neither multiple incoming nor multiple
  outgoing edges with each other).

* a *segment graph* is a graph whose nodes are unbranched segments, and
  whose edges have either multiple-conditions or multiple-dependents.

* a *loop* is the existence of multiple paths from an origin segment to
  a destination segment.  By definition, a loop begins at a multiple-
  dependents branching, and closes at a multiple-conditions junction.

Interfaces
----------

Basic Data Types
................

.. class:: Variable(graph, label, obsLabel=None, parent=None)

   *graph* should be the :class:`DependencyGraph` that this variable 
   is part of.  *label* should be its unique identifier in that 
   dependency graph (customarily a text label provided as a string).

   Note that in Markov chains, each step in the chain is a separate
   variable.  For a homogeneous Markov chain, these are separate
   instances of the same variable type (i.e. the same set of possible
   states) but emit different observations (from an *observation sequence*).
   :class:`Variable` acommodates this by binding an observation label
   *obsLabel* (specifying what observation(s) are emitted by this variable)
   as part of its unique identifier.  In other words, two 
   :class:`Variable` objects with the same *label* but different
   *obsLabel* values are treated as different variables.

   *parent* identifies what subgraph this variable is part of.

Variable objects have the following attributes:

.. attribute:: Variable.label

   The "name" of the variable, typically a string.

.. attribute:: Variable.obsLabel

   Represents information about the "current state of observations"
   at this point in the model, including both
   
   * what observation(s) if any are emitted by this node.

   * "where we currently are" in the observation set, e.g. the current
     position in an observation sequence; or what tags have been used
     so far to subset the total observation set.

.. attribute:: Variable.graph

   What :class:`DependencyGraph` this variable is part of.


.. class:: Node(state, var)

   Represents a single variable-observation-state in the compiled
   observation-state graph.

A node object has the following attributes:

.. attribute:: Node.var

   The :class:`Variable` that this node represents.

.. attribute:: Node.state

   The :class:`State` that this node represents.


Building Dependency Structures
..............................

.. class:: DependencyGraph(graph)

   Represents the dependency structure of a graphical model consisting of
   one or more variables linked by edges representing dependency relations.
   *graph* should follow the Pygr convention for representing graphs as
   dictionary (dict-like) objects whose keys are source nodes, and whose
   associated values are dictionaries, whose keys are destination nodes
   and whose associated values are edge objects, i.e. 
   in the form `{source:{dest:edge}}`.

   A node can be specified simply as a text label (string), or as a
   :class:`Variable` object.

   Each edge must be a :class:`StateGraph` object or equivalent interface.

   If a source node object is a tuple, it will be treated as a set of
   multiple conditions.  The values in the tuple can either be text
   labels (strings) or :class:`Variable` objects.  They specify the
   list of variables that the destination variable depends on.
   Each *edge* must therefore be a state graph object that takes a tuple
   of multiple variable-states as a key (instead of a single state object,
   as is the standard case for a Markov edge).  The *source* tuple is
   converted automatically into a :class:`MultiCondition` object.

   If a destination node object is *callable*, it will be treated as a
   generator of multiple destination nodes.  Specifically, it will be called
   as `dest(source, **edge)`, where *source* is the :class:`Node` object
   representing the source state, and *edge* is value associated with
   *dest* in the *graph*.  This allows it to generate an appropriate
   set of destination nodes customized to a specific *source*.
   This call must return a dictionary whose keys are destination
   labels (i.e. a string; or a :class:`Variable` object), and
   whose associated values are state graph objects.  For an example
   of such a generator, see :class:`BranchGenerator`.

.. class:: BranchGenerator(label, stateGraph, iterTag=None, **tags)

   A callable generator of multiple destination :class:`Variable`,
   using the specified *iterTag*.  All values of *iterTag* in the 
   observation set will be generated as separate variables each with
   that subset of observation(s).  *tags*, if provided, is used to
   pre-filter the observation set *prior* to generating the *iterTag* subsets.
   Each variable will be created with the specified *label* and
   associated *stateGraph*.  Note that since each :class:`Variable`
   is bound to a distinct set of observations, they are treated as
   different variables (even though they share the same *label* value).


Building State - Transition Structures
......................................

Each :class:`Variable` has one or more possible *states*, and a 
single edge from one variable to another typically consists of many
possible state-to-state transitions with associated transition 
probabilities.  These are represented by two kinds of classes:

* *state-graph classes*: a :class:`StateGraph` object acts as a 
  function that produces a dictionary of all possible destination states
  (given an origin state), along with their associated transition
  probabilities.  It could either represent a simple Markov edge
  or a multiple-condition relation (in which one target variable depends
  on multiple source variables).  Thus the real content of a state
  graph is that it controls what states can be reached from what
  origin states, with what probability.

.. class:: StateGraph(graph)

   Generic state graph for Markov transitions.  *graph* must be a 
   standard dictionary-representation of a graph, whose nodes are state
   objects and edges are transition probabilities connecting allowed state
   transitions.

   You can write your own state graph classes; all they need to do is
   provide the following call interface:

.. method:: StateGraph.__call__(sources, targetVar, state=None, parent=None)

   For Markov edges *sources* is simply the origin :class:`Node` object
   for which we must generate the set of possible destination nodes.

   For multi-condition edges, *sources* is a tuple of :class:`Node`
   objects to be used as the condition for generating a set of destination
   nodes.

   *targetVar* is the generic label for the destination variable, i.e.
   without the final *obsLabel* (which will be added by the individual
   :class:`State` calls).

   *state* is optional information that can be ignored at the moment.

   *parent* is the :class:`Node` containing this subgraph, if any.
   This argument should simply be passed to the :class:`State` calls.

   This call must return a dictionary whose keys are destination 
   :class:`Node` objects and whose associated values are their
   transition probabilities.  These :class:`Node` objects should be
   generated by calling whatever set of :class:`State` objects are
   allowed transitions from this origin state.

* *state classes*: a :class:`State` object acts as a function that produces a 
  new node in the compiled observation-state graph
  representing that *state* of a particular
  *variable* emitting specific *observations*.  In other words
  it plays the most basic role during compiling the observation-state
  graph of adding one more node to the graph.  In so doing it mainly has
  control over what observation(s) to bind to the new node
  (typically based on what observation(s) were bound to the source
  node, and what "move" in observation-space the state corresponds to.
  For example, for a state in a Markov chain, the move is simply to
  take the next observation in the observation sequence).

.. class:: State(name, emission)

   *name* is simply used to identify the state.  The special names
   `'START'` and `'STOP'` identify the beginning and end points of
   a graph or subgraph.

   *emission* must be a dictionary-like object that takes observation
   values as keys, and returns their associated emission probabilities.

To create your own subclass of State, you should supply your own
version of its call interface:

.. method:: State.__call__(fromNode, targetVar, obsLabel, edge, parent)

   returns a new :class:`Node` representing the specified *targetVar*
   :class:`Variable` bound to the appropriate observation(s) for this
   state, derived from *obsLabel* via whatever "move algorithm" is 
   appropriate for this kind of state.

   Additional information is provided to the function as optional data
   that may be helpful for you:

   * *fromNode*: the origin node of this transition

   * *edge*: the transition probability of this transition

   * *parent*: the :class:`Node` containing this subgraph, if any.
     This *must* be used to construct the result :class:`Node`.

**Example state subclasses**

.. class:: LinearState(name, emission)

   Selects the next observation from the observation sequence
   (appropriate for a Markov chain).

.. class:: VarFilterState(name, emission)

   Selects observation(s) tagged with `var=value` where *value* must
   match the name of the current :class:`Variable`.

.. class:: SilentState(name)

   State that emits no observations.

.. class:: StopState(useObsLabel=True)

   Terminates the path and marks it as a valid path for probability
   calculation.  Note that any path that does not terminate at StopState
   is excluded from all probability calculations.

   If *useObsLabel* is True, its obsLabel will be the *obsLabel* it receives
   (but of course it *emits* no observation, just like SilentState).

.. class:: LinearStateStop(name, emission)

   Only returns :class:`StopState` if the observation sequence is exhausted.


Storing Observations
....................

Currently support is provided for two different ways of matching
observations to variables in a model:

* *an observation sequence*: for Markov chain models.  Use 
  :class:`ObsSequenceLabel` as the observation container and
  :class:`LinearState` as the state type (it calls obsLabel.get_next()
  to obtain the next observation in the sequence.

* *tagged observations*: each observation can be tagged with one or 
  more *key=value* pairs.  Each variable or state can then select
  its observations by filtering on specified tag values.
  Use :class:`ObsSet` as the observation container, and 
  :class:`BranchGenerator` to generate multiple branches for different
  values of a given tag, or :class:`VarFilterState` to select 
  observations tagged to match the name of the current variable.

.. class:: ObsSequenceLabel(seq)

   Creates a container for an observation sequence.  *seq* must 
   support the sequence protocol, specifically `len(seq)` and slicing
   `seq[i:j]`.

.. class:: ObsSet(name)

   Creates a container for tagged observations; each observation
   can be tagged with one or more *tag=value* bindings.

.. method:: ObsSet.add_obs(values, **tags)

   add a list of observations *values* with kwargs key=value *tags*.


Performing Analyses
...................

The :class:`Model` class is the top-level interface for compiling
the model and running analyses on it in conjunction with a specific 
set of observations.

.. class:: Model(dependencyGraph, obsLabel, logPmin=neginf)

   Top-level interface for computing the posterior likelihood
   of a set of observations on a dependency graph.

   Any state with observation likelihood less than or equal to *logPmin*
   will truncate a path.  Its default value simply truncates
   zero-probability paths.

   Creating a :class:`Model` instance compiles the complete
   state graph implied by the :class:`DependencyGraph` (which may
   invoke subgraph compilation), as applied to the specific
   set of observations provided by *obsLabel*.

.. method:: Model.calc_fb()

   Performs the forward-backward algorithm to compute the posterior
   probability of all states, and the posterior likelihood of all
   observations.

.. method:: Model.save_graphviz(filename, **kwargs)

   save a graphviz visualization of the compiled state graph to 
   the specified file path, passing
   *kwargs* to the :func:`save_graphviz()` function.  Requires the
   **gvgen** package.

Internal Interfaces
...................

Users don't normally need to create these classes themselves.

.. class:: MultipleCondition(conditions, targetVar, stateGraph)

   Represents a multiple-conditions dependency, and generates the
   combinatorial set of edges associated with the possible condition states.

   *conditions* must be the list of variables that *targetVar* depends on.
   *stateGraph* must be a callable object that is called as:
   `stateGraph(vec, targetVar, parent=parent)`, where vec is a tuple of
   states (one for each variable in the *conditions*), and *parent* is
   the current subgraph being compiled.  It must return a dictionary
   whose keys are possible states of *targetVar* and whose associated
   values are the probability of each state conditioned on *vec*.


