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
  to another.

* an *unbranched segment* is a sequence of variables that form a Markov
  chain (i.e. these variables have neither multiple incoming nor multiple
  outgoing edges with each other).

* a *segment graph* is a graph whose nodes are unbranched segments, and
  whose edges have either multiple-conditions or multiple-dependents.

* a *loop* is the existence of multiple paths from an origin segment to
  a destination segment.  By definition, a loop begins at a multiple-
  dependents branching, and closes at a multiple-conditions junction.

Interfaces
----------

.. class:: DependencyGraph(graph, multiCondSet=None)

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
   converted automatically into a :class:`MultiCondition` object
   and registered on the *multiCondSet* object.

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

.. class:: ObsSet(name)

   Creates a container for tagged observations; each observation
   can be tagged with one or more *tag=value* bindings.

.. method:: ObsSet.add_obs(values, **tags)

   add a list of observations *values* with kwargs key=value *tags*.


.. class:: Model(dependencyGraph, obsLabel, logPmin=neginf, multiCondSet=None)

   Top-level interface for computing the posterior likelihood
   of a set of observations on a dependency graph.

   Any state with observation likelihood less than or equal to *logPmin*
   will truncate a path.  Its default value simply truncates
   zero-probability paths.

   If your *dependencyGraph* contains multiple-condition edges,
   you must supply the corresponding *multiCondSet* object that 
   stores them.

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


