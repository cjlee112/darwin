import UserDict
from math import *
import random
from sys import maxint
try:
    import numpy
except ImportError:
    pass

# log-probability convenience functions

neginf = float('-inf') # IEEE754 negative infinity

def log_sum(logX, logY):
    'return log of sum of two arguments passed in log-form'
    if logX < logY:
        logX,logY = logY,logX
    if logY == neginf:
        return logX
    return logX + log(1. + exp(logY - logX))

def log_sum_list(logList):
    'return log of sum of arguments passed in log-form'
    if len(logList) == 1: # handle trivial case
        return logList[0]
    try: # use numpy array methods
        top = logList.max()
        return top + log(numpy.exp(logList - top).sum())
    except (AttributeError,NameError): # use regular Python math
        pass
    top = max(logList)
    return top + log(sum([exp(x - top) for x in logList]))
    
def safe_log(x):
    "Doesn't blow up like math.log(0.) does"
    if x==0.:
        return neginf
    return log(x)

# basic classes for dependency - observation - state graphs

class Node(object):
    '''Node class for compiled state-instance graphs.  Note that
    multiple Node instances with the same state, ruleID, obsLabel will
    compare and hash as equal.  This enables different paths to arrive
    at the same node even though they construct different object
    instances -- the different instances will compare as equal when
    looking them up in the forward - backward dictionaries.'''
    def __init__(self, state, var):
        self.state = state
        self.var = var

    def get_children(self, fromNode=None, parent=None):
        '''Get descendants in form [{node:pTransition}] '''
        results = []
        if fromNode is None:
            fromNode = self
        for varLabel, stateGraph in self.var.get_children().items():
            results.append(stateGraph(fromNode, varLabel, state=self.state, parent=parent))
        return results

    def __hash__(self):
        return hash((self.state,self.var))

    def __cmp__(self, other):
        try:
            return cmp((self.state,self.var),
                       (other.state,other.var))
        except AttributeError:
            return cmp(id(self), id(other))

    def __repr__(self):
        return '<%s: %s>' % (repr(self.state), repr(self.var))

    def get_ll(self):
        try:
            f = self.state.get_ll
        except AttributeError: # START and STOP lack this method
            return ()
        return f(self)

    def log_p_obs(self):
        'compute total log-likelihood for all obs emitted by this node'
        return sum(self.get_ll())


class StartNode(Node):
    def __init__(self, graph, obsLabel, parent=None):
        label = graph.get_start(obsLabel=obsLabel, parent=parent)
        Node.__init__(self, 'START', label)
        self.isub = 'START' # dummy value


class StopNode(Node):
    def __init__(self, graph, obsLabel=None, parent=None):
        Node.__init__(self, 'STOP', NodeLabel(graph, 'STOP', obsLabel, parent))
        self.isub = 'STOP' # dummy value


class NodeLabel(object):
    '''Reference to a specific vertex in a graph'''
    def __init__(self, graph, label, obsLabel=None, parent=None):
        self.graph = graph
        self.label = label
        self.obsLabel = obsLabel
        self.parent = parent

    def __hash__(self):
        return hash((self.graph, self.label, self.obsLabel, id(self.parent)))

    def __cmp__(self, other):
        try:
            return cmp((self.graph, self.label, self.obsLabel, id(self.parent)),
                  (other.graph, other.label, other.obsLabel, id(other.parent)))
        except AttributeError:
            return cmp(id(self), id(other))

    def __repr__(self):
        return str((self.label, self.obsLabel))

    def get_obs_label(self, obsLabel, parent=None):
        if parent is None:
            parent = self.parent
        return self.__class__(self.graph, self.label, obsLabel, parent)

    def get_children(self, **kwargs):
        '''Get child,obs pairs for child nodes of this node '''
        return self.graph.__getitem__(self, **kwargs)

    def __len__(self):
        'Get count of descendants'
        return len(self.graph[self])

class DependencyGraph(object):
    '''Graph of the form {source:{target:edge}}
    where source, target are NodeLabel objects, and
    edge is the state graph specifying the state transitions from source
    to target.'''
    labelClass = NodeLabel

    def __init__(self, graph):
        '''graph should be dict of {sourceLabel:{destLabel:stateGraph}}
        edges.  destLabel can be a NodeLabel object (allowing you to
        specify a cross-connection to a node in another graph),
        or simply any Python value, in which case it will be treated
        as label for creating a NodeLabel bound to this LabelGraph.'''
        self.graph = graph

    def __getitem__(self, node):
        'get dict of {target NodeLabel:stateGraph} pairs'
        try:
            if node.graph is not self:
                raise KeyError
            d = self.graph[node.label]
        except (KeyError, AttributeError):
            raise KeyError('node not in this graph')
        if callable(d): # treat as function for generating target dict
            d = d(node)
        results = {}
        for label,edge in d.items():
            if not isinstance(label, self.labelClass):
                label = self.get_node(label)
            results[label] = edge
        return results

    def get_start(self, **kwargs):
        'get START node for this graph'
        return self.get_node('START', **kwargs)

    def get_node(self, label, *args, **kwargs):
        'construct label object with specified args'
        return self.labelClass(self, label, *args, **kwargs)

    def __hash__(self):
        return id(self)

    def simulate_seq(self, n):
        'simulate markov chain of length n'
        node = StartNode(self, obsLabel=ObsSequenceSimulator())
        s = []
        obs = []
        for i in range(n):
            p = random.random()
            total = 0.
            for stateGroup in node.get_children():
                for dest,edge in stateGroup.items(): # choose next state
                    if dest.state == 'STOP':
                        continue
                    total += edge
                    if p <= total:
                        break
                break # this algorithm can only handle linear chain ...
            s.append(dest)
            obs.append(dest.var.obsLabel.get_obs())
            node = dest
        return s,tuple(obs)


class ObsExhaustedError(IndexError):
    pass

class ObsSequenceLabel(object):
    'simple label for iterating over sequence'
    def __init__(self, seq, start= 0, length=0, label=None):
        self.seq = seq
        self.start = start
        self.stop = start + length
        if self.stop > len(self.seq):
            raise ObsExhaustedError('no more obs in sequence')
        self.label = label
    def get_obs(self):
        if self.stop - self.start == 1:
            return self.seq[self.start]
        else:
            return self.seq[self.start : self.stop]
    def get_next(self, empty=False, length=1,**kwargs):
        if empty:
            length = 0
        return self.__class__(self.seq, self.stop, length, self.label)
    def __repr__(self):
        if self.label is not None:
            return str(self.label) + ':' + str(self.start)
        else:
            return str(self.start)
    def __hash__(self):
        return hash((self.label, self.start, self.stop))
    def __cmp__(self, other):
        try:
            return cmp((self.label, self.start, self.stop),
                       (other.label, other.start, other.stop))
        except AttributeError:
            return cmp(id(self), id(other))


class ObsSet(object):
    'container for tagged observations'
    def __init__(self, name):
        'name must be unique within a given model'
        self.name = name
        self._obs = []
        self._tags= {}

    def add_obs(self, values, **tags):
        'add list of observations [values] with kwargs key=value tags'
        iobs = len(self._obs)
        for k,v in tags.items():
            self._tags.setdefault(k, {}).setdefault(v, set()).add(iobs)
        self._obs.append(values)

    def get_subset(self, **tags):
        'select subset that matches tag key=value constraints'
        return ObsSubset(self, **tags)

    def get_tag_dict(self, tag):
        'get dict of {k:ObsSubset(tag=k)} subsets of this set'
        d = {}
        for k in self._tags[tag]:
            d[k] = self.get_subset(**{tag:k})
        return d

    def get_next(self, **kwargs): # dummy method for LinearState compatibility
        return self

    def __hash__(self):
        return hash(self.name)

    def __cmp__(self, other):
        try:
            return cmp(self.name, other.name)
        except AttributeError:
            return cmp(id(self), id(other))

    def __repr__(self):
        return 'ObsSet(%s)' % self.name


class ObsSubset(object):
    def __init__(self, obsSet, **tags):
        self.obsSet = obsSet
        self.tags = tuple(tags.items())

    def get_obs(self):
        for k,v in self.tags:
            try:
                subset = subset.intersection(self.obsSet._tags[k][v])
            except NameError:
                subset = self.obsSet._tags[k][v]
        for iobs in subset:
            try:
                results = numpy.concatenate((results, self.obsSet._obs[iobs]))
            except NameError:
                results = self.obsSet._obs[iobs]
        return results

    def get_subset(self, **tags):
        'select subset that matches (additional) tag key=value constraints'
        return self.__class__(self.obsSet, self.tags + tags.items())

    def get_next(self, **kwargs): # dummy method for LinearState compatibility
        return self

    def get_tag_dict(self, tag):
        'get dict of {k:ObsSubset(tag=k)} subsets of this subset'
        d = {}
        for k in self.obsSet._tags[tag]:
            d[k] = self.get_subset(**{tag:k})
        return d

    def __hash__(self):
        return hash((self.obsSet, self.tags))

    def __cmp__(self, other):
        try:
            return cmp((self.obsSet, self.tags),
                       (other.obsSet, other.tags))
        except AttributeError:
            return cmp(id(self), id(other))

    def __repr__(self):
        return 'ObsSet(%s, %s)' % (self.obsSet.name,
                  ', '.join([('%s=%s' % t) for t in self.tags]))


class BranchGenerator(object):
    def __init__(self, label, stateGraph, iterTag=None, **tags):
        self.label = label
        self.stateGraph = stateGraph
        self.iterTag = iterTag
        self.tags = tags

    def __call__(self, nodeLabel):
        'return {NodeLabel:stateGraph} dict'
        d = {}
        subset = nodeLabel.obsLabel # assume obsLabel iterable
        if self.tags: # filter using these tag constraints
            subset = subset.get_subset(** self.tags)
        if self.iterTag: # generate branches for each value of this tag
            subset = subset.get_tag_dict(self.iterTag).values()
        for obsLabel in subset:
            try: # already a NodeLabel
                newLabel = self.label.get_obs_label(obsLabel)
            except AttributeError: # need to create a new NodeLabel
                newLabel = nodeLabel.graph.get_node(self.label)
                newLabel.obsLabel = obsLabel
            d[newLabel] = self.stateGraph
        return d

class EndlessSeq(object):
    'dummy object reports length of infinity'
    def __len__(self):
        return maxint

class ObsSequenceSimulator(ObsSequenceLabel):
    def __init__(self, seq=None, state=None, start=0, length=0):
        if seq is None:
            seq = EndlessSeq() # no limit on how long simulation can run
        if state is None: # doesn't emit anything
            length = 0
        ObsSequenceLabel.__init__(self, seq, start, length)
        self.state = state
    def get_obs(self):
        try:
            return self.state.emission.rvs(self.stop - self.start)
        except AttributeError:
            return ()
    def get_next(self, state, empty=False, length=1, **kwargs):
        if empty: # ensure that we don't emit any obs
            state = None
        return self.__class__(self.seq, state, self.stop, length)


class Model(object):
    def __init__(self, dependencyGraph, obsLabel, logPmin=neginf):
        '''graph represents the dependency structure; it must
        be a dictionary whose keys are dependency group IDs, and
        associated values are lists of state graphs that nodes in
        this dependency group participate in.'''
        self.dependencyGraph = dependencyGraph
        self.clear()
        self.start = StartNode(self.dependencyGraph, obsLabel)
        self.stop = StopNode(self.dependencyGraph)
        self.compiledGraph = {}
        self.compiledGraphRev = {}
        self.logPobsDict = {self.start:self.start.log_p_obs()}
        self.b = {}
        compile_graph(self.compiledGraph, self.compiledGraphRev,
                      self.start, self.b, self.logPobsDict, logPmin, None)

    def clear(self):
        def del_if_found(attr):
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        for attr in ('b', 'bsub', 'f', 'fsub', 'logPobsDict', 'g', 'start',
                     'stop', 'bLeft'):
            del_if_found(attr)

    def calc_fb(self):
        '''Returns forward and backward probability matrices for all
        possible states at all positions in the dependency graph'''
        self.bsub, self.bLeft, self.logPobs = p_backwards(self.b, self.start,
                                      self.compiledGraph, self.logPobsDict)
        self.f,self.fsub = p_forwards(self.compiledGraphRev, self.logPobsDict,
                                      self.b, self.bsub, self.stop,
                                      self.start)
        return self.logPobs

    def posterior(self, node, asLog=False):
        'get posterior probability for the specified node'
        try:
            logP = self.fsub[node] + self.b[node] - self.logPobs
        except KeyError: # some nodes have no path to stop, so inaccessible
            logP = neginf
        if asLog:
            return logP
        else:
            return exp(logP)

    def posterior_ll(self):
        return posterior_ll(self.f)

    def save_graphviz(self, filename, **kwargs):
        outfile = file(filename, 'w')
        try:
            save_graphviz(outfile, self.compiledGraph, self.posterior, **kwargs)
        finally:
            outfile.close()

class BasicHMM(Model):
    '''Convenience subclass for creating a simple linear-traversal HMM
    given its state graph, priors, and termination edges.  Each of these
    must be specified as a StateGraph whose nodes are states and whose
    edge values are transition probabilities.'''
    def __init__(self, sg, prior):
        vg = LabelGraph({0:{0:sg}, 'START':{0:prior}})
        Model.__init__(self, vg)

class TrivialMap(object):
    'maps any and all keys to the specified value'
    def __init__(self, v):
        self.v = v
    def __getitem__(self, k):
        return self.v

class StateGraph(object):
    '''Provides graph interface to nodes in HMM '''
    def __init__(self, graph):
        '''graph supplies the allowed state-state transitions and probabilities'''
        self.graph = graph

    def __call__(self, fromNode, targetLabel, state=None, parent=None):
        '''return dict of {label:{node:pTransition}} from fromNode'''
        results = {}
        if state is None:
            state = fromNode.state
        if targetLabel.obsLabel is not None:
            obsLabel = targetLabel.obsLabel
        else:
            obsLabel = fromNode.var.obsLabel
        for dest,edge in self.graph[state].items():
            newNode = dest(fromNode, targetLabel, obsLabel, edge, parent)
            if newNode:
                results[newNode] = edge
        return results
        
# state classes
#
# __call__() interface allows each state type to control how it
# "moves" in obs space, e.g. a linear chain state just advances the
# obsID +1; a pairwise match state could advance both x,y +1... etc.

class State(object):
    def __init__(self, name, emission):
        self.emission = emission
        self.name = name

    def __call__(self, fromNode, targetLabel, obsLabel, edge, parent):
        '''default move operator just echoes the next obs'''
        try:
            newLabel = targetLabel.get_obs_label(obsLabel.get_next(state=self),
                                                 parent)
        except ObsExhaustedError:
            return None
        return Node(self, newLabel)

    def get_ll(self, node):
        '''generate the log-likelihood of observations for node,
        returned as a list [ll1,ll2,...].
        This baseclass method treats all obs as independent, but
        subclasses can implement more interesting joint probability structures'''
        def get_plist(obs): # workaround to fall back to pdf() method if needed
            try: # scipy.stats models return pmf attr even it doesn't exist...
                return self.emission.pmf(obs)
            except AttributeError:
                return self.emission.pdf(obs)
        obsList = node.var.obsLabel.get_obs()
        if len(obsList) > 0:
            try:
                return numpy.log(get_plist(obsList))
            except NameError:
                return [safe_log(p) for p in get_plist(obsList)]
        else:
            return () # no obs values here...
                
    def __hash__(self):
        return id(self)

    def __repr__(self):
        return self.name

class LinearState(State):
    '''Models a state in a linear chain '''
    pass


class FilterState(State):
    'filters observations according to tag=value kwargs'
    def __init__(self, name, emission, **tags):
        State.__init__(self, name, emission)
        self.tags = tags
        
    def __call__(self, fromNode, targetLabel, obsLabel, edge, parent):
        newLabel = targetLabel.get_obs_label(obsLabel.get_subset(** self.tags),
                                             parent)
        return Node(self, newLabel)


class SilentState(State):
    'state that emits nothing'
    def __init__(self, name):
        self.name = name

    def __call__(self, fromNode, targetLabel, obsLabel, edge, parent):
        obsLabel = obsLabel.get_next(empty=True) # we emit nothing
        newLabel = NodeLabel(targetLabel.graph, targetLabel.label, obsLabel,
                             parent)
        return Node(self, newLabel)

class StopState(State):
    def __init__(self, useObsLabel=True):
        State.__init__(self, 'STOP', None)
        self.useObsLabel = useObsLabel
    def __call__(self, fromNode, targetLabel, obsLabel, edge, parent):
        if self.useObsLabel:
            obsLabel = obsLabel.get_next(empty=True) # we emit nothing
        else:
            obsLabel = None
        return StopNode(targetLabel.graph, obsLabel, parent)
        
class LinearStateStop(StopState):
    def __call__(self, fromNode, targetLabel, obsLabel, edge, parent):
        '''Only return STOP node if at the end of the obs set '''
        if not State.__call__(self, fromNode, targetLabel, obsLabel, edge, parent):
            # exhausted obs, so transition to STOP
            return StopNode(targetLabel.graph, None, parent)

class SeqMatchState(State):
    def __init__(self, name, emission, match_f):
        State.__init__(self, name, emission)
        self.match_f = match_f

    def __call__(self, fromNode, targetLabel, obsLabel, edge, parent):
        if obsLabel.stop < len(obsLabel.seq):
            s = self.match_f(obsLabel.seq[obsLabel.stop:]) # run matcher
            if s:
                obsLabel = obsLabel.get_next(length=len(s))
                return Node(self, targetLabel.get_obs_label(obsLabel, parent))


class EmissionDict(dict):
    'state interface with arbitrary obs --> probability mapping'
    def pmf(self, obs):
        l = []
        for o in obs:
            try:
                l.append(self[o])
            except KeyError:
                l.append(0.)
        return l
    def __hash__(self):
        return id(self)
    def rvs(self, n):
        'generate sample of random draws of size n'
        obs = []
        for i in range(n):
            p = random.random()
            total = 0.
            for o,edge in self.items(): # generate observation
                total += edge
                if p <= total:
                    obs.append(o)
                    break
        return obs

def compile_subgraph(g, gRev, node, b, logPobsDict, logPmin):
    'recurse to subgraph of this node; creates node.stops as list of endnodes'
    node.stops = {} # for list of unique stop nodes in this subgraph
    start = StartNode(node.state.subgraph, obsLabel=node.var.obsLabel,
                      parent=node)
    logPobsDict[start] = 0.
    g[node] = ({start:1.},) # save edge from node to start of its subgraph
    gRev.setdefault(start, {})[node] = 1.
    compile_graph(g, gRev, start, b, logPobsDict, logPmin, node) # subgraph

def compile_graph(g, gRev, node, b, logPobsDict, logPmin, parent):
    '''Generate complete traversal of variable / observation / state graphs
    and return forward graph in the form {src:[{dest:edge}]} and reverse
    graph in the form {dest:{src:edge}}.  In the forward form, each
    source node has multiple target graphs, one for each independent
    target variable.

    Populates logPobsDict with the total log probability of the observations
    at each node.  Also assigns backward probability of 100% to any STOP node
    to enforce path termination at that point.

    logPmin enforces truncation of the path at low (or zero) probability
    nodes.

    parent, if not None, must be node containing this subgraph'''
    if hasattr(node.state, 'subgraph'):
        compile_subgraph(g, gRev, node, b, logPobsDict, logPmin)
        fromNodes = node.stops # generate edges from subgraph stop node(s)
    else:
        fromNodes = (node,) # generate edges from this node
    for fromNode in fromNodes:
        targets = []
        for stateGroup in node.get_children(fromNode, parent):
            d = {}
            for dest,edge in stateGroup.items(): # multiple states sum...
                if not hasattr(dest, 'isub'):
                    dest.isub = len(targets)
                try:
                    logPobs = logPobsDict[dest]
                except KeyError:
                    logPobsDict[dest] = logPobs = dest.log_p_obs()
                    if dest.state == 'STOP': # terminate the path
                        if parent is not None: # add to parent's stop-node list
                            parent.stops[dest] = None
                        else: # backwards probability of terminal node = 1
                            b[dest] = 0.
                        g[dest] = () # prevent recursion on dest
                if logPobs <= logPmin:
                    continue # truncate: do not recurse to dest
                gRev.setdefault(dest, {})[fromNode] = edge # save reverse graph
                d[dest] = edge
                if dest not in g: # subtree not already compiled, so recurse
                    compile_graph(g, gRev, dest, b, logPobsDict, logPmin,
                                  parent)
            if d: # non-empty set of forward edges
                targets.append(d)
        g[fromNode] = targets # save forward graph, even if empty

def p_backwards(b, start, compiledGraph, logPobsDict):
    '''backwards probability algorithm
    Returns backwards probabilities.'''
    bsub = {} # backward prob graph for each child of a given node
    bLeft = {} # backward prob for siblings to our left
    p_backwards_sub(start, b, compiledGraph, logPobsDict, bsub, bLeft)
    for node,l in bLeft.items():
        bLeft[node] = log_sum_list(l)
    return bsub, bLeft, b[start]
        
def p_backwards_sub(node, b, g, logPobsDict, bsub, bLeft):
    '''Computes b[node] = log p(X_p | node), where X_p means all
    obs emitted by descendants of this node Theta_p.

    Also stores bsub[node][r] = log p(X_pr | node), where
    X_pr means all obs emitted by descendants of child r of this node.'''
    for target in g[node]: # multiple dependencies multiply...
        logP = []
        for dest,edge in target.items(): # multiple states sum...
            try:
                logP.append(b[dest] + safe_log(edge) + logPobsDict[dest])
            except KeyError:  # need to compute this value
                p_backwards_sub(dest, b, g, logPobsDict, bsub, bLeft)
                logP.append(b[dest] + safe_log(edge) + logPobsDict[dest])
        if logP: # non-empty list
            lsum = log_sum_list(logP)
            bsub.setdefault(node, {})[dest.isub] = lsum
            try:
                logProd
            except NameError:
                logProd = lsum
            else:
                bLeft.setdefault(dest, []).append(logProd)
                logProd += lsum
    try:
        b[node] = logProd
    except NameError:
        b[node] = neginf

def p_forwards(g, logPobsDict, b, bsub, stop, start):
    '''g: reverse graph generated by p_backwards()
    Reverse traversal begins at STOP by default.
    Returns forward probabilities'''
    f = {}
    fsub = {}
    f[start] = 0.
    fsub[start] = 0.
    p_forwards_sub(g, logPobsDict, stop, b, bsub, f, fsub)
    return f,fsub
    
def p_forwards_sub(g, logPobsDict, dest, b, bsub, f, fsub):
    logP = []
    logPall = []
    for src,edge in g[dest].items():
        try:
            logP.append(f[src] + logPobsDict[src] + safe_log(edge))
        except KeyError:  # need to compute this value
            p_forwards_sub(g, logPobsDict, src, b, bsub, f, fsub)
            logP.append(f[src] + logPobsDict[src] + safe_log(edge))
        logPall.append(fsub[src] + safe_log(edge) + logPobsDict[dest]
                       + b[src] - bsub[src][dest.isub])
    f[dest] = log_sum_list(logP)
    fsub[dest] = log_sum_list(logPall)


def posterior_ll(f):
    '''compute posterior log-likelihood list for each obs group.
    Result is returned as dict of form {obsID:[ll1, ll2, ...]}'''
    d = {}
    for node,f_ti in f.items(): # join different states indexed by obsID
        ll = node.get_ll()
        llObs = [f_ti] # 1st entry is f_ti w/o any obs likelihood
        for logP in ll:
            f_ti += logP
            llObs.append(f_ti)
        d.setdefault(node.var.obsLabel, []).append(llObs)
    llDict = {}
    for obsLabel,ll in d.items():
        nobs = len(ll[0]) # actually this is #obs + 1
        logSum = []
        for iobs in range(nobs): # sum over all states for each obs
            logSum.append(log_sum_list([llstate[iobs] for llstate in ll]))
        # posterior likelihood for each obs
        llDict[obsLabel] = [(logSum[iobs] - logSum[iobs - 1])
                            for iobs in range(1, nobs)]
    return llDict

def save_graphviz(outfile, g, post_f=None, majorColor='red',
                  label_f=lambda x:str(x.state)):
    '''generate dot file from graph g
    post_f, if not None, must be function that returns posterior probability
    of a node
    majorColor is the color assigned to nodes with > 50% probability.'''
    from gvgen import GvGen
    gd = GvGen()
    gNodes = {}
    gClusters = {}
    colors = ('black', 'green', 'blue', 'red')
    for node in g:
        try:
            parent = gClusters[node.var]
        except KeyError:
            parent = gClusters[node.var] = gd.newItem(str(node.var))
        gNodes[node] = gd.newItem(label_f(node), parent)
        if post_f and post_f(node) > 0.5:
            gd.propertyAppend(gNodes[node], 'color', majorColor)
    for node, dependencies in g.items():
        for i,targets in enumerate(dependencies):
            for dest,edge in targets.items():
                e = gd.newLink(gNodes[node], gNodes[dest])
                gd.propertyAppend(e, 'color', colors[i % len(colors)])
    gd.dot(outfile)

