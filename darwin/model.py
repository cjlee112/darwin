import UserDict
from math import *
import random

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
    multiple Node instances with the same state, ruleID, obsTuple will
    compare and hash as equal.  This enables different paths to arrive
    at the same node even though they construct different object
    instances -- the different instances will compare as equal when
    looking them up in the forward - backward dictionaries.'''
    def __init__(self, state, var):
        self.state = state
        self.var = var

    def get_children(self, fromNode=None, parent=None):
        '''Get descendants in form {label:{node:pTransition}} '''
        results = None
        if fromNode is None:
            fromNode = self
        for varLabel, stateGraph in self.var.get_children().items():
            d = stateGraph(fromNode, varLabel, state=self.state, parent=parent)
            try:
                results.update(d)
            except AttributeError:
                results = d
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

    def get_ll_dict(self):
        try:
            f = self.state.get_ll_dict
        except AttributeError: # START and STOP lack this method
            return {}
        return f(self)

    def log_p_obs(self):
        'compute total log-likelihood for all obs emitted by this node'
        logPobs = 0.
        for l in self.get_ll_dict().itervalues():
            logPobs += sum(l)
        return logPobs


class StartNode(Node):
    def __init__(self, graph, obsTuple=None, obsGraphs=None, parent=None):
        if obsTuple is None:
            obsTuple = tuple([g.get_start() for g in obsGraphs])
        label = graph.get_start(obsTuple=obsTuple, parent=parent)
        Node.__init__(self, 'START', label)

class StopNode(Node):
    def __init__(self, graph, obsTuple=None, parent=None):
        Node.__init__(self, 'STOP', NodeLabel(graph, 'STOP', obsTuple, parent))

class Label(object):
    '''Reference to a specific vertex in a graph'''
    def __init__(self, graph, label, values=None):
        self.graph = graph
        self.label = label
        if values is not None:
            self.values = values

    def __repr__(self):
        return str(self.label)
    def __hash__(self):
        return hash((self.graph, self.label))
    def __cmp__(self, other):
        try:
            return cmp((self.graph, self.label), (other.graph, other.label))
        except AttributeError:
            return cmp(id(self), id(other))
    def get_children(self, **kwargs):
        '''Get child,obs pairs for child nodes of this node '''
        return self.graph.__getitem__(self, **kwargs)
    def __len__(self):
        'Get count of descendants'
        return len(self.graph[self])

class ObsLabel(Label):
    pass

class NodeLabel(Label):
    def __init__(self, graph, label, obsTuple=None, parent=None):
        self.graph = graph
        self.label = label
        self.obsTuple = obsTuple
        self.parent = parent
    def __hash__(self):
        return hash((self.graph, self.label, self.obsTuple, id(self.parent)))
    def __cmp__(self, other):
        try:
            return cmp((self.graph, self.label, self.obsTuple, id(self.parent)),
                  (other.graph, other.label, other.obsTuple, id(other.parent)))
        except AttributeError:
            return cmp(id(self), id(other))
    def __repr__(self):
        return str((self.label, self.obsTuple))
    def get_obs_label(self, obsID, parent=None):
        if parent is None:
            parent = self.parent
        return self.__class__(self.graph, self.label, (obsID,), parent)

class LabelGraph(object):
    labelClass = Label

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
        results = {}
        for label,edge in d.items():
            if not isinstance(label, self.labelClass):
                label = self.get_target(label, edge)
            results[label] = edge
        return results

    def get_target(self, label, edge):
        return self.labelClass(self, label) # target doesn't use edge value

    def get_start(self, **kwargs):
        'get START node for this graph'
        return self.get_label('START', **kwargs)

    def get_label(self, label, *args, **kwargs):
        'construct label object with specified args'
        return self.labelClass(self, label, *args, **kwargs)

    def __hash__(self):
        return id(self)


class ObsGraph(LabelGraph):
    labelClass = ObsLabel
    def __getitem__(self, node, **kwargs): # extra arguments passed for sim'n
        return LabelGraph.__getitem__(self, node)
    
    def get_target(self, label, edge):
        return self.labelClass(self, label, edge) # store obs in ObsLabel



class ObsSequence(ObsGraph):
    '''simple linear obs graph producing integer obs label values'''
    def __init__(self, seq):
        self.seq = seq

    def __getitem__(self, node, **kwargs):
        try:
            if node.graph is not self:
                raise AttributeError
            i = node.label + 1
        except AttributeError:
            raise KeyError('node not in this graph')
        if i < len(self.seq):
            return {self.get_label(i, (self.seq[i],)):self.seq[i]}
        else: # reached end of sequence
            return {}

    def get_start(self):
        return self.get_label(-1)

class ObsSeqSimulator(ObsSequence):
    def __getitem__(self, node, fromNode=None, targetLabel=None, state=None):
        try:
            obs = state.emission.rvs(1)
            return {ObsLabel(self, node.label + 1, obs):obs}
        except AttributeError:
            return {ObsLabel(self, node.label + 1, None):None}

class NodeGraph(LabelGraph):
    '''Graph of the form {source:{target:edge}}
    where source, target are NodeLabel objects, and
    edge is the state graph specifying the state transitions from source
    to target.'''
    labelClass = NodeLabel

    def simulate_seq(self, n):
        'simulate markov chain of length n'
        node = StartNode(self, obsGraphs=(ObsSeqSimulator(None),))
        s = []
        obs = []
        for i in range(n):
            p = random.random()
            total = 0.
            for label,sg in node.get_children().items():
                for dest,edge in sg.items(): # choose next state
                    if dest.state == 'STOP':
                        continue
                    total += edge
                    if p <= total:
                        break
                break # this algorithm can only handle linear chain ...
            s.append(dest)
            obs.append(dest.var.obsTuple[0].values[0])
            node = dest
        return s,obs

class TrivialGraph(LabelGraph):
    'Graph containing single node with self-edge'
    def __init__(self, var, stateGraph):
        LabelGraph.__init__(self, {var.label:{var.label:stateGraph}})

class Model(object):
    def __init__(self, dependencyGraph, obsGraphs, logPmin=neginf):
        '''graph represents the dependency structure; it must
        be a dictionary whose keys are dependency group IDs, and
        associated values are lists of state graphs that nodes in
        this dependency group participate in.'''
        self.dependencyGraph = dependencyGraph
        self.clear()
        self.start = StartNode(self.dependencyGraph, obsGraphs=obsGraphs)
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
                                      self.b, self.bsub, self.stop)
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

class StateGraph(UserDict.DictMixin):
    '''Provides graph interface to nodes in HMM '''
    def __init__(self, graph):
        '''graph supplies the allowed state-state transitions and probabilities'''
        self.graph = graph

    def __call__(self, fromNode, targetLabel, state=None, parent=None):
        '''return dict of {label:{node:pTransition}} from fromNode'''
        results = {}
        if state is None:
            state = fromNode.state
        for dest,edge in self.graph[state].items():
            for k,v in dest(fromNode, targetLabel, edge, parent).items():
                try:
                    results[k].update(v)
                except KeyError:
                    results[k] = v
        return results

    def __invert__(self):
        try:
            return self._inverse
        except AttributeError:
            pass
        inv = {}
        for src,d in self.graph.items():
            for dest,edge in d.items():
                inv.setdefault(dest, {})[src] = edge
        self._inverse = self.__class__(inv)
        self._inverse._inverse = self
        return self._inverse
        
# state classes
#
# __call__() interface allows each state type to control how it
# "moves" in obs space, e.g. a linear chain state just advances the
# obsID +1; a pairwise match state could advance both x,y +1... etc.

class State(object):
    def __init__(self, name, emission):
        self.emission = emission
        self.name = name

    def __call__(self, fromNode, targetLabel, edge, parent):
        '''default move operator just echoes the obs graph branches'''
        results = {}
        obsPaths = fromNode.var.obsTuple[0]. \
                   get_children(fromNode=fromNode, targetLabel=targetLabel,
                                state=self)
        for obsLabel in obsPaths:
            newLabel = targetLabel.get_obs_label(obsLabel, parent)
            targetNode = Node(self, newLabel)
            results.setdefault(newLabel, {})[targetNode] = edge
        return results

    def get_ll_dict(self, node):
        '''generate the log-likelihood of observations for node,
        returned as a dict of the form {obsID:[ll1,ll2,...]}.
        This baseclass method treats all obs as independent, but
        subclasses can implement more interesting likelihood models'''
        d = {}
        def get_plist(obs): # workaround to fall back to pdf() method if needed
            try: # scipy.stats models return pmf attr even it doesn't exist...
                return self.emission.pmf(obs)
            except AttributeError:
                return self.emission.pdf(obs)
        for obsLabel in node.var.obsTuple:
            try:
                obsList = obsLabel.values
            except AttributeError: # no obs values here...
                pass
            else:
                d[obsLabel] = [safe_log(p) for p in get_plist(obsList)]
        return d
                
    def __hash__(self):
        return id(self)

    def __repr__(self):
        return self.name

class LinearState(State):
    '''Models a state in a linear chain '''
    pass

class SilentState(State):
    def __init__(self, name):
        self.name = name

    def __call__(self, fromNode, targetLabel, edge, parent):
        targetLabel = NodeLabel(targetLabel.graph, targetLabel.label,
                                fromNode.var.obsTuple, parent)
        return {targetLabel:{Node(self, targetLabel):edge}}

    def get_ll_dict(self, node):
        return {}

class StopState(State):
    def __init__(self):
        State.__init__(self, 'STOP', None)
    def __call__(self, fromNode, targetLabel, edge, parent):
        return {'STOP':{StopNode(targetLabel.graph, fromNode.var.obsTuple,
                                 parent):edge}}
        
class LinearStateStop(StopState):
    def __call__(self, fromNode, targetLabel, edge, parent):
        '''Only return STOP node if at the end of the obs set '''
        if not State.__call__(self, fromNode, targetLabel, edge, parent):
            # exhausted obs, so transition to STOP
            return {'STOP':{StopNode(targetLabel.graph, None, parent):edge}}
        else:
            return {}

class SeqMatchState(State):
    def __init__(self, name, emission, match_f):
        State.__init__(self, name, emission)
        self.match_f = match_f

    def __call__(self, fromNode, targetLabel, edge, parent):
        obsLabel = fromNode.var.obsTuple[0]
        ipos = obsLabel.label + 1
        if ipos < len(obsLabel.graph.seq):
            s = self.match_f(obsLabel.graph.seq[ipos:])# run matcher
            if s:
                obsLabel = ObsLabel(obsLabel.graph, ipos + len(s) - 1, s)
                newLabel = targetLabel.get_obs_label(obsLabel, parent)
                targetNode = Node(self, newLabel)
                return {newLabel:{targetNode:edge}}
            else:
                return {}
        else:
            return {}

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
    start = StartNode(node.state.subgraph, obsTuple=node.var.obsTuple,
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
        for label,sg in node.get_children(fromNode, parent).items():
            d = {}
            for dest,edge in sg.items(): # multiple states sum...
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
    logProd = 0.
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
            bsub.setdefault(node, {})[dest.var] = lsum
            if logProd < 0.:
                bLeft.setdefault(dest, []).append(logProd)
            logProd += lsum
    b[node] = logProd

def p_forwards(g, logPobsDict, b, bsub, stop):
    '''g: reverse graph generated by p_backwards()
    Reverse traversal begins at STOP by default.
    Returns forward probabilities'''
    f = {}
    fsub = {}
    p_forwards_sub(g, logPobsDict, stop, b, bsub, f, fsub)
    return f,fsub
    
def p_forwards_sub(g, logPobsDict, dest, b, bsub, f, fsub):
    if dest.state == 'START':
        f[dest] = 0.
        fsub[dest] = 0.
        return f
    logP = []
    logPall = []
    for src,edge in g[dest].items():
        try:
            logP.append(f[src] + logPobsDict[src] + safe_log(edge))
        except KeyError:  # need to compute this value
            p_forwards_sub(g, logPobsDict, src, b, bsub, f, fsub)
            logP.append(f[src] + logPobsDict[src] + safe_log(edge))
        logPall.append(fsub[src] + safe_log(edge) + logPobsDict[dest]
                       + b[src] - bsub[src][dest.var])
    f[dest] = log_sum_list(logP)
    fsub[dest] = log_sum_list(logPall)


def posterior_ll(f):
    '''compute posterior log-likelihood list for each obs group.
    Result is returned as dict of form {obsID:[ll1, ll2, ...]}'''
    d = {}
    for node,f_ti in f.items(): # join different states indexed by obsID
        for obsID,ll in node.get_ll_dict().items():
            llObs = [f_ti] # 1st entry is f_ti w/o any obs likelihood
            for logP in ll:
                f_ti += logP
                llObs.append(f_ti)
            d.setdefault(node.var.get_obs_label(obsID), []).append(llObs)
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
    for node in g:
        try:
            parent = gClusters[node.var]
        except KeyError:
            parent = gClusters[node.var] = gd.newItem(str(node.var))
        gNodes[node] = gd.newItem(label_f(node), parent)
        if post_f and post_f(node) > 0.5:
            gd.propertyAppend(gNodes[node], 'color', majorColor)
    for node, dependencies in g.items():
        for targets in dependencies:
            for dest,edge in targets.items():
                e = gd.newLink(gNodes[node], gNodes[dest])
    gd.dot(outfile)

