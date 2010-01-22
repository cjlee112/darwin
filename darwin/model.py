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

    def get_children(self):
        '''Get descendants in form {label:{node:pTransition}} '''
        results = None
        for varLabel, stateGraph in self.var.get_children().items():
            d = stateGraph(self, varLabel)
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
    def __init__(self, graph, obsGraphs):
        obsTuple = tuple([g.get_start() for g in obsGraphs])
        label = graph.get_start(obsTuple=obsTuple)
        Node.__init__(self, 'START', label)

class StopNode(Node):
    def __init__(self, graph):
        Node.__init__(self, 'STOP', NodeLabel(graph, 'STOP', None))

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
    def __init__(self, graph, label, obsTuple):
        self.graph = graph
        self.label = label
        self.obsTuple = obsTuple
    def __hash__(self):
        return hash((self.graph, self.label, self.obsTuple))
    def __cmp__(self, other):
        try:
            return cmp((self.graph, self.label, self.obsTuple),
                       (other.graph, other.label, other.obsTuple))
        except AttributeError:
            return cmp(id(self), id(other))
    def __repr__(self):
        return str((self.label, self.obsTuple))
    def get_obs_label(self, obsID):
        return self.__class__(self.graph, self.label, (obsID,))


class ObsSequence(object):
    '''simple linear obs graph '''
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
            return {ObsLabel(self, i, (self.seq[i],)):self.seq[i]}
        else: # reached end of sequence
            return {}

    def get_start(self):
        return ObsLabel(self, -1)

    def __hash__(self):
        return id(self)

class ObsSeqSimulator(ObsSequence):
    def __getitem__(self, node, fromNode=None, targetLabel=None, state=None):
        try:
            obs = state.emission.rvs(1)
            return {ObsLabel(self, node.label + 1, obs):obs}
        except AttributeError:
            return {ObsLabel(self, node.label + 1, None):None}

class LabelGraph(object):
    '''Graph '''
    def __init__(self, graph):
        self.graph = graph

    def __getitem__(self, node):
        try:
            if node.graph is not self:
                raise KeyError
            d = self.graph[node.label]
        except (KeyError, AttributeError):
            raise KeyError('node not in this graph')
        results = {}
        for label,sg in d.items():
            var = NodeLabel(self, label, None)
            results[var] = sg
        return results

    def get_start(self, klass=NodeLabel, **kwargs):
        return klass(self, 'START', **kwargs)

    def __hash__(self):
        return id(self)

class TrivialGraph(LabelGraph):
    'Graph containing single node with self-edge'
    def __init__(self, var, stateGraph):
        LabelGraph.__init__(self, {var.label:{var.label:stateGraph}})

class DependencyGraph(UserDict.DictMixin):
    def __init__(self, graph):
        '''graph represents the dependency structure; it must
        be a dictionary whose keys are dependency group IDs, and
        associated values are lists of state graphs that nodes in
        this dependency group participate in.'''
        self.graph = graph

    def simulate_seq(self, n):
        'simulate markov chain of length n'
        node = StartNode(self.graph, (ObsSeqSimulator(None),))
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

    def p_backwards(self, obsGraphs, logPmin=neginf):
        '''backwards probability algorithm
        Begins at START by default.
        Returns backwards probabilities.'''
        b = {}
        bsub = {} # backward prob graph for each child of a given node
        bLeft = {} # backward prob for siblings to our left
        self.start = StartNode(self.graph, obsGraphs)
        self.stop = StopNode(self.graph)
        node = self.start
        g = {}
        logPobsDict = {node:node.log_p_obs()}
        self.p_backwards_sub(node, b, g, logPobsDict, bsub, bLeft, logPmin)
        for node,l in bLeft.items():
            bLeft[node] = log_sum_list(l)
        return b,g,logPobsDict,bsub
        
    def p_backwards_sub(self, node, b, g, logPobsDict, bsub, bLeft, logPmin):
        '''Computes b[node] = log p(X_p | node), where X_p means all
        obs emitted by descendants of this node Theta_p.

        Also stores bsub[node][r] = log p(X_pr | node), where
        X_pr means all obs emitted by descendants of child r of this node.'''
        logProd = 0.
        for label,sg in node.get_children().items(): # multiple dependencies multiply...
            logP = []
            for dest,edge in sg.items(): # multiple states sum...
                try:
                    logPobs = logPobsDict[dest]
                except KeyError:
                    if dest.state == 'STOP': # terminate the path
                        b[dest] = logPobsDict[dest] = logPobs = 0.
                    else:
                        logPobsDict[dest] = logPobs = dest.log_p_obs()
                if logPobs <= logPmin:
                    continue # truncate: do not recurse to dest
                g.setdefault(dest, {})[node] = edge # save reverse graph
                try:
                    logP.append(b[dest] + safe_log(edge) + logPobs)
                except KeyError:  # need to compute this value
                    self.p_backwards_sub(dest, b, g, logPobsDict, bsub,
                                         bLeft, logPmin)
                    logP.append(b[dest] + safe_log(edge) + logPobs)
            if logP: # non-empty list
                lsum = log_sum_list(logP)
                bsub.setdefault(node, {})[dest.var] = lsum
                if logProd < 0.:
                    bLeft.setdefault(dest, []).append(logProd)
                logProd += lsum
        b[node] = logProd

    def calc_fb(self, obsGraphs, **kwargs):
        '''Returns forward and backward probability matrices for all
        possible states at all positions in the dependency graph'''
        b,g,logPobsDict,bsub = self.p_backwards(obsGraphs, **kwargs)
        f,fsub = p_forwards(g, logPobsDict, b, bsub, self.stop)
        return f, b, fsub, bsub, logPobsDict

class BasicHMM(DependencyGraph):
    '''Convenience subclass for creating a simple linear-traversal HMM
    given its state graph, priors, and termination edges.  Each of these
    must be specified as a StateGraph whose nodes are states and whose
    edge values are transition probabilities.'''
    def __init__(self, sg, prior):
        vg = LabelGraph({0:{0:sg}, 'START':{0:prior}})
        DependencyGraph.__init__(self, vg)

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

    def __call__(self, fromNode, targetLabel):
        '''return dict of {label:{node:pTransition}} from fromNode'''
        results = {}
        for dest,edge in self.graph[fromNode.state].items():
            for k,v in dest(fromNode, targetLabel, edge).items():
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

    def __call__(self, fromNode, targetLabel, edge):
        '''default move operator just echoes the obs graph branches'''
        results = {}
        obsPaths = fromNode.var.obsTuple[0]. \
                   get_children(fromNode=fromNode, targetLabel=targetLabel,
                                state=self)
        for obsLabel in obsPaths:
            newLabel = targetLabel.get_obs_label(obsLabel)
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

class StopState(State):
    def __init__(self):
        State.__init__(self, 'STOP', None)
    def __call__(self, fromNode, targetLabel, edge):
        return {'STOP':{StopNode(targetLabel.graph):edge}}
        
class LinearStateStop(StopState):
    def __call__(self, fromNode, targetLabel, edge):
        '''Only return STOP node if at the end of the obs set '''
        if not State.__call__(self, fromNode, targetLabel, edge):
            # exhausted obs, so transition to STOP
            return StopState.__call__(self, fromNode, targetLabel, edge)
        else:
            return {}

class EmissionDict(dict):
    'state interface with arbitrary obs --> probability mapping'
    def pmf(self, obs):
        return [self[o] for o in obs]
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

