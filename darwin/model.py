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
    top = max(logList)
    return top + log(sum([exp(x - top) for x in logList]))
    
def safe_log(x):
    "Doesn't blow up like math.log(0.) does"
    if x==0.:
        return neginf
    return log(x)


class Node(object):
    '''Node class for compiled state-instance graphs.
    Note that multiple Node instances with the same state, depID, obsID
    will compare and hash as equal.  This enables different paths
    to arrive at the same node even though they construct different
    object instances -- the different instances will compare as
    equal when looking them up in the forward - backward dictionaries.'''
    def __init__(self, state, depID=None, obsID=None, obsDict=None):
        self.state = state
        self.depID = depID
        self.obsID = obsID
        self.obsDict = obsDict
    def __hash__(self):
        return hash((self.state,self.depID,self.obsID))
    def __cmp__(self, other):
        try:
            return cmp((self.state,self.depID,self.obsID),
                       (other.state,other.depID,other.obsID))
        except AttributeError:
            return cmp(id(self), id(other))
    def __repr__(self):
        return '<%s: %s (%s)>' % (repr(self.state), str(self.obsID),
                                  str(self.depID))


START = Node('START', 'START')
STOP = Node('STOP', 'STOP')

class ObservationDict(dict):
    '''return set of observations for a model node '''
    def __getitem__(self, node):
        try:
            obsID = node.obsID
        except AttributeError:
            raise KeyError('not a model node!')
        return dict.__getitem__(self, obsID)

def obs_sequence(seq):
    'transform seq into a ObservationDict'
    d = ObservationDict()
    for i,s in enumerate(seq):
        d[i] = (s,)
    return d

class DependencyGraph(UserDict.DictMixin):
    def __init__(self, graph, obsDict=None):
        '''graph represents the dependency structure; it must
        be a dictionary whose keys are dependency group IDs, and
        associated values are lists of state graphs that nodes in
        this dependency group participate in.'''
        self.graph = graph
        self.obsDict = obsDict

    def __getitem__(self, k):
        '''Return results from state graphs that this node participates in'''
        try:
            depID = k.depID
        except AttributeError:
            raise KeyError('key not in this DependencyGraph')
        stateGraphs = self.graph[depID]
        return [sg[k] for sg in stateGraphs] # list of state graphs

    def __invert__(self):
        'generate inverse dependency graph'
        try:
            return self._inverse
        except AttributeError:
            pass
        inv = {}
        for k,v in self.graph.items():
            for sg in v:
                sgInv = ~sg
                sgInv.depID = k
                inv.setdefault(sg.depID, []).append(sgInv)
        self._inverse = self.__class__(inv, self.obsDict)
        self._inverse._inverse = self
        return self._inverse

    def simulate_seq(self, n):
        'simular markov chain of length n'
        node = START
        s = []
        obs = []
        for i in range(n):
            p = random.random()
            total = 0.
            for sg in self[node]:
                for dest,edge in sg.items(): # choose next state
                    total += edge
                    if p <= total:
                        break;
                break # this algorithm can only handle linear chain ...
            s.append(dest)
            p = random.random()
            total = 0.
            for o,edge in dest.state.emission.items(): # generate observation
                total += edge
                if p <= total:
                    break;
            obs.append(o)
            node = dest
        return s,obs

    def p_backwards(self, obsDict, node=START, b=None, g=None):
        '''backwards probability algorithm
        Begins at START by default.
        Returns backwards probabilities.'''
        if b is None:
            b = {}
            node.obsDict = obsDict
        if g is None:
            g = {}
        prod = 1.
        for sg in self[node]: # multiple dependencies multiply...
            p = 0.
            hasTransitions = False
            for dest,edge in sg.items(): # multiple states sum...
                g.setdefault(dest, {})[node] = edge # save reverse graph
                if dest.depID == 'STOP':
                    p += edge
                    b.setdefault(dest, 1.)
                    hasTransitions = True
                    continue
                hasTransitions = True
                pObs = 1.
                try:
                    obs = obsDict[dest]
                except KeyError: # allow nodes with no obs
                    pass
                else:
                    for po in dest.state.pmf(obs):
                        pObs *= po
                try:
                    p += b[dest] * edge * pObs
                except KeyError:  # need to compute this value
                    self.p_backwards(obsDict, dest, b, g)
                    p += b[dest] * edge * pObs
            if hasTransitions:
                prod *= p
        b[node] = prod
        return b

    def calc_fb(self, obsDict):
        '''Returns forward and backward probability matrices for all
        possible states at all positions in the dependency graph'''
        g = {}
        b = self.p_backwards(obsDict, g=g)
        f = p_forwards(g, obsDict)
        return f, b


class StateGraph(UserDict.DictMixin):
    '''Provides graph interface to nodes in HMM '''
    def __init__(self, graph, depID=0):
        '''graph supplies the allowed state-state transitions '''
        self.graph = graph
        self.depID = depID

    def __getitem__(self, fromNode):
        '''return dict of destination nodes & edges for fromNode '''
        targets = self.graph[fromNode.state]
        d = {}
        for dest,edge in targets.items():
            try: # construct destination node using dest type
                toNode = dest(fromNode, self.depID)
            except StopIteration: # no valid destination from here
                pass
            else:
                d[toNode] = edge
        return d

    def __invert__(self):
        try:
            return self._inverse
        except AttributeError:
            pass
        inv = {}
        for src,d in self.graph.items():
            for dest,edge in d.items():
                inv.setdefault(dest, {})[src] = edge
        self._inverse = self.__class__(inv, self.depID)
        self._inverse._inverse = self
        return self._inverse
        
# state classes
#
# __call__() interface allows each state type to control how it
# "moves" in obs space, e.g. a linear chain state just advances the
# obsID +1; a pairwise match state could advance both x,y +1... etc.
# Should raise StopIteration if no valid next node, e.g. if obs
# are exhausted.  Since we are using a constructor interface,
# we use StopIteration (rather than a return value like None) to
# signal "no valid next node".

class State(object):
    def __init__(self, name, emission):
        self.emission = emission
        self.name = name

    def pmf(self, obs):
        '''Generate likelihoods for obs set '''
        return self.emission.pmf(obs)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return self.name

class LinearState(State):
    '''Models a state in a linear chain '''
    def __call__(self, fromNode, depID):
        '''Construct next node in HMM after fromNode'''
        obsID = fromNode.obsID
        if obsID is None:
            obsID = 0
        else:
            obsID += 1
        if obsID < 0 or (fromNode.obsDict is not None
                         and obsID >= len(fromNode.obsDict)):
            raise StopIteration # no more observations, so HMM ends here
        return Node(self, depID, obsID, fromNode.obsDict)

class LinearStateStop(object):
    def __call__(self, fromNode, depID):
        '''Only return STOP node if at the end of the obs set '''
        if fromNode.obsDict is not None \
               and fromNode.obsID + 1 >= len(fromNode.obsDict):
            return STOP # exhausted obs, so transition to STOP
        raise StopIteration # no path to STOP
        
class EmissionDict(dict):
    'state interface with arbitrary obs --> probability mapping'
    def pmf(self, obs):
        return [self[o] for o in obs]
    def __hash__(self):
        return id(self)

def p_forwards(g, obsDict, dest=STOP, f=None):
    '''g: reverse graph generated by p_backwards()
    Reverse traversal begins at STOP by default.
    Returns forward probabilities'''
    if f is None:
        f = {}
    if dest.depID == 'START':
        f[dest] = 1.
        return f
    pObs = 1.
    try:
        obs = obsDict[dest]
    except KeyError: # allow nodes with no obs
        pass
    else:
        for po in dest.state.pmf(obs):
            pObs *= po
    p = 0.
    for src,edge in g[dest].items():
        try:
            p += f[src] * edge
        except KeyError:  # need to compute this value
            p_forwards(g, obsDict, src, f)
            p += f[src] * edge
    f[dest] = p * pObs
    return f


# test code

def ocd_test(p6=.5, n=100):
    'Occasionally Dishonest Casino example'
    p = (1. - p6) / 5.
    L = LinearState('L', EmissionDict({1:p, 2:p, 3:p, 4:p, 5:p, 6:p6}))
    p = 1. / 6.
    F = LinearState('F', EmissionDict({1:p, 2:p, 3:p, 4:p, 5:p, 6:p}))
    stop = LinearStateStop()
    sg = StateGraph({F:{F:0.95, L:0.05}, L:{F:0.1, L:0.9}})
    prior = StateGraph({'START':{F:2./3., L:1./3.}})
    term = StateGraph({F:{stop:1.}, L:{stop:1.}}, 'STOP')
    dg = DependencyGraph({0:[sg, term], 'START':[prior]})

    s,obs = dg.simulate_seq(n)
    obsDict = obs_sequence(obs)
    f, b = dg.calc_fb(obsDict)
    pObs = b[START]
    for i in range(n): # print posteriors
        nodeF = Node(F, 0, i, obsDict)
        nodeL = Node(L, 0, i, obsDict)
        print '%s:%0.3f\t%s:%0.3f\tTRUE:%s,%d' % \
              (nodeF, f[nodeF] * b[nodeF] / pObs,
               nodeL, f[nodeL] * b[nodeL] / pObs,
               s[i], obs[i])
    return dg
