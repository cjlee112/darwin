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

class Variable(object):
    '''Represents label of a variable instance in
    a dependency-observation graph'''
    def __init__(self, ruleID, varID, obsTuple):
        self.ruleID = ruleID
        self.obsTuple = obsTuple
        self.varID = varID

    def __hash__(self):
        return hash((self.ruleID,self.varID,self.obsTuple))

    def __cmp__(self, other):
        try:
            return cmp((self.ruleID,self.varID,self.obsTuple),
                       (other.ruleID,other.varID,other.obsTuple))
        except AttributeError:
            return cmp(id(self), id(other))
        

class Node(object):
    '''Node class for compiled state-instance graphs.  Note that
    multiple Node instances with the same state, ruleID, obsTuple will
    compare and hash as equal.  This enables different paths to arrive
    at the same node even though they construct different object
    instances -- the different instances will compare as equal when
    looking them up in the forward - backward dictionaries.'''
    def __init__(self, state, ruleID=None, obsTuple=(), obsDict=None,
                 varID=0):
        self.state = state
        self.obsDict = obsDict
        self.var = Variable(ruleID, varID, obsTuple)

    def set_label(self, ruleID, varID):
        self.var.ruleID = ruleID
        self.var.varID = varID

    def get_obs_label(self, obsID):
        return (self.var.ruleID, self.var.varID, obsID)
        
    def __hash__(self):
        return hash((self.state,self.var))

    def __cmp__(self, other):
        try:
            return cmp((self.state,self.var),
                       (other.state,other.var))
        except AttributeError:
            return cmp(id(self), id(other))

    def __repr__(self):
        return '<%s: %s %s>' % (repr(self.state), str(self.var.obsTuple),
                                  str((self.var.ruleID,self.var.varID)))

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


START = Node('START', 'START')
STOP = Node('STOP', 'STOP')

class ObservationDict(dict):
    pass

def obs_sequence(ruleID, seq, varID=0):
    'transform seq into a ObservationDict'
    d = ObservationDict()
    for i,s in enumerate(seq):
        d[ruleID,varID,i] = (s,)
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
        d = {} # groups nodes with same ruleID,varID together
        for ruleID,varGraph in self.graph[k.var.ruleID].items():
            for varID,stateGraph in varGraph[k.var.varID].items():
                for node,edge in stateGraph[k].items():
                    node.set_label(ruleID, varID)
                    d.setdefault((ruleID,varID), {})[node] = edge
        return d.values() # list of state graphs

    def __invert__(self): # REWRITE THIS for new ruleID:varID:sg structure!!
        'generate inverse dependency graph'
        try:
            return self._inverse
        except AttributeError:
            pass
        inv = {}
        for k,v in self.graph.items():
            for sg in v:
                sgInv = ~sg
                sgInv.ruleID = k
                inv.setdefault(sg.ruleID, []).append(sgInv)
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

    def p_backwards(self, obsDict, node=START):
        '''backwards probability algorithm
        Begins at START by default.
        Returns backwards probabilities.'''
        b = {}
        bsub = {} # backward prob graph for each child of a given node
        node.obsDict = obsDict
        g = {}
        logPobsDict = {node:node.log_p_obs()}
        self.p_backwards_sub(node, b, g, logPobsDict, bsub)
        return b,g,logPobsDict,bsub
        
    def p_backwards_sub(self, node, b, g, logPobsDict, bsub):
        '''Computes b[node] = log p(X_p | node), where X_p means all
        obs emitted by descendants of this node Theta_p.

        Also stores bsub[node][r] = log p(X_pr | node), where
        X_pr means all obs emitted by descendants of child r of this node.'''
        logProd = 0.
        for sg in self[node]: # multiple dependencies multiply...
            logP = []
            for dest,edge in sg.items(): # multiple states sum...
                g.setdefault(dest, {})[node] = edge # save reverse graph
                try:
                    logPobs = logPobsDict[dest]
                except KeyError:
                    logPobsDict[dest] = logPobs = dest.log_p_obs()
                if dest.var.ruleID == 'STOP':
                    logP.append(safe_log(edge))
                    b.setdefault(dest, 0.)
                    continue
                try:
                    logP.append(b[dest] + safe_log(edge) + logPobs)
                except KeyError:  # need to compute this value
                    self.p_backwards_sub(dest, b, g, logPobsDict, bsub)
                    logP.append(b[dest] + safe_log(edge) + logPobs)
            if logP: # non-empty list
                lsum = log_sum_list(logP)
                bsub.setdefault(node, {})[dest.var] = lsum
                logProd += lsum
        b[node] = logProd

    def calc_fb(self, obsDict):
        '''Returns forward and backward probability matrices for all
        possible states at all positions in the dependency graph'''
        b,g,logPobsDict,bsub = self.p_backwards(obsDict)
        f,fsub = p_forwards(g, logPobsDict, b, bsub)
        return f, b, fsub, bsub, logPobsDict

class BasicHMM(DependencyGraph):
    '''Convenience subclass for creating a simple linear-traversal HMM
    given its state graph, priors, and termination edges.  Each of these
    must be specified as a StateGraph whose nodes are states and whose
    edge values are transition probabilities.'''
    def __init__(self, sg, prior, term):
        DependencyGraph.__init__(self, {0:{0:{0:{0:sg}},
                                           'STOP':{0:{0:term}}},
                                        'START':{0:{0:{0:prior}}}})

class StateGraph(UserDict.DictMixin):
    '''Provides graph interface to nodes in HMM '''
    def __init__(self, graph):
        '''graph supplies the allowed state-state transitions and probabilities'''
        self.graph = graph

    def __getitem__(self, fromNode):
        '''return dict of destination nodes & edges for fromNode '''
        targets = self.graph[fromNode.state]
        d = {}
        for dest,edge in targets.items():
            try: # construct destination node using dest type
                toNode = dest(fromNode)
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
        self._inverse = self.__class__(inv)
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

    def get_ll_dict(self, node):
        '''generate the log-likelihood of observations for node,
        returned as a dict of the form {obsID:[ll1,ll2,...]}.
        This baseclass method treats all obs as independent, but
        subclasses can implement more interesting likelihood models'''
        d = {}
        try:
            f = self.emission.pmf
        except AttributeError:
            f = self.emission.pdf
        for obsID in node.var.obsTuple:
            d[obsID] = [safe_log(p)
                        for p in f(node.obsDict[node.get_obs_label(obsID)])]
        return d
                
    def __hash__(self):
        return id(self)

    def __repr__(self):
        return self.name

class LinearState(State):
    '''Models a state in a linear chain '''
    def __call__(self, fromNode):
        '''Construct next node in HMM after fromNode'''
        try:
            obsID = fromNode.var.obsTuple[0] + 1
        except IndexError: # treat empty list as START: go to 1st obsID
            obsID = 0
        if obsID < 0 or (fromNode.obsDict is not None
                         and obsID >= len(fromNode.obsDict)):
            raise StopIteration # no more observations, so HMM ends here
        return Node(self, None, (obsID,), fromNode.obsDict)

class LinearStateStop(object):
    def __call__(self, fromNode):
        '''Only return STOP node if at the end of the obs set '''
        if fromNode.obsDict is not None \
               and fromNode.var.obsTuple[0] + 1 >= len(fromNode.obsDict):
            return STOP # exhausted obs, so transition to STOP
        raise StopIteration # no path to STOP
        
class EmissionDict(dict):
    'state interface with arbitrary obs --> probability mapping'
    def pmf(self, obs):
        return [self[o] for o in obs]
    def __hash__(self):
        return id(self)

def p_forwards(g, logPobsDict, b, bsub):
    '''g: reverse graph generated by p_backwards()
    Reverse traversal begins at STOP by default.
    Returns forward probabilities'''
    f = {}
    fsub = {}
    p_forwards_sub(g, logPobsDict, STOP, b, bsub, f, fsub)
    return f,fsub
    
def p_forwards_sub(g, logPobsDict, dest, b, bsub, f, fsub):
    if dest.var.ruleID == 'START':
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
            d.setdefault(node.get_obs_label(obsID), []).append(llObs)
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
    term = StateGraph({F:{stop:1.}, L:{stop:1.}})
    dg = BasicHMM(sg, prior, term)

    s,obs = dg.simulate_seq(n)
    obsDict = obs_sequence(0, obs)
    f, b, fsub, bsub, ll = dg.calc_fb(obsDict)
    logPobs = b[START]
    llDict = posterior_ll(f)
    for i in range(n): # print posteriors
        nodeF = Node(F, 0, (i,), obsDict)
        nodeL = Node(L, 0, (i,), obsDict)
        print '%s:%0.3f\t%s:%0.3f\tTRUE:%s,%d,%0.3f' % \
              (nodeF, exp(fsub[nodeF] + b[nodeF] - logPobs),
               nodeL, exp(fsub[nodeL] + b[nodeL] - logPobs),
               s[i], obs[i], exp(llDict[nodeF.get_obs_label(i)][0]))
    return dg
