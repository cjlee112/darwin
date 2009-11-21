import UserDict

class Node(object):
    def __init__(self, state, depID=None, obsID=None):
        self.state = state
        self.depID = depID
        self.obsID = obsID
    def __hash__(self):
        return hash((self.state,self.depID,self.obsID))
    def __cmp__(self, other):
        try:
            return cmp((self.state,self.depID,self.obsID),
                       (other.state,other.depID,other.obsID))
        except AttributeError:
            return cmp(id(self), id(other))


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
        self.graph = graph
        self.obsDict = obsDict

    def __getitem__(self, k):
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

class StateGraph(UserDict.DictMixin):
    def __init__(self, graph, depID=0):
        self.graph = graph
        self.depID = depID

    def __getitem__(self, fromNode):
        targets = self.graph[fromNode.state]
        d = {}
        for dest,edge in targets.items():
            d[dest(fromNode, self.depID)] = edge
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
        

class LinearState(object):
    def __init__(self, emission):
        self.emission = emission

    def __call__(self, fromNode, depID):
        obsID = fromNode.obsID
        if obsID is None:
            obsID = -1
        return Node(self, depID, obsID + 1)

    def pmf(self, obs):
        return self.emission.pmf(obs)

    def __hash__(self):
        return id(self)

class EmissionDict(dict):
    'state interface with arbitrary obs --> probability mapping'
    def pmf(self, obs):
        return [self[o] for o in obs]
    def __hash__(self):
        return id(self)


def simulate_seq(dg, n):
    'simular markov chain of length n'
    import random
    node = START
    s = []
    obs = []
    for i in range(n):
        p = random.random()
        total = 0.
        for sg in dg[node]:
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

def p_backwards(dg, obsDict, node=START, b=None):
    'backwards probability algorithm'
    if b is None:
        b = {}
    prod = 1.
    for sg in dg[node]: # multiple dependencies multiply...
        p = 0.
        for dest,edge in sg.items(): # multiple states sum...
            pObs = 1.
            try:
                obs = obsDict[dest]
            except KeyError: # exhausted obs, terminate here
                b[node] = 1.
                return b
            for po in dest.state.pmf(obs):
                pObs *= po
            try:
                p += b[dest] * edge * pObs
            except KeyError:  # need to compute this value
                p_backwards(dg, obsDict, dest, b)
                p += b[dest] * edge * pObs
        prod *= p
    b[node] = prod
    return b

        
def ocd_test(p6=.5):
    p = (1. - p6) / 5.
    loaded = EmissionDict()
    loaded[6] = p6
    for i in range(1, 6):
        loaded[i] = p
    fair = EmissionDict()
    for i in range(1, 7):
        fair[i] = 1./6.
    F = LinearState(fair)
    L = LinearState(loaded)
    sg = StateGraph({F:{F:0.95, L:0.05}, L:{F:0.1, L:0.9}})
    prior = StateGraph({'START':{F:2./3., L:1./3.}})
    dg = DependencyGraph({0:[sg], 'START':[prior]})
    return dg

