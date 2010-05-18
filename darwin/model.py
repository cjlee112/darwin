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
    startNewSeg = False
    def __init__(self, state, var):
        self.state = state
        self.var = var

    def get_children(self, fromNode=None, parent=None):
        '''Get descendants in form [{node:pTransition}] '''
        results = []
        if fromNode is None:
            fromNode = self
        variables = self.var.graph[self]
        for varLabel, stateGraph in variables.items():
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

    def walk(self, g):
        if self in g:
            return
        g[self] = d = {}
        try:
            edges = self.var.seg.g[self]
        except KeyError:
            pass
        else:
            for dest, e in edges.items(): # edges w/in segment
                d[dest] = e
                dest.walk(g)
        try:
            substart = self.segmentGraph.start # this node contains subgraph
        except AttributeError:
            pass
        else:
            d[substart] = 1.
            substart.walk(g) # walk subgraph
        if self in self.var.seg.exitStates: # walk edges to next segment(s)
            try:
                segs = self.var.seg.segmentGraph.g[self.var.seg]
            except KeyError:
                pass
            else:
                for seg in segs:
                    for dest, edges in self.var.seg.segmentGraph.gRev[seg].items():
                        for sources, e in edges.items():
                            if self in sources:
                                ## print 'LINK', self, dest
                                d[dest] = e
                                dest.walk(g)
                                break

            for dest, sources in self.var.seg.segmentGraph.stops.items():
                try:
                    d[dest] = sources[self] # add edge to STOP if present
                except KeyError:
                    pass
                else:
                    g.setdefault(dest, {}) # ensure STOP node included in graph
                    
        

class StartNode(Node):
    startNewSeg = True
    def __init__(self, graph, obsLabel, parent=None):
        label = graph.get_start(obsLabel=obsLabel, parent=parent)
        Node.__init__(self, 'START', label)
        self.isub = 'START' # dummy value


class StopNode(Node):
    'represents STOP in compiled state-obs graph'
    def __init__(self, graph, obsLabel=None, parent=None):
        Node.__init__(self, 'STOP',
                      Variable(graph, 'STOP', obsLabel, parent))
        self.isub = 'STOP' # dummy value


class ContinuationNode(Node):
    'continuation point after completing subgraph contained in node origin'
    def __init__(self, origin, exitNode=None):
        if exitNode:
            var = origin.var.get_obs_label(exitNode.var.obsLabel)
            Node.__init__(self, id(self), var)
            self.exitNode = exitNode
        else: # just copy parent variable
            Node.__init__(self, id(self), origin.var)
        self.origin = origin

    def p_subgraph(self, logPobsDict):
        'calculate total log-prob for subgraph contained in self.origin'
        try:
            logP = self.origin.segmentGraph.logPtotal
        except AttributeError:
            logP = self.origin.segmentGraph.p_forward(logPobsDict)
        if hasattr(self, 'exitNode'):
            return self.origin.segmentGraph.fprob[self.origin.segmentGraph.start]\
                   [self.exitNode]
        else:
            return logP

    def walk(self, g):
        Node.walk(self, g)
        try:
            substop = self.exitNode
            g.setdefault(substop, {})[self] = 1.
        except AttributeError:
            for substop in self.origin.segmentGraph.stops:
                g.setdefault(substop, {})[self] = 1.


class Variable(object):
    '''a variable in a dependency graph'''
    def __init__(self, graph, label, obsLabel=None, parent=None, seg=None):
        self.graph = graph
        self.label = label
        self.obsLabel = obsLabel
        self.parent = parent
        self.seg = seg

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
        'get Variable with same label but new obsLabel'
        if parent is None:
            parent = self.parent
        return self.__class__(self.graph, self.label, obsLabel, parent,
                              self.seg)

    def get_obs(self):
        'return observations, filtered using tags if present'
        try:
            tags = self.tags
        except AttributeError:
            return self.obsLabel.get_obs()
        else:
            return self.obsLabel.get_subset(**tags).get_obs()


class MultiCondition(object):
    'represents a variable dependency conditioned on multiple variables'
    def __init__(self, conditions, targetVar, stateGraph):
        self.conditions = conditions
        self.targetVar = targetVar
        self.stateGraph = stateGraph
        self.states = {}
        self.sourceVars = [None] * len(conditions)
        self.nBind = 0

    def bind_var(self, var):
        'bind this variable to this MultiCondition'
        self.states[var] = []
        i = self.conditions.index(var.label)
        self.sourceVars[i] = var
        self.nBind += 1 # count total number of bound variables

    def __call__(self, node, targetVar, state=None, parent=None):
        'generate all possible multicond edge vectors for node in cumm. order'
        self.states[node.var].append(node)
        ## print 'Multicondition(%s)' % repr(node)
        if self.nBind < len(self.conditions):
            return {} # incomplete bindings, so do nothing now...
        l = []
        for vec in self.gen_vec2(self.sourceVars.index(node.var)):
            for dest, edge in self.stateGraph(vec, self.targetVar,
                                              parent=parent).items():
                l.append(MultiEdge(vec, dest, edge))
                ## print '\t', vec, dest, edge
        return MultiEdgeSet(l)

    def gen_vec2(self, iConst, i=0):
        'recursive generator of all possible state combinations'
        try:
            var = self.sourceVars[i]
        except IndexError:
            yield () # truncate the recursion
            return
        if i == iConst: # restricted to last state of this variable
            constState = (self.states[var][-1],)
            for substates in self.gen_vec2(iConst, i + 1):
                yield constState + substates
        else: # iterate over all possible states of this variable
            for state in self.states[var]:
                for substates in self.gen_vec2(iConst, i + 1):
                    yield (state,) + substates
            

class MultiEdge(object):
    def __init__(self, vec, dest, edge):
        self.vec = vec
        self.dest = dest
        self.edge = edge


class MultiEdgeSet(object):
    def __init__(self, edges):
        self.edges = edges

    def items(self):
        for edge in self.edges:
            yield edge.dest, edge


class DependencyGraph(object):
    labelClass = Variable

    def __init__(self, graph, joinTags=()):
        '''graph should be dict of {sourceLabel:{destLabel:stateGraph}}
        edges.  destLabel can be a Variable object (allowing you to
        specify a cross-connection to a node in another graph),
        or simply any Python value, in which case it will be treated
        as label for creating a Variable.

        joinTags is a tuple of zero or more tag keys which should
        be used for matching multicond variables by their observation
        labels.  For example, if you want multicond joins only
        on variables that share the same value of the matingID tag,
        set joinTags=("matingID",).'''
        multiCond = {}
        for k, targets in graph.items():
            if isinstance(k, tuple): # multiple conditions
                for sourceLabel in k:
                    multiCond[sourceLabel] = (k, targets)
        self.graph = graph
        self.multiCond = multiCond # {label:(sourceLabels, targets)}
        self.multiCondVar = {} # {sourceVar:{destVar:multiCond}}
        self.multiCondBind = {} # {sourceLabel:[multicond,...]}
        self.joinTags = joinTags

    def __getitem__(self, node):
        'get dict of {targetVariable:stateGraph} pairs'
        try:
            if node.var.graph is not self:
                raise KeyError
            d = self.graph[node.var.label]
        except (KeyError, AttributeError):
            d = {} # empty set -- no Markov edges
        results = {}
        for label,edge in d.items(): # process Markov edges
            if callable(label): # treat as function for generating target dict
                for newlabel, edge in label(node, **edge).items():
                    newVar = self.get_var(newlabel, obsLabel=node.var.obsLabel,
                                          seg=node.var.seg)
                    results[newVar] = edge
                    ## print 'added', newlabel, 'to results:', len(results)
            else:
                results[self.get_var(label, seg=node.var.seg)] = edge
        # next, process multicond edges
        try: # this variable already part of an existing multicond?
            mcResults = self.multiCondVar[node.var] # use pre-compiled multiconds
        except KeyError:
            try: # need to bind this variable to existing multicond?
                multiSet = self.multiCondBind[node.var.label]
                joinTags = tuple([node.var.obsLabel.tags[tag]
                                  for tag in self.joinTags])
                multiCond = multiSet[joinTags]
            except KeyError:
                try:
                    sourceLabels, targets = self.multiCond[node.var.label]
                except KeyError: # node.var has no multicond children
                    if len(results) > 1 or node.startNewSeg: # attach to new seg
                        self.set_var_segments(results.keys(), node.var.seg)
                    return results # so nothing further to do
                # create a new multicond relation
                joinTags = tuple([node.var.obsLabel.tags[tag]
                                  for tag in self.joinTags])
                for destLabel, sg in targets.items():
                    destVar = self.get_var(destLabel, obsLabel=node.var.obsLabel)
                    multiCond = MultiCondition(sourceLabels,
                                               destVar, sg)
                    multiCond.bind_var(node.var)
                    self.multiCondVar.setdefault(node.var, {}) \
                           [destVar] = multiCond
                    for label in sourceLabels: # link to labels
                        if label != node.var.label:
                            self.multiCondBind.setdefault(label, {}) \
                                     [joinTags] = multiCond
            else: # bind this variable to existing multicond
                self.multiCondVar.setdefault(node.var, {}) \
                           [multiCond.targetVar] = multiCond
                multiCond.bind_var(node.var)
                del self.multiCondBind[node.var.label]
            mcResults = self.multiCondVar[node.var]
        results.update(mcResults)
        self.set_var_segments(results.keys(), node.var.seg) # put in child segments
        return results

    def set_var_segments(self, vars, seg):
        'put these variables in separate segments as children of seg'
        for var in vars:
            joinTags = tuple([var.obsLabel.tags[tag] for tag in self.joinTags])
            var.seg = seg.get_child(var.label, joinTags)

    def get_start(self, **kwargs):
        'get START node for this graph'
        return self.get_var('START', **kwargs)

    def get_var(self, label, *args, **kwargs):
        'construct label object with specified args'
        if isinstance(label, self.labelClass):
            return label # already wrapped, no need to do anything
        return self.labelClass(self, label, *args, **kwargs)

    def __hash__(self):
        return id(self)

    def simulate_seq(self, n):
        'simulate markov chain of length n'
        segmentGraph = SegmentGraph(self, ObsSequenceSimulator())
        node = segmentGraph.start
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


class Segment(object):
    '''Unbranched segment of one or more Variables linked by compiled
    state graphs representing their allowed state --> state transitions.'''

    def __init__(self, segmentGraph, node=None):
        self.g = {} # {source:{dest:edge}}
        self.gRev = {} # {dest:{source:edge}}
        self.entryStates = set() # entry points in this segment
        self.exitStates = set() # exit points in this segment
        self.children = {} # segments that depend on this segment
        if node:
            self.add_entry(node)
        self.segmentGraph = segmentGraph
        segmentGraph.segments.append(self) # add to list of all segments        

    def add_entry(self, node):
        self.entryStates.add(node)
        node.var.seg = self

    def add_exit(self, node):
        self.exitStates.add(node)
        node.var.seg = self

    def add_edge(self, source, dest, edge):
        'add an internal edge within this segment'
        self.g.setdefault(source, {})[dest] = edge
        self.gRev.setdefault(dest, {})[source] = edge

    def get_child(self, k, joinTags):
        'return child segment with index value k'
        if k == 'STOP':
            return self.segmentGraph.stopSegment
        try:
            return self.children[k, joinTags]
        except KeyError:
            ## print 'created new child segment for var.label', k, joinTags
            self.children[k, joinTags] = seg = self.__class__(self.segmentGraph)
            return seg

    def get_predecessors(self):
        'generate all segments that this segment depends on'
        try:
            return self._predecessors
        except AttributeError:
            pass
        try:
            return self.segmentGraph.gRevSegs[self]
        except KeyError:
            return () # no predecessors

    def count_branches(self):
        try:
            return len(self.segmentGraph.g[self])
        except (KeyError, AttributeError):
            return 0

    def find_loop_starts(self):
        'scan predecessors of this segment to identify loops and their start points'
        l = self.get_predecessors()
        results = {}
        for i, branch1 in enumerate(l):
            deps1 = branch1.generate_dependencies()
            for j in range(i):
                branch2 = l[j]
                deps2 = branch2.generate_dependencies()
                for start1 in deps1:
                    if start1 in deps2: # found loop start
                        s = results.setdefault(start1, set()) # save loop start
                        s.add(branch1) # add its branches
                        s.add(branch2)
                        break
        self.loopStarts = results
        loopBranches = set()
        for s in results.values():
            loopBranches.update(s)
        self.loopBranches = loopBranches

    def generate_dependencies(self):
        "get this segment's dependencies, in order from closest to furthest"
        l = list(self.dep)
        for dep in self.dep:
            l += dep.generate_dependencies()
        return l

    def get_non_loop_branches(self, sources):
        'generate subset of sources that are not part of a loop'
        for source in sources:
            if source.var.seg not in self.loopBranches:
                yield source

    def get_loop_branches(self, sources):
        'generate loops ending at this segment, each with its sources'
        for loopStart, s in self.loopStarts.items():
            branches = [source for source in sources if source.var.seg in s]
            yield loopStart, branches


def p_segment(dest, f, logPobsDict, g):
    'symmetric calc for either forward or backwards prob on an unbranched segment'
    try:
        return f[dest] # already computed, so nothing further to do
    except KeyError:
        pass
    try:
        origin = dest.origin
    except AttributeError:
        pass
    else: # process this as a subgraph continuation point
        f[dest] = p_segment(origin, f, logPobsDict, g) \
                  + dest.p_subgraph(logPobsDict)
        return
    logP = []
    for src, edge in g[dest].items():
        logP.append(p_segment(src, f, logPobsDict, g) + logPobsDict[src]
                    + safe_log(edge))
    if logP: # non-empty list
        f[dest] = result = log_sum_list(logP)
    else:  # zero probability
        f[dest] = result = neginf
    return result
        
def p_branch(states, fstart, logPobsDict):
    l = []
    for source, p in states.items():
        l.append(fstart[source] + logPobsDict[source] + safe_log(p))
    return log_sum_list(l)


def p_loop(loopStart, branches, edges, fprob, fstart, logPobsDict):
    l = []
    for lsState in loopStart.exitStates: # sum over all loopStart states
        # calculate forward up to loopStart
        logP2 = fstart[lsState] + logPobsDict[lsState]
        loopCalc = fprob[lsState] # condition on loopStart
        for branch in branches:
            l2 = []
            for source, p in edges[branch].items():
                l2.append(loopCalc[source] + logPobsDict[source] +
                          safe_log(p))
            logP2 += log_sum_list(l2)
        l.append(logP2)
    return log_sum_list(l) # product of different loops




class SegmentGraph(object):
    '''reduced representation as a graph whose nodes are unbranched segments,
    with edges connecting them.  Each segment graph edge is either
    multiple-dependents (1:many) or multiple-conditions (many:1).'''
    def __init__(self, depGraph, obsLabel, parent=None):
        self.g = {} # {sourceSeg:set([destSeg1, destSeg2]), ...}
        self.gRev = {} # {destSeg:{dest:{(source1, source2,...):edge}}}
        self.gRevSegs = {}  # {destSeg:(sourceSeg1, sourceSeg2, ...)}
        self.stops = {} # {stop:{source:edge}}
        self.segments = []
        seg = Segment(self) # create START segment
        self.start = StartNode(depGraph, obsLabel, parent)
        seg.add_entry(self.start) # this is seg's only exit
        seg.dep = frozenset() # START has no dependencies
        self.stopSegment = Segment(self)

    def add_edge(self, source, dest, edge, multiDest):
        '''Save an edge from source node to dest node, either by saving to
        the appropriate segment, or as a segment-to-segment edge.  Multi-condition
        edges are indicated by a special class, MultiEdge.
        Assumes source.var segment already in self.varDict, but creates
        new segment for dest.var if not already in self.varDict.'''
        if isinstance(edge, MultiEdge):
            return self.add_multi_condition(dest, edge)
        elif dest in self.stops: # STOP edge        
            self.stops[dest][source] = edge
            source.var.seg.add_exit(source)
            return
        if dest.var.seg != source.var.seg:
            source.var.seg.add_exit(source)
            dest.var.seg.add_entry(dest)
            if dest.var.seg not in self.gRevSegs: # new seg-seg edge
                self.gRevSegs[dest.var.seg] = (source.var.seg,)
                self.gRev[dest.var.seg] = {}
                self.g.setdefault(source.var.seg, set()).add(dest.var.seg)
            self.gRev[dest.var.seg].setdefault(dest, {})[(source,)] = edge
        else:
            dest.var.seg.add_edge(source, dest, edge)

    def add_multi_condition(self, dest, edge):
        dest.var.seg.add_entry(dest)
        if dest.var.seg not in self.gRevSegs: # new seg-seg edge
            for source in edge.vec:
                self.g.setdefault(source.var.seg, set()).add(dest.var.seg)
            self.gRev[dest.var.seg] = {}
            self.gRevSegs[dest.var.seg] = tuple([source.var.seg for source in
                                                 edge.vec])
        self.gRev[dest.var.seg].setdefault(dest, {})[edge.vec] = edge.edge
        for source in edge.vec:
            source.var.seg.add_exit(source)

    def mark_end(self, node):
        'mark node as STOP'
        self.stops[node] = {}

    def get_stop_segment(self):
        'analyze the segment representing the STOP state'
        try: # use cached values if present
            return self.stopSegment, self.contSeg, self.termSegs, \
                   self.exitConts, self.exitEdges
        except AttributeError:
            pass
        if sum([len(t) for t in self.stops.values()]) == 0:
            raise ValueError('no exit to STOP in this SegmentGraph')
        termSegs = set() # find all segments with edge to STOP
        edges = {}
        conts = {}
        contSeg = None
        for dest, sources in self.stops.items():
            for source, p in sources.items():
                seg = source.var.seg
                if dest.var.obsLabel is not None:
                    if contSeg is None or contSeg == seg:
                        conts.setdefault(dest, {})[source] = p
                        contSeg = seg
                    else:
                        raise ValueError('multiple continuation segments!')
                else:
                    edges.setdefault(seg, {})[source] = p
                    termSegs.add(seg)
        if contSeg is not None:
            self.stopSegment._predecessors = tuple(termSegs) + (contSeg,)
        else:
            self.stopSegment._predecessors = tuple(termSegs)
        self.contSeg, self.termSegs, self.exitConts, self.exitEdges = \
                      contSeg, termSegs, conts, edges
        return self.stopSegment, contSeg, termSegs, conts, edges
    
    def p_backward(self, logPobsDict):
        'simple backwards probability calculation'
        logP = 0.
        b = {}
        for stop, d in self.stops.items():
            for node, edge in d.items():
                b[node] = safe_log(edge) # mark states with edges to STOP
        for seg in self.g[self.start.var.seg]: # product over multiple variables
            l = []
            for dest, d in self.gRev[seg].items(): # sum over all states
                edge = d[(self.start,)] 
                p_segment(dest, b, logPobsDict, seg.g) # calculate b[dest]
                l.append(b[dest] + safe_log(edge) + logPobsDict[dest])
            logP += log_sum_list(l)
        return logP

    def p_forward(self, logPobsDict):
        'loop-aware forward probability calculation'
        stopSegment, contSeg, termSegs, conts, edges = self.get_stop_segment()
        self.analyze_deps(stopSegment)
        for segment in self.segments: # analyze segment loop structure
            segment.find_loop_starts()
        self.fprob = ForwardProbability(self, logPobsDict) # probability graph
        fstart = self.fprob[self.start] # condition on START state
        logP = 0.
        for loopStart, branches in stopSegment.loopStarts.items():
            if contSeg in branches:
                raise ValueError('subgraph continuation in a loop!')
            logP += p_loop(loopStart, branches, edges, self.fprob, fstart,
                           logPobsDict)
        for sourceSeg, states in edges.items(): # non loop branches
            if sourceSeg not in stopSegment.loopBranches:
                logP += p_branch(states, fstart, logPobsDict)
        l = []
        for dest, states in conts.items():
            logPdest = p_branch(states, fstart, logPobsDict)
            fstart[dest] = logP + logPdest
            l.append(logPdest)
        if l:
            logP += log_sum_list(l)
        self.logPtotal = logP
        return logP

    def seed_forward(self, condition, f):
        'add terminations for forward calc based on condition'
        source = condition.var.seg
        f[condition] = 0. # terminate on this condition
        for destSeg in self.g[source]:
            if len(self.gRevSegs[destSeg]) == 1:
                # markov edge: terminate on condition's targets
                for dest, d in self.gRev[destSeg].items():
                    try:
                        f[dest] = safe_log(d[(condition,)])
                    except KeyError:
                        pass

    def analyze_deps(self, segment=None):
        'determine proximal dependencies for segment and its predecessors'
        if not hasattr(segment, 'dep'): # determine this segment's dependencies
            dep = set()
            for seg in segment.get_predecessors():
                dep.update(self.analyze_deps(seg))
            segment.dep = dep
        if segment.count_branches() > 1: # multidep exit becomes new dep
            return frozenset((segment,))
        else: # just echo its dependencies forward
            return segment.dep
        

class ForwardDict(object):
    'forward probability dictionary conditioned on a specified state'
    def __init__(self, parent, condition):
        self.parent = parent
        self.condition = condition
        self.f = {}
        self.parent.segmentGraph.seed_forward(condition, self.f)

    def __getitem__(self, dest):
        'get forward probability up to dest, conditioned on condition'
        try:
            return self.f[dest] # get from cache
        except KeyError:
            pass
        segment = dest.var.seg
        if dest not in segment.entryStates: # segment end, must recurse to start
            for state in segment.entryStates:
                self[state] # force calculation of all start states
            p_segment(dest, self.f, self.parent.logPobsDict, segment.gRev)
        elif len(segment.segmentGraph.gRevSegs[segment]) == 1: # markov start
            l = []  # markov dependency, just sum incoming edges
            for sources, edge in segment.segmentGraph.gRev[segment] \
                    [dest].items():
                l.append(self[sources[0]] +
                         self.parent.logPobsDict[sources[0]] +
                         safe_log(edge))
            self.f[dest] = log_sum_list(l)
        else: # multicond segment start
            l = []
            for sources, p in segment.segmentGraph.gRev[segment][dest]\
                    .items():
                logP = safe_log(p) # process non-loop branches
                ## print 'multicond edge -->', dest, logP
                for branch in segment.get_non_loop_branches(sources):
                    ## print '\tfrom', branch, self[branch] + self.parent.logPobsDict[branch]
                    logP += self[branch] + self.parent.logPobsDict[branch]
                for loopStart, branches in segment.get_loop_branches(sources):
                    lsStates = loopStart.exitStates
                    l2 = []
                    for lsState in lsStates: # sum over all loopStart states
                        # calculate forward up to loopStart
                        logP2 = self[lsState] + \
                                self.parent.logPobsDict[lsState]
                        loopCalc = self.parent[lsState] # condition on loopStart
                        for branch in branches:
                            ## print '\tloop', branch, loopCalc[branch] + self.parent.logPobsDict[branch]
                            logP2 += loopCalc[branch] + \
                                self.parent.logPobsDict[branch]
                        l2.append(logP2)
                    logP += log_sum_list(l2)
                l.append(logP)
            self.f[dest] = log_sum_list(l)
        return self.f[dest]

    def __setitem__(self, dest, logP):
        self.f[dest] = logP


class ForwardProbability(object):
    def __init__(self, segmentGraph, logPobsDict):
        self.segmentGraph = segmentGraph
        self.logPobsDict = logPobsDict
        self.cond = {}

    def __getitem__(self, condition):
        try:
            return self.cond[condition]
        except KeyError:
            self.cond[condition] = d = ForwardDict(self, condition)
            return d

                        
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
        self.tags = tags

    def get_obs(self):
        for k,v in self.tags.items():
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
        newtags = self.tags.copy()
        newtags.update(tags) # allow new tags to overwrite old tags
        return self.__class__(self.obsSet, **newtags)

    def get_next(self, **kwargs): # dummy method for LinearState compatibility
        return self

    def get_tag_dict(self, tag):
        'get dict of {k:ObsSubset(tag=k)} subsets of this subset'
        d = {}
        for k in self.obsSet._tags[tag]:
            d[k] = self.get_subset(**{tag:k})
        return d

    def __hash__(self):
        tags = self.tags.items()
        tags.sort() # guarantee consistent order for comparison
        return hash((self.obsSet, tuple(tags)))

    def __cmp__(self, other):
        try:
            return cmp((self.obsSet, self.tags),
                       (other.obsSet, other.tags))
        except AttributeError:
            return cmp(id(self), id(other))

    def __repr__(self):
        return 'ObsSet(%s, %s)' % (self.obsSet.name,
                  ', '.join([('%s=%s' % t) for t in self.tags.items()]))


class BranchGenerator(object):
    def __init__(self, label, stateGraph, iterTag=None, **tags):
        self.label = label
        self.stateGraph = stateGraph
        self.iterTag = iterTag
        self.tags = tags

    def __call__(self, node):
        'return {Variable:stateGraph} dict'
        d = {}
        subset = node.var.obsLabel # assume obsLabel iterable
        if self.tags: # filter using these tag constraints
            subset = subset.get_subset(** self.tags)
        if self.iterTag: # generate branches for each value of this tag
            subset = subset.get_tag_dict(self.iterTag).values()
        for obsLabel in subset:
            try: # already a Variable
                newLabel = self.label.get_obs_label(obsLabel)
            except AttributeError: # need to create a new Variable
                newLabel = node.var.graph.get_var(self.label)
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
        self.stop = StopNode(self.dependencyGraph)
        self.segmentGraph = SegmentGraph(dependencyGraph, obsLabel)
        self.start = self.segmentGraph.start
        self.compiledGraph = {}
        self.compiledGraphRev = {}
        self.logPobsDict = {self.start:self.start.log_p_obs()}
        self.b = {}
        compile_graph(self.compiledGraph, self.compiledGraphRev,
                      self.start, self.b, self.logPobsDict, logPmin, None,
                      self.segmentGraph)

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
        g = {}
        self.start.walk(g) # generate node graph
        outfile = file(filename, 'w')
        try:
            save_graphviz(outfile, g, None, **kwargs)
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

    def __call__(self, fromNode, targetVar, state=None, parent=None):
        '''return dict of {node:pTransition} from fromNode'''
        results = {}
        if state is None:
            state = fromNode.state
        if targetVar.obsLabel is not None:
            obsLabel = targetVar.obsLabel
        else:
            obsLabel = fromNode.var.obsLabel
        for dest,edge in self.graph[state].items():
            newNode = dest(fromNode, targetVar, obsLabel, edge, parent)
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

    def __call__(self, fromNode, targetVar, obsLabel, edge, parent):
        '''default move operator just echoes the next obs'''
        try:
            newLabel = targetVar.get_obs_label(obsLabel.get_next(state=self),
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
        obsList = node.var.get_obs() # use filtered obs for this variable
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


class VarFilterState(State):
    'filters obs according to associated variable name via tag var=name'
    def __call__(self, fromNode, targetVar, obsLabel, edge, parent):
        obsLabel = obsLabel.get_subset(var=targetVar.label)
        newLabel = targetVar.get_obs_label(obsLabel, parent)
        return Node(self, newLabel)


class FilterState(State):
    'filters observations according to tag=value kwargs'
    def __init__(self, name, emission, **tags):
        State.__init__(self, name, emission)
        self.tags = tags
        
    def __call__(self, fromNode, targetVar, obsLabel, edge, parent):
        newLabel = targetVar.get_obs_label(obsLabel.get_subset(** self.tags),
                                             parent)
        return Node(self, newLabel)


class SilentState(State):
    'state that emits nothing'
    def __init__(self, name):
        self.name = name

    def __call__(self, fromNode, targetVar, obsLabel, edge, parent):
        obsLabel = obsLabel.get_next(empty=True) # we emit nothing
        newLabel = targetVar.get_obs_label(obsLabel, parent)
        return Node(self, newLabel)

class StopState(State):
    def __init__(self, useObsLabel=True):
        State.__init__(self, 'STOP', None)
        self.useObsLabel = useObsLabel
    def __call__(self, fromNode, targetVar, obsLabel, edge, parent):
        if self.useObsLabel:
            obsLabel = obsLabel.get_next(empty=True) # we emit nothing
        else:
            obsLabel = None
        return StopNode(targetVar.graph, obsLabel, parent)
        
class LinearStateStop(StopState):
    def __call__(self, fromNode, targetVar, obsLabel, edge, parent):
        '''Only return STOP node if at the end of the obs set '''
        if not State.__call__(self, fromNode, targetVar, obsLabel, edge, parent):
            # exhausted obs, so transition to STOP
            return StopNode(targetVar.graph, None, parent)

class SeqMatchState(State):
    def __init__(self, name, emission, match_f):
        State.__init__(self, name, emission)
        self.match_f = match_f

    def __call__(self, fromNode, targetVar, obsLabel, edge, parent):
        if obsLabel.stop < len(obsLabel.seq):
            s = self.match_f(obsLabel.seq[obsLabel.stop:]) # run matcher
            if s:
                obsLabel = obsLabel.get_next(length=len(s))
                return Node(self, targetVar.get_obs_label(obsLabel, parent))


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


def compile_subgraph(g, gRev, node, b, logPobsDict, logPmin, segmentGraph):
    'recurse to subgraph of this node; creates node.stops as list of endnodes'
    #segmentGraph.varDict[node.var].varStates.setdefault(node.var, set()).add(node)
    node.stops = {} # for list of unique stop nodes in this subgraph
    node.segmentGraph = SegmentGraph(node.state.subgraph, node.var.obsLabel,
                                     node)
    start = node.segmentGraph.start
    logPobsDict[start] = 0.
    g[node] = ({start:1.},) # save edge from node to start of its subgraph
    gRev.setdefault(start, {})[node] = 1.
    print '\n\npush', node
    compile_graph(g, gRev, start, b, logPobsDict, logPmin, node,
                  node.segmentGraph) # subgraph
    try:
        node.segmentGraph.get_stop_segment() # find exitConts for extending node
    except ValueError: # no exit to STOP
        print "pop", node, '\tNO EXIT\n\n'
        node.stops = ()
        return
    print "pop", node, len(node.segmentGraph.exitConts), '\n\n'
    if len(node.segmentGraph.exitConts) > 0: # use the exitConts as continuation states
        node.stops = [ContinuationNode(node, c)
                      for c in node.segmentGraph.exitConts]
    else: # create a single continuation state
        node.stops = (ContinuationNode(node),)


def compile_graph(g, gRev, node, b, logPobsDict, logPmin, parent, segmentGraph):
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
        compile_subgraph(g, gRev, node, b, logPobsDict, logPmin, segmentGraph)
        fromNodes = node.stops # generate edges from subgraph stop node(s)
    else:
        fromNodes = (node,) # generate edges from this node
    for fromNode in fromNodes:
        targets = []
        variables = node.get_children(fromNode, parent) # regular edges
        multiDest = len(variables) > 1
        for varStates in variables: # generate multiple dependencies
            d = {}
            for dest,edge in varStates.items(): # multiple states sum...
                if not hasattr(dest, 'isub'):
                    dest.isub = len(targets)
                try:
                    logPobs = logPobsDict[dest]
                except KeyError:
                    logPobsDict[dest] = logPobs = dest.log_p_obs()
                    if dest.state == 'STOP': # terminate the path
                        if parent is not None: # add to parent's stop-node list
                            segmentGraph.mark_end(dest)
                            # parent.stops[dest] = None
                        else: # backwards probability of terminal node = 1
                            segmentGraph.mark_end(dest)
                            b[dest] = 0.
                        g[dest] = () # prevent recursion on dest
                if logPobs <= logPmin:
                    continue # truncate: do not recurse to dest
                ## print fromNode, '--->', dest
                segmentGraph.add_edge(fromNode, dest, edge, multiDest) # rm next 2 lines
                gRev.setdefault(dest, {})[fromNode] = edge # save reverse graph
                d[dest] = edge
                if dest not in g: # subtree not already compiled, so recurse
                    compile_graph(g, gRev, dest, b, logPobsDict, logPmin, parent,
                                  segmentGraph)
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


class Graphviz(object):
    'simple dot format writer; handles nested subgraphs correctly, unlike gvgen'
    def __init__(self):
        self.content = {}
        self.edges = {}
        self.nodes = []
        self.toplevel = []

    def add_node(self, label, parent=None):
        i = len(self.nodes)
        self.nodes.append(dict(label=label))
        if parent is not None:
            try:
                self.content[parent].append(i)
            except KeyError:
                self.content[parent] = [i]
        else:
            self.toplevel.append(i)
        return i

    def add_edge(self, node1, node2, label=None):
        self.edges.setdefault(node1, {})[node2] = label

    def print_branch(self, ifile, node, level=1):
        try:
            children = self.content[node]
        except KeyError:
            print >>ifile, '  ' * level + self.node_repr(node) + ' [label="' \
                  + self.nodes[node]['label'] + '"];'
        else:
            print >>ifile, '  ' * level + 'subgraph ' + self.node_repr(node) \
                  + ' {\n' + '  ' * level + 'label="' \
                  + self.nodes[node]['label'] + '";'
            for child in children:
                self.print_branch(ifile, child, level + 1)
            print >>ifile, '  ' * level + '}'

    def node_repr(self, node):
        if node in self.content:
            return 'cluster' + str(node)
        else:
            return 'node' + str(node)

    def print_dot(self, ifile):
        print >>ifile, 'digraph G {\ncompound=true;'
        for node in self.toplevel:
            self.print_branch(ifile, node)
        for node, edges in self.edges.items():
            for node2 in edges:
                print >>ifile, self.node_repr(node) + '->' + self.node_repr(node2) \
                      + ';'
        print >>ifile, '}'


def save_graphviz(outfile, g, post_f=None, majorColor='red',
                  label_f=lambda x:str(x.state)):
    '''generate dot file from graph g
    post_f, if not None, must be function that returns posterior probability
    of a node
    majorColor is the color assigned to nodes with > 50% probability.'''
    gd = Graphviz()
    gNodes = {}
    gClusters = {}
    colors = ('black', 'green', 'blue', 'red')
    iseg = 0
    for node in g:
        try:
            parent = gClusters[node.var]
        except KeyError:
            try:
                parentSeg = gClusters[node.var.seg]
            except KeyError:
                parentSeg = gClusters[node.var.seg] = \
                            gd.add_node('segment%d' % iseg)
                iseg += 1
            parent = gClusters[node.var] = gd.add_node(str(node.var), parentSeg)
        gNodes[node] = gd.add_node(label_f(node), parent)
    for node, targets in g.items():
        for dest, edge in targets.items():
            gd.add_edge(gNodes[node], gNodes[dest])
    gd.print_dot(outfile)

