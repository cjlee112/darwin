from darwin.model import *

from darwin import parser

match_f = parser.REMatcher(r"(\d+)")
em = parser.SimpleEmitter(10)
number = SeqMatchState('#', em, match_f)

space = LinearState('w', EmissionDict({' ':1.}))

stop = LinearStateStop()

sg = StateGraph({number:{space:1., stop:1.}, space:{number:1., stop:1.}})

prior = StateGraph({'START':{number:1./2., space:1./2.}})

hmm = NodeGraph({'theta':{'theta':sg}, 'START':{'theta':prior}})

obsGraph = ObsSequence('12 33 495')
m = Model(hmm, (obsGraph,))

logPobs = m.calc_fb()

m.save_graphviz('test.dot')
