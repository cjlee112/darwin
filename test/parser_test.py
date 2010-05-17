from darwin.model import *

from darwin import parser

def number_test(program='12 33 495'):
    match_f = parser.REMatcher(r"(\d+)")
    em = parser.SimpleEmitter(10)
    number = SeqMatchState('#', em, match_f)

    space = LinearState('w', EmissionDict({' ':1.}))

    stop = LinearStateStop()

    sg = StateGraph({number:{space:1., stop:1.}, space:{number:1., stop:1.}})

    prior = StateGraph({'START':{number:1./2., space:1./2.}})

    hmm = DependencyGraph({'theta':{'theta':sg}, 'START':{'theta':prior}})

    obsLabel = ObsSequenceLabel(program)
    m = Model(hmm, obsLabel)

    logPobs = m.calc_fb()
    return m

#m.save_graphviz('test.dot')

def math_test(program='2+7'):
    match_f = parser.REMatcher(r"(\d+)")
    em = parser.SimpleEmitter(10)
    number = SeqMatchState('#', em, match_f)

    stopNow = StopState() # stop immediately

    addExpr = SilentState('add')
    mathExpr = SilentState('math')
    plus = LinearState('+', EmissionDict({'+':1.}))
    sg = StateGraph({number:{plus:1.}, plus:{mathExpr:1.},
                     mathExpr:{stopNow:1.}})
    prior = StateGraph({'START':{number:1.}})
    subgraph = DependencyGraph({'theta':{'theta':sg}, 'START':{'theta':prior}})
    addExpr.subgraph = subgraph

    sg = StateGraph({number:{stopNow:1.}, addExpr:{stopNow:1.}})
    prior = StateGraph({'START':{number:0.5, addExpr:0.5}})
    subgraph = DependencyGraph({'theta':{'theta':sg}, 'START':{'theta':prior}})
    mathExpr.subgraph = subgraph
   
    stopFinal = StopState(useObsLabel=False)
    sg = StateGraph({mathExpr:{stopFinal:1.}})
    prior = StateGraph({'START':{mathExpr:1.}})
    hmm = DependencyGraph({'theta':{'theta':sg}, 'START':{'theta':prior}})
    obsLabel = ObsSequenceLabel(program)
    m = Model(hmm, obsLabel)

    logPobs = m.calc_fb()
    return m
    
