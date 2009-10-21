
from scipy import stats
import random


class Locus(object):
    def __init__(self,name,phenotype):
        self.name=name
        self.phenotype=phenotype
    def __add__(self,other):
        'get phenotype of combined self + other locus'
        if isinstance(self,DominantLocus):
            return self
        elif isinstance(other,DominantLocus):
            return other
        else:
            return self
    def __call__(self,n):
        'get phenotype sample of size n'
        return self.phenotype.rvs(n)
    def __repr__(self):
        return self.name
class DominantLocus(Locus): pass
class RecessiveLocus(Locus): pass

class Chromosome(list):
    def __mul__(self,other):
        'return recombinants of self and other'
        swap = 0
        recomb = (Chromosome([self[0]]),Chromosome([other[0]]))
        for i in range(1,len(self)):
            distance = self[i][0] - self[i-1][0]
            swap = (swap + int(stats.poisson(distance).rvs(1))) % 2 # RECOMBINATION
            recomb[swap].append(self[i])
            recomb[1-swap].append(other[i])
        return recomb


class Genome(dict):
    def __mul__(self,other):
        'return single progeny of self and other'
        child = Genome()
        for k,chrPair in self.items():
            chrSelf = (chrPair[0]*chrPair[1])[random.randint(0,1)]
            otherPair = other[k]
            chrOther = (otherPair[0]*otherPair[1])[random.randint(0,1)]
            child[k] = (chrSelf,chrOther)
        return child
    def __repr__(self):
        s = '{'
        afterFirst = False
        for k,chrPair in self.items():
            if afterFirst:
                s += ', '
            afterFirst = True
            s += repr(k)+': ['
            for i in range(len(chrPair[0])):
                if i>0:
                    s += ', '
                s += '(%s,%s)' % (repr(chrPair[0][i][1]),repr(chrPair[1][i][1]))
            s += ']'
        return s+'}'
    def __call__(self):
        'get phenotype list'
        l = []
        for k,chrPair in self.items():
            for i in range(len(chrPair[0])):
                l.append(chrPair[0][i][1] + chrPair[1][i][1])
        return l

