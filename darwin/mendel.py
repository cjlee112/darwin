# -*- coding: utf-8 -*-

from scipy import stats
import random


class Allele(object):
    def __init__(self, name, phenotype):
        self.name = name
        self.phenotype = phenotype
    def rvs(self, n):
        'get phenotype sample of size n'
        return (self.phenotype.rvs(n))
    def __repr__(self):
        return self.name

class DominantAllele(Allele):
    def __add__(self, other):
        return self

class RecessiveAllele(Allele):
    def __add__(self, other):
        return other

class Chromosome(list):
    '''list of (pos,allele) tuples, where x is position expressed as a
    genetic distance in Morgans, e.g.
    chr1 = Chromosome([(0.01,Wh),(0.3,Wr)])'''
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
    '''dictionary of chromosomes whose keys are chromosome IDs,
    and whose associated values are tuples each containing
    a pair of Chromosome instances.
    e.g.
    g = Genome({1:(chr1a,chr1b), 2:(chr2a,chr2b)})'''
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

