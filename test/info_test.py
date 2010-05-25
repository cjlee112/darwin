from darwin import entropy
from scipy import stats
import numpy
import random

def calc_entropy(data, m):
    vec = entropy.box_entropy(data, m)
    return numpy.average(vec.sample)

def correlation_test(n=100, yErr=0.1, m=7):
    xpdf = stats.norm(0, 1)
    epdf = stats.norm(0, yErr)
    x = xpdf.rvs(n)
    y = (1. - yErr) * x + epdf.rvs(n)
    Hxy = calc_entropy([(x[i], y[i]) for i in range(n)], m)
    Hx = calc_entropy(x, m)
    Hy = calc_entropy(y, m)
    Hy_x = Hxy - Hx
    Ixy = Hy - Hy_x
    print Hx, Hy, Hxy, Hy_x, Ixy
    
