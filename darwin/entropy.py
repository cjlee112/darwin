# -*- coding: utf-8 -*-
import numpy
from math import log, pi, sqrt

def calc_dist(vectors):
    'calculate euclidean distance^2 for every vector pair'
    a = numpy.core.array(vectors)
    n = len(vectors)
    z = numpy.core.zeros((n))
    l = []
    for i in range(n):
        b,c = numpy.ix_(z,a[i])
        d = b+c
        e = a-d
        l.append(numpy.core.sum(e*e,1))
    return l

def calc_entropy(vectors,n=2):
    'estimate entropy from sample of points in high-dim space'
    ndim = len(vectors[0])
    fac = 1
    for i in xrange(2,ndim/2):
        fac *= i
    log_b = log(n*ndim*fac*.5/len(vectors))-ndim*.5*log(pi)
    distances = calc_dist(vectors)
    log_p = 0.
    for a in distances:
        a.sort()
        r2 = (a[n]+a[n-1])/2 # AVERAGE DISTANCE CONTAINING n POINTS
        log_p -= ndim*.5*log(r2)
    return -log_b-log_p/len(vectors) # AVERAGE -log(p)



def calc_box(vectors,dx):
    'calculate number of points within box width dx'
    dx *= .5
    a = numpy.core.array(vectors)
    n = len(vectors)
    z = numpy.core.zeros((n))
    l = numpy.core.zeros((n))
    for i in range(n):
        b,c = numpy.ix_(z,a[i])
        d = b+c
        e = numpy.core.abs(a-d) # GET DIFFERENCE VECTORS
        l[i] = numpy.core.sum(numpy.core.alltrue(numpy.core.less(e,dx),1))
    return l

def calc_density(vectors,dx):
    return numpy.core.divide(calc_box(vectors,dx),len(vectors)*pow(dx,len(vectors[0])))


class SampleEstimator(object):
    def __init__(self, sample=None, mean=None, variance=None):
        if variance is not None: # store values supplied by user
            self.mean = mean
            self.variance = variance
            return
        self.mean = numpy.average(sample)
        diffs = sample - self.mean
        self.variance = numpy.average(diffs * diffs) / len(sample)

    def __neg__(self):
        return self.__class__(mean= -self.mean, variance=self.variance)

    def __sub__(self, other):
        return self.__class__(mean= self.mean - other.mean,
                              variance=self.variance + other.variance)

    def __add__(self, other):
        return self.__class__(mean= self.mean + other.mean,
                              variance=self.variance + other.variance)

    def __isub__(self, other):
        self.mean -= other.mean
        self.variance += other.variance
        return self

    def __iadd__(self, other):
        self.mean += other.mean
        self.variance += other.variance
        return self
    
    def get_bound(self, p=0.05):
        '''calculate percentile bound, e.g. p=0.05 gives lower bound
        with 95% confidence'''
        if p < 0.5:
            return self.mean - sqrt(0.5 * self.variance / p)
        else:
            return self.mean + sqrt(0.5 * self.variance / (1. - p))

class LogPVector(object):
    def __init__(self, sample):
        self.sample = sample
    def __neg__(self):
        return LogPVector(-self.sample)
    def __sub__(self, other):
        return SampleEstimator(self.sample -  other.sample)


def box_entropy(vectors, m):
    '''calculate differential entropy using specified number of points m
    vectors: sampled data points;
    m: number of nearest points to include in each density-sampling box'''
    if not hasattr(vectors, 'ndim'):
        a = numpy.core.array(vectors)
    else:
        a = vectors
    n = len(a)
    if a.ndim == 1:
        a = a.reshape((n, 1))
    ndim = a.shape[1]
    rows = numpy.arange(n)
    e1 = numpy.core.zeros((n))
    e2 = numpy.core.zeros((n))
    nm = numpy.core.zeros((n))
    for i in range(n):
        d = numpy.core.abs(a-a[i]) # GET DIFFERENCE VECTORS
        e = d[rows,numpy.argmax(d,1)] # GET LARGEST DIFFERENCE
        e.sort()
        for m2 in xrange(m,n): # ENSURE THAT VOLUME IS NOT ZERO...
            if e[m2]>0.: break
        e1[i] = e[m2-1]
        e2[i] = e[m2]
        nm[i] = m2 - 1 # don't count center point -- unbiased calculation
    hvec = ndim*numpy.log(2.*e2) + numpy.log(0.5*((e1/e2)**ndim)+0.5) \
           - numpy.log(nm/(n - 1))
    return LogPVector(hvec)

## def box_entropy(vectors,m):
##     d = box_density(vectors,m)
##     return -sum(numpy.core.log(d))/len(vectors)

'''
argh, entropy calculation in a high-dim space via MC sampling is not easy.
The problem is that with high-dimensionality only a hypersphere with a radius
substantially larger than the
side-length of the entire bounding box contains more than an
infinitesimal fraction of the total volume, and thus has any prayer of
containing a sample point.  But then you run into problems with the
sample box going outside the bounding box, and also the bias that every
sample box is of course centered on a data point...
'''


def sample_Le(vectors, model):
    '''calculate average log-likelihood and bound by LoLN.
    vectors: sampled data points;
    model: likelihood model with pdf() method'''
    logP = numpy.log(model.pdf(vectors))
    return LogPVector(logP)


def d2_entropy(v, m):
    '''experimental method for estimating entropy using mean-squared
    distance of a sample of m-1 closest points around a given point'''
    v.sort()
    n = len(v)
    r = m
    l = 0
    dvec = numpy.core.zeros((n))
    for c in range(n):
        while r < n and abs(v[l] - v[c]) > abs(v[r] - v[c]):
            r += 1
            l += 1
        d2 = 0.
        for i in range(l, r):
            d = v[i] - v[c]
            d2 += d * d
        dvec[c] = d2
    dvec = numpy.core.ones((n)) / numpy.sqrt(dvec * (12./(m - 1.)))
    return LogPVector( - numpy.log(dvec * ((m - 1.) / (n - 1.))))


def grow_interval(target, l, r, yi):
    'expand interval on either left or right depending on which is closer'
    if l > 0 and \
        (r == n or abs(target[l-1][0] - yi) < abs(target[r][0] - yi)):
        return l - 1, r, abs(target[l-1][0] - yi)
    else:
        return l, r + 1, abs(target[r][0] - yi)


def find_rect(target, m, i, ratio):
    '''target: sorted list of (y, x, ix) tuples;
    ratio: x/y ratio for desired box;
    m: number of points to use for density estimation;
    i: index of the center point in target[];
    [l:r] is the current search interval'''
    n = len(target)
    if n < m:
        raise IndexError('less than m points??')
    xi = target[i][1]
    yi = target[i][0]
    l = i
    r = i + 1
    while r - l < m: # must enclose at least m points
        l,r,dy = grow_interval(target, l, r, yi)
    dx = dy * ratio
    while True:
        nin = 0
        dxNext = float('inf') # positive infinity
        for j in range(l, r): # count points within (dx, dy) box
            if abs(target[j][1] - xi) <= dx:
                nin += 1
            elif abs(target[j][1] - xi) < dxNext:
                dxNext = abs(target[j][1] - xi)
        if nin >= m:
            return dx,dy,m,rect_points(target, dx, l, r, xi)
        lnew,rnew,dnew = grow_interval(target, l, r, yi)
        if dnew < dxNext / ratio: # expand [l,r]
            l,r,dy = lnew,rnew,dnew
            dx = dy * ratio
        else: # expand dx (within [l,r])
            dx = dxNext
            dy = dx / ratio

def rect_points(target, dx, l, r, xi):
    '''return list of points in rectangle (dx,dy) as tuples (x-xi,y,x,ix)
    sorted in order of increasing distance from xi'''
    l = []
    for j in range(l, r): # count points within (dx, dy) box
        if abs(target[j][1] - xi) <= dx:
            l.append((abs(target[j][1] - xi),) + target[j])
    l.sort()
    return l

## def find_x(source, x):
##     l = 0
##     r = len(source)
##     while :
##         mid = (l + r) / 2
##         if source[mid][0] > x:
##             r = mid
##         else:
##             l = 

def cond_density(vectors, m):
    '''calculate conditional density using rectangle method,
    m: number of points to use as sample '''
    source = vectors.copy()
    source.sort()
    target = []
    for i,p in enumerate(source):
        target.append((p[1],p[0],i))
    target.sort()
    n = len(target)
    for i in range(n):
        # TODO: need to compute rectangle side ratio around this point
        dx,dy,m2,plist = find_rect(target, m, i, ratio)
        dxmid = (plist[-2][0] + plist[-1][0]) / 2. # midpoint betw. m, m-1
        ywidth = 2. * dxmid / ratio
        l = source.searchsorted(target[i][1] - dxmid) # total w/in dxmid
        r = source.searchsorted(target[i][1] + dxmid, side='right')
        density = (m2 - 2) / ((r - l - 1) * ywidth)
        
