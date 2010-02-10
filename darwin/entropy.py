# -*- coding: utf-8 -*-
import numpy
from math import log, pi, sqrt
import random

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

def discrete_box_entropy(vectors, m):
    # Check for observations as strings for the multinomial.
    if isinstance(vectors[0], str):
        count = 0
        mapping = dict()
        for v in vectors:
            if v not in mapping:
                mapping[v] = count
                count += 1
        vectors = [mapping[v] for v in vectors]
        if len(set(vectors)) == 1:
            return LogPVector(1)
    return box_entropy(vectors, m)

def box_entropy(vectors, m):
    '''calculate differential entropy using specified number of points m
    vectors: sampled data points;
    m: number of nearest points to include in each density-sampling box'''
    # handle string data from multinomial
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

def discrete_sample_Le(vectors, model):
    logP = numpy.log(model.pmf(vectors))
    return LogPVector(logP)

def sample_Le(vectors, model):
    '''calculate average log-likelihood and bound by LoLN.
    vectors: sampled data points;
    model: likelihood model with pdf() method'''
    # Detect discrete vs. continuous models.
    logP = numpy.log(model.pdf(vectors))
    return LogPVector(logP)


def d2_density(v, m):
    '''experimental method for estimating density using mean-squared
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
    return dvec * ((m - 1.) / (n - 1.))

def d_density(v, m):
    '''experimental method for estimating density using mean
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
            d2 += abs(v[i] - v[c])
        dvec[c] = d2
    dvec = numpy.core.ones((n)) / (dvec * (4./(m - 1.)))
    return dvec * ((m - 1.) / (n - 1.))

def d2_entropy(v, m, use_d2=True):
    if use_d2:
        dvec = d2_density(v, m)
    else:
        dvec = d_density(v, m)
    return LogPVector( - numpy.log(dvec))


def grow_interval(target, l, r, cy, n, k=0):
    'expand interval on either left or right depending on which is closer'
    if l > 0 and \
        (r == n or abs(target[l-1][k] - cy) < abs(target[r][k] - cy)):
        return l - 1, r, abs(target[l-1][k] - cy), l - 1
    else:
        return l, r + 1, abs(target[r][k] - cy), r 



def grow_block(target, l, r, cy, k=0):
    'expand interval by one or more points with same y-value'
    n = len(target)
    dy = None
    while l > 0 or r < n:
        lnew,rnew,dnew,ynew = grow_interval(target, l, r, cy, n, k)
        if dy is not None and dnew > dy:
            return l, r, dy, inew
        l, r, dy, inew = lnew, rnew, dnew, ynew
    if dy is None:
        raise StopIteration # exhausted target
    return l, r, dy, inew


def find_radius(target, l, r, cy, m, k=0, minradius=1.):
    '''find box enclosing at least m points with non-zero volume.
    target must be sorted list / array of tuples.
    Then return radius for mid-point between m-th furthest point and
    m-1 th furthest point.  If all points have identical coordinates,
    (zero radius) return minradius.'''
    n = len(target)
    while r - l < m or target[r-1][k] - target[l][k] <= 0:
        try:
            l, r, dy, inew = grow_interval(target, l, r, cy, n, k)
        except StopIteration:
            break
    dist = [abs(target[j][k] - cy) for j in range(l, r)]
    dist.sort()
    yradius = (dist[-2] + dist[-1]) / 2. # midpoint betw 2 furthest pts
    if yradius <= 0.:
        yradius = minradius
    return l, r - 1, yradius


def find_ratio(source, target, i, m):
    '''find the ratio of side lengths enclosing at least m points
    (and non-zero volume) on source vs
    target axes, centered at target point i'''
    cy = target[i][0]
    ly, ry, yradius = find_radius(target, i, i + 1, cy, m)
    ix = target[i][2] # index of this point in source list
    lx, rx, xradius = find_radius(source, ix, ix + 1, source[ix][0], ry - ly)
    if rx - lx > ry - ly: # need to expand yradius to match number of points
        ly, ry, yradius = find_radius(target, ly, ry, cy, rx - lx)
    return xradius / yradius


def find_rect(target, i, m, ratio, nonZero=False):
    '''target: sorted list of (y, x, ix) tuples;
    ratio: x/y ratio for desired box;
    m: number of points to use for density estimation;
    i: index of the center point in target[];
    nonZero=True forces find_rect() to return a pointset with non-zero
    x radius (by expanding m if necessary until the xradius > 0).
    returns #points, xradius, yradius, points-list'''
    n = len(target)
    if n < m:
        raise IndexError('less than m points??')
    yi,xi = target[i][:2]
    l, r, dy, ynew = grow_block(target, i, i + 1, yi)
    dxOuter = dy * ratio
    dxLast = dxPrev = 0.
    while True:
        xradius = (dxLast + dxPrev) / 2. # midway between furthest two points
        yradius = xradius / ratio
        nin = 0
        dxNext = float('inf') # positive infinity
        dxSum = 0.
        for j in range(l, r): # count points within (xradius, yradius)
            dx = abs(target[j][1] - xi)
            if dx <= xradius and abs(target[j][0] - yi) <= yradius :
                nin += 1
                dxSum += dx
            elif dx > dxLast and dx < dxNext:
                dxNext = dx
        if nin > m and yradius > 0. and (not nonZero or dxSum > 0.):
            return nin - 1, xradius, yradius, \
                   rect_points(target, l, r, xi, yi, xradius, yradius)
        dxPrev = dxLast
        if dxOuter is None and dxNext == float('inf'):  # no more points
            return nin - 1, xradius, yradius, \
                   rect_points(target, l, r, xi, yi, xradius, yradius)
        elif dxOuter is None or dxNext < dxOuter:
            dxLast = dxNext
        else: # expand our (l,r) interval
            dxLast = dxOuter
            try:
                l,r,dy,ynew = grow_block(target, l, r, yi)
                dxOuter = dy * ratio
            except StopIteration:
                dxOuter = None # no more points in target list


def rect_points(target, l, r, cx, cy, xradius, yradius):
    'get list of points inside rectangle defined by (cx,cy) and radii'
    points = []
    for j in range(l, r):
        if abs(target[j][1] - cx) <= xradius \
           and abs(target[j][0] - cy) <= yradius:
            points.append(j)
    return points

def points_radius(target, points, i, k=0):
    '''get radius based on midpoint between furthest
    and next furthest points '''
    cx = target[i][k]
    dist = [abs(target[j][k] - cx) for j in points]
    dist.sort()
    return (dist[-2] + dist[-1]) / 2.


def get_ratio(xdata, ydata, m, nsample):
    '''calculate the ratio of xradius vs. yradius via sampling.
    First calculates ratio on 1D projections of the data onto x, y axes.
    Then recalculates ratio using 2D rectangle sampling method using the
    ratio calculated from 1D data.
    xdata, ydata must both be sorted'''
    n = len(xdata)
    sample = [random.randrange(n) for i in range(nsample)]
    xrads = [find_radius(xdata, i, i + 1, xdata[i][0], m) for i in sample]
    xrads = [(t[2] / (t[1] - t[0])) for t in xrads]
    yrads = [find_radius(ydata, i, i + 1, ydata[i][0], m) for i in sample]
    yrads = [(t[2] / (t[1] - t[0])) for t in yrads]
    ratio = float(sum(xrads)) / sum(yrads)
    # now we measure local radii / ratio for a sample of points
    # using find_rect()
    sample = [random.randrange(n) for i in range(nsample)]
    ratios = []
    for i in sample:
        m2,xradius,yradius,points = find_rect(ydata, i, m, ratio, True)
        # now measure average x/y radii at this point
        xrads = [abs(ydata[i][1] - ydata[j][1]) for j in points]
        yrads = [abs(ydata[i][0] - ydata[j][0]) for j in points]
        ratios.append(float(sum(xrads)) / sum(yrads))
    return sum(ratios) / nsample


def cond_entropy(vectors, m, nsample=50):
    '''calculate conditional entropy using rectangle method,
    m: number of points to use as sample.

    Is this calculation biased?  It seems to be systematically
    overestimating the density (i.e. underestimating the entropy).'''
    source = [t for t in vectors] # convert to list
    source.sort() # sort tuples... numpy.sort() NOT usable for this
    xdata = numpy.array([t[0] for t in source]) # 1D array for searchsorted
    target = [(t[1],t[0],i) for (i,t) in enumerate(source)]
    target.sort()
    ratio = get_ratio(source, target, m, nsample)
    n = len(target)
    dvec = numpy.core.zeros((n))
    for i in range(n):
        ## ratio = find_ratio(source, target, i, m)
        m2,xradius,yradius,points = find_rect(target, i, m, ratio)
        l = xdata.searchsorted(target[i][1] - xradius) # total w/in xradius
        r = xdata.searchsorted(target[i][1] + xradius, side='right')
        dvec[i] = m2 / ((r - l - 1) * 2. * yradius)
    return LogPVector( - numpy.log(dvec))

    
