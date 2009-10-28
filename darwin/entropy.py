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
    pValue=0.4 implies 20% below lower bound, 20% above upper bound.
    If you compare lower bound of -Le vs upper bound of He with these
    p-values, implies ~4% probability that true -Le-He < estimated -Le-He
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
