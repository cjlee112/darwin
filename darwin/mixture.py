import numpy

class Mixture(object):
    def __init__(self, models):
        self.models = models

    def pdf(self, data):
        p = numpy.core.zeros((len(data)))
        for prior,model in self.models:
            p += prior * model.pdf(data)
        return p

    def rvs(self, n):
        priors = [t[0] for t in self.models]
        counts = numpy.random.multinomial(n, priors, size=1)
        sample = None
        for i,c in enumerate(counts[0]):
            if c > 0:
                vals = self.models[i][1].rvs(c)
                if sample is None:
                    sample = vals
                else:
                    sample = numpy.concatenate((sample,vals))
        return sample
