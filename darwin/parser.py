import re

class REMatcher(object):
    'regexp matcher for use as match_f for SeqMatchState'
    def __init__(self, regexp):
        self.matcher = re.compile(regexp)

    def __call__(self, s):
        m = self.matcher.match(s)
        if m:
            print 'REMatcher: ', m.group()
            return m.group() # extract the matched string
        else:
            return None

class SimpleEmitter(object):
    def __init__(self, base):
        self.base = base

    def pmf(self, obs):
        print 'SimpleEmitter: ', obs
        return [pow(self.base, len(o)) for o in obs]
