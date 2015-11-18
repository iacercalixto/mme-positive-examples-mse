from abc import ABCMeta
import numpy as np

class NeuralNetworkWeights(object):
    __metaclass__ = ABCMeta
    
    def __init__(self):
        object.__init__(self)
    
    # orthogonal initialization for weights
    # see Saxe et al. ICLR'14
    def ortho_weight(self, ndim, sample_from_distribution=np.random.randn):
        if sample_from_distribution == np.random.uniform:
            W = sample_from_distribution(size=(ndim, ndim))
        else:
            W = sample_from_distribution(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        return u.astype('float32')
    
    # weight initializer, normal by default
    def norm_weight(self, nin, nout=None, scale=0.01, ortho=True):
        if nout is None:
            nout = nin
        if nout == nin and ortho:
            W = self.ortho_weight(nin)
        else:
            W = scale * np.random.randn(nin, nout)
        return W.astype('float32')
    
    # weight initializer, normal by default
    def uniform_weight(self, nin, nout=None, scale=0.01, ortho=True):
        if nout is None:
            nout = nin
        if nout == nin and ortho:
            W = self.ortho_weight(nin,
                                  sample_from_distribution=np.random.uniform)
        else:
            W = scale * np.random.uniform(size=(nin, nout))
        return W.astype('float32')