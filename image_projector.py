import numpy as np
import theano
import theano.tensor as T
import logging

from collections import OrderedDict

from neural_network_weights import NeuralNetworkWeights

class ImageProjector(NeuralNetworkWeights):
    
    def __init__(self):
        pass
    
    def init_parameters(self, n_in, n_out):
        """ Initialise parameters with default values/distributions
            using the given parameters for network parameters' shapes.
        """
        #self.input = input
        
        # parameters
        W_i_init = self.uniform_weight(n_in, n_out)
        self.W_i = theano.shared(value=W_i_init, name='W_i', borrow=True)
        
        self.params = [ self.W_i ]
    
    def load_parameters(self, params):
        """ Load network parameters directly from params
        """
        # load (aka. deep copy) parameters in params into network
        c=0
        self.params = []
        names  = ['W_i']
        for n,p in zip(names, params):
            self.params.append(theano.shared(name   = p.name,
                                             value  = p.get_value(borrow=True)))
            
            setattr(self, n, self.params[c])
            c+=1
            assert( len(self.params) == c )
    
    def create(self, input):
        """ Create network for given input
        """
        assert(not self.params is None and not len(self.params) == 0)
        
        # updates
        self.updates = OrderedDict()
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)
        
        self.y_pred = T.dot(input, self.W_i)
        
        # L1 and L2 norms
        self.L1 = 0
        self.L1 += abs(self.W_i.sum())
        self.L1.name = 'L1_regulariser'
        self.L2_sqr = 0
        self.L2_sqr += (self.W_i ** 2).sum()
        self.L2_sqr.name = 'L2_regulariser'
        
        # loss and predict functions
        self.loss = lambda y: self.mse(y)
        
        self._predict = theano.function([input], self.y_pred)
        
    def mse(self, y):
        """ error between output and target """
        return T.mean((self.y_pred - y) ** 2)
    
    def predict(self, X):
        return self._predict(X)