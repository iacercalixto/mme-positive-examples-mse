import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator
import logging
import time
import sys, os
import datetime
import cPickle as pickle
from collections import OrderedDict
from itertools import izip

import os, sys
import logging
reload(logging)
logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from neural_network_weights import NeuralNetworkWeights

class RNN(NeuralNetworkWeights):
    
    def __init__(self):
        logger.info("Using RNN...")
    
    def load_parameters(self, params, word_embeddings):
        """ Directly load given parameters into the network.
        """
        self.word_embeddings = theano.shared(value  = word_embeddings,
                                             name   = 'word_embeddings',
                                             borrow = True)
        
        # load (aka. deep copy) parameters in params into network
        c=0
        self.params = []
        names  = ['W', 'W_in', 'bh']
        for n,p in zip(names, params):
            self.params.append(theano.shared(name   = p.name,
                                             value  = p.get_value(borrow=True)))
            
            setattr(self, n, self.params[c])
            #logger.info("self.%s = %s (type %s)" % (n, str(self.params[c]), str(type(self.params[c]))))
            c+=1
            assert( len(self.params) == c )
    
    def init_parameters(self,
                        n_in,        # word embeddings dimension
                        n_hidden,    # multimodal embeddings dimension
                        vocabulary_size,
                        word_embeddings = None):
        """ Initialise network parameters with default values/distributions
            (using sizes provided as parameters' shapes).
        """
        # word embeddings
        if word_embeddings is None:
            word_embeddings = self.norm_weight(vocabulary_size, n_in)
        self.word_embeddings = theano.shared(value  = word_embeddings,
                                             name   = 'word_embeddings',
                                             borrow = True)
        
        # recurrent weights as a shared variable
        W_init = self.norm_weight(n_hidden)
        self.W = theano.shared(value=W_init, name='W', borrow=True)
        
        # input to hidden layer weights
        W_in_init = self.norm_weight(n_in, n_hidden)
        self.W_in = theano.shared(value=W_in_init, name='W_in', borrow=True)

        ## hidden to output layer weights
        #W_out_init = self.norm_weight(n_hidden, n_out)
        #self.W_out = theano.shared(value=W_out_init, name='W_out', borrow=True)
        
        bh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.bh = theano.shared(value=bh_init, name='bh', borrow=True)

        #by_init = np.zeros((n_out,), dtype=theano.config.floatX)
        #self.by = theano.shared(value=by_init, name='by', borrow=True)
        
        self.params = [self.W, self.W_in, self.bh]
        #self.params = [self.W, self.W_in, self.W_out, self.bh, self.by]
    
    def create(self,
               minibatch_sentences,   # (n_timesteps x n_examples x word embeddings dimension)
               minibatch_mask = None, # masks for minibatch_sentences
               activation=T.nnet.sigmoid):
        
        assert(not self.params is None and not len(self.params) == 0)
        
        # minibatch_sentences is 3D tensor
        # (n_words_in_input_sentences x n_sentences_in_minibatch x word_embeddings_dimensionality)
        n_timesteps = minibatch_sentences.shape[0]
        n_examples  = minibatch_sentences.shape[1]
        
        n_in = self.word_embeddings.shape[1]
        n_hidden = self.W.shape[0]
        
        #self.input = self.word_embeddings[minibatch_sentences.flatten()]
        input = self.word_embeddings[minibatch_sentences]
        input.reshape([n_timesteps, n_examples, n_in])
        
        if minibatch_mask == None:
            minibatch_mask = T.alloc(1., minibatch_sentences.shape[0], 1)
            #minibatch_mask = np.ones((n_timesteps, n_examples, 1))
        mask = minibatch_mask.reshape([n_timesteps, n_examples, 1])
        
        self.activation = activation
        
        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction
        self.updates = OrderedDict()
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)
        
        # recurrent function (using sigmoid activation function)
        # and linear output activation function (currently unused)
        def step(x_t, mask, h_tm1):
            h_t = self.activation( T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W) + self.bh )
            #y_t = T.dot(h_t, self.W_out) + self.by
            #return [h_t, y_t]
            return h_t
        
        h0 = T.unbroadcast(T.alloc(0., n_examples, n_hidden), 0)
        
        # mapping from word embeddings layer into first hidden layer
        #projected_input = T.dot(self.input, self.W_first) + self.b_first
        #projected_input = self.input
        
        # the hidden state `h` for the entire sequences, and the output for the
        # entire sequences `y_pred` (first dimension is always time)
        #[h, y_pred], _ = theano.scan(step,
        h, updates = theano.scan(step,
                                 sequences=[input, mask],
                                 outputs_info=[h0],
                                 n_steps=input.shape[0])
        
        self.last_h = h[-1]
        self.last_h.name = 'last_h'
        
        # create a predict function
        self._predict = theano.function([minibatch_sentences, minibatch_mask],
                                        self.last_h)
        
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        self.L1 += abs(self.W.sum())
        self.L1 += abs(self.W_in.sum())
        #self.L1 += abs(self.W_out.sum())
        self.L1.name = 'L1_regulariser'

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        self.L2_sqr += (self.W ** 2).sum()
        self.L2_sqr += (self.W_in ** 2).sum()
        #self.L2_sqr += (self.W_out ** 2).sum()
        self.L2_sqr.name = 'L2_regulariser'
        
        self.loss = lambda h: self.mse_h(h)
        #self.loss = lambda y: self.mse(y)
    
    def mse_h(self, h):
        # error between output and hidden memory final state
        return T.mean((self.last_h - h) ** 2)
    
    def predict(self, X):
        return self._predict(X, np.ones_like(X, dtype=theano.config.floatX))