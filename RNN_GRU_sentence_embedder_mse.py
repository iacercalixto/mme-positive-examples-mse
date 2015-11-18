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

class RNN_GRU(NeuralNetworkWeights):
    def __init__(self):
        logger.info("Using RNN_GRU...")
    
    def init_parameters(self, word_embeddings, n_in, n_hidden, vocabulary_size):
        # initialise word embeddings
        if word_embeddings is None:
            word_embeddings = self.norm_weight(vocabulary_size, n_in)
        self.word_embeddings = theano.shared(value  = word_embeddings,
                                             name   = 'word_embeddings',
                                             borrow = True)
        
        # initialise GRU parameters
        # gates input to hidden layer weights
        W_init = np.concatenate([self.norm_weight(n_in, n_hidden),
                                 self.norm_weight(n_in, n_hidden)], axis=1)
        self.W = theano.shared(value=W_init, name='W', borrow=True)
        
        b_init = np.zeros((2 * n_hidden,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_init, name='b', borrow=True)
        
        # gates recurrent weights U as a shared variable
        U_init = np.concatenate([self.ortho_weight(n_hidden),
                                 self.ortho_weight(n_hidden)], axis=1)
        self.U = theano.shared(value=U_init, name='U', borrow=True)
        
        # input to hidden layer weights
        Wx_init = self.norm_weight(n_in, n_hidden)
        self.Wx = theano.shared(value=Wx_init, name='Wx', borrow=True)
        
        # recurrent weights U as a shared variable
        Ux_init = self.ortho_weight(n_hidden)
        self.Ux = theano.shared(value=Ux_init, name='Ux', borrow=True)
        
        bx_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.bx = theano.shared(value=bx_init, name='bx', borrow=True)
        
        self.params = [self.W, self.b, self.U, self.Wx, self.bx, self.Ux]
    
    def load_parameters(self, params, word_embeddings):
        self.word_embeddings = theano.shared(value  = word_embeddings,
                                             name   = 'word_embeddings',
                                             borrow = True)
        
        # load (aka. deep copy) parameters in params into network
        c=0
        self.params = []
        names  = ['W', 'b', 'U', 'Wx', 'bx', 'Ux']
        for n,p in zip(names, params):
            self.params.append(theano.shared(name   = p.name,
                                             value  = p.get_value(borrow=True)))
            
            setattr(self, n, self.params[c])
            c+=1
            assert( len(self.params) == c )
    
    def create(self,
               minibatch_sentences, # (n_timesteps x n_examples x word embeddings dimension)
               minibatch_mask=None  # masks for minibatch_sentences
        ):
        # minibatch_sentences is 3D tensor
        # (n_words_in_input_sentences x n_sentences_in_minibatch x word_embeddings_dimensionality)
        n_timesteps = minibatch_sentences.shape[0]
        n_examples  = minibatch_sentences.shape[1]
        #if minibatch_sentences.ndim == 3:
        #    n_samples = minibatch_sentences.shape[1]
        dim = self.Ux.shape[1]
        n_in = self.word_embeddings.shape[1]
        
        if minibatch_mask == None:
            minibatch_mask = T.alloc(1., minibatch_sentences.shape[0], 1)
        
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
        
        input = self.word_embeddings[minibatch_sentences]
        input.reshape([n_timesteps, n_examples, n_in])
        mask = minibatch_mask.reshape([n_timesteps, n_examples, 1])
        
        input_ = T.dot(input, self.W) + self.b
        inputx = T.dot(input, self.Wx) + self.bx
        
        # inputs, outputs, non_sequences
        # inputs:        m_, x_, xx_ (mask, input_, inputx)
        # outputs:       h (h0)
        # non_sequences: U, Ux, dim (self.U, self.Ux, dim)
        def _step_slice(m_, x_, xx_, h_, U, Ux, dim):
            preact = T.dot(h_, U)
            preact += x_
    
            r = T.nnet.sigmoid(_slice(preact, 0, dim))
            u = T.nnet.sigmoid(_slice(preact, 1, dim))
    
            preactx = T.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_
    
            h = T.tanh(preactx)
    
            h = u * h_ + (1. - u) * h
            h = m_[:,None] * h + (1. - m_)[:,None] * h_
    
            return h
        
        h0    = T.unbroadcast(T.alloc(0., n_examples, dim), 0)
        seqs  = [mask, input_, inputx]
        _step = _step_slice
        
        rval, updates = theano.scan(_step,
                                    sequences     = seqs,
                                    outputs_info  = [h0],
                                    non_sequences = [self.U, self.Ux, dim],
                                    name          = 'GRU_layer',
                                    n_steps       = n_timesteps,
                                    strict        = True)
        
        h = rval
        self.last_h = h[-1]
        
        # create predict function
        self._predict = theano.function([minibatch_sentences, minibatch_mask],
                                        self.last_h)
        
        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction
        self.updates = OrderedDict()
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)
        
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        self.L1 += abs(self.W.sum())
        self.L1 += abs(self.U.sum())
        self.L1 += abs(self.Wx.sum())
        self.L1 += abs(self.Ux.sum())
        self.L1.name = 'L1_regulariser'

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        self.L2_sqr += (self.W ** 2).sum()
        self.L2_sqr += (self.U ** 2).sum()
        self.L2_sqr += (self.Wx ** 2).sum()
        self.L2_sqr += (self.Ux ** 2).sum()
        self.L2_sqr.name = 'L2_regulariser'
        
        self.loss = lambda h: self.mse_h(h)
        #self.loss = lambda y: self.mse(y)

    def mse_h(self, h):
        # error between output and hidden memory final state
        return T.mean((self.last_h - h) ** 2)
    
    def predict(self, X):
        return self._predict(X, np.ones_like(X, dtype=theano.config.floatX))