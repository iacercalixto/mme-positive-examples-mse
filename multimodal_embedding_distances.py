from abc import ABCMeta
import numpy as np
import sys, os
import cPickle as pickle
import traceback
import numpy as np
import theano
import theano.tensor as T

import os, sys
import logging
reload(logging)
logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MultimodalEmbeddingDistances(object):
    """ Classes that inherit from this class can compute distances between
        sentence/image vectors.
        
        When computing distances (calling the get_distances_* functions),
        it is assumed there exist both numpy arrays
        self.sentence_embeddings and self.image_embeddings
    """
    __metaclass__ = ABCMeta
    
    def __init__(self):
        object.__init__(self)
    
    def get_distances_from_image_embedding(self, x_sentence_embedding):
        assert(len(self.image_embeddings) != 0)
        
        if x_sentence_embedding.ndim==2:
            return self.get_distances_from_image_embedding_batch(x_sentence_embedding)
        elif x_sentence_embedding.ndim==1:
            return self.get_distances_from_image_embedding_example(x_sentence_embedding)
        else:
            raise NotImplementedError("x_sentence_embedding must have one or two dimensions!")
    
    def get_distances_from_image_embedding_batch(self, x_sentences_embeddings):
        """ Given some arbitrary sentences embeddings (matrix),
            return a ranking of the closest images to each
            according to the current state of the model.
        """
        assert(len(self.image_embeddings) != 0)
        raise Exception("Use get_distances_from_image_embedding_example()")
        
        if not x_sentences_embeddings.ndim==2:
            raise Exception("x_sentences_embeddings must have two dimensions!")
        
        X = T.matrix('x_sentences', dtype='float32')
        Y = self.image_embeddings
        
        squared_euclidean_distances = (X ** 2).sum(1).reshape((X.shape[0], 1)) + \
                                      (Y ** 2).sum(1).reshape((1, Y.shape[0])) - \
                                      2 * X.dot(Y.T)
        squared_euclidean_distances = T.sqrt(T.maximum(squared_euclidean_distances,0.))
        f_euclidean = theano.function([X], squared_euclidean_distances)
        
        distances = f_euclidean(x_sentences_embeddings)
        return distances, distances.argsort()
    
    def get_distances_from_image_embedding_example(self, x_sentence_embedding):
        """ Given an arbitrary sentence embedding (vector),
            return a ranking of the closest images to it
            according to the current state of the model.
        """
        assert(len(self.image_embeddings) != 0)
        
        if not x_sentence_embedding.ndim==1:
            raise Exception("x_sentence_embedding must have one dimension!")
        
        X = T.vector('x_sentence', dtype='float32')
        Y = self.image_embeddings
        
        vector_matrix_row_wise_dot_product = X.reshape((X.shape[0], 1)).T.dot(Y.T)
        f_euclidean = theano.function([X], vector_matrix_row_wise_dot_product)
        
        distances = f_euclidean(x_sentence_embedding)[0]
        return distances, distances.argsort()[::-1]
    
    def get_distances_from_sentence_embedding(self, x_image_embedding):
        assert(len(self.sentence_embeddings) != 0)
        if x_image_embedding.ndim==2:
            return self.get_distances_from_sentence_embedding_batch(x_image_embedding)
        elif x_image_embedding.ndim==1:
            return self.get_distances_from_sentence_embedding_example(x_image_embedding)
        else:
            raise NotImplementedError("x_image_embedding must have one or two dimensions!")
    
    def get_distances_from_sentence_embedding_example(self, x_image_embedding):
        """ Given an arbitrary image embedding (vector),
            return a ranking of the sentences closest to it
            according to the current state of the model.
        """
        assert(len(self.sentence_embeddings) != 0)
        if not x_image_embedding.ndim==1:
            raise Exception("x_image_embedding must have one dimension!")
        
        X = T.vector('x_image', dtype='float32')
        Y = self.sentence_embeddings
        
        vector_matrix_row_wise_dot_product = X.reshape((X.shape[0], 1)).T.dot(Y.T)
        f_euclidean = theano.function([X], vector_matrix_row_wise_dot_product)
        
        distances = f_euclidean(x_image_embedding)[0]
        return distances, distances.argsort()[::-1]
    
    def get_distances_from_sentence_embedding_batch(self, x_images_embeddings):
        """ Given some arbitrary image embeddings (matrix),
            return a ranking of the sentences closest to each
            according to the current state of the model.
        """
        assert(len(self.sentence_embeddings) != 0)
        raise Exception("Use get_distances_from_sentence_embedding_batch()")
        
        if not x_images_embeddings.ndim==2:
            raise Exception("x_images_embeddings must have two dimensions!")
        
        X = T.matrix('x_images', dtype='float32')
        Y = self.sentence_embeddings
        
        squared_euclidean_distances = (X ** 2).sum(1).reshape((X.shape[0], 1)) + \
                                      (Y ** 2).sum(1).reshape((1, Y.shape[0])) - \
                                      2 * X.dot(Y.T)
        squared_euclidean_distances = T.sqrt(T.maximum(squared_euclidean_distances,0.))
        f_euclidean = theano.function([X], squared_euclidean_distances)
        
        distances = f_euclidean(x_images_embeddings)
        return distances, distances.argsort()