import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator
import logging
import time
import sys, os
import datetime
from collections import OrderedDict, deque
from itertools import izip, chain

import os, sys
import logging
reload(logging)
logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from persistent_object import PersistentObject
from RNN_sentence_embedder_mse import RNN
from RNN_GRU_sentence_embedder_mse import RNN_GRU
from RNN_GRU_multilayer_sentence_embedder_mse import RNN_Multilayer
from image_projector import ImageProjector
from plot_losses import plot_losses_vs_time
from multimodal_embedding_distances import MultimodalEmbeddingDistances

class MultimodalEmbeddingsLearner(PersistentObject,BaseEstimator,MultimodalEmbeddingDistances):
    """ Create multimodal embeddings by fitting positive
        sentence/image pairs from Flickr8k
        into one same (multimodal) vector space.
    """
    
    def setState(self, state):
        """ Set parameters from state sequence.
            Parameters must be in predefined order.
        """
        meta_params, sentence_proj_params, sentence_proj_update_mom, \
                image_proj_params, image_proj_update_mom = state
        
        # call to BaseEstimator class method
        self.set_params(**meta_params)
        self.ready()
        self.__setSentenceParams(sentence_proj_params, sentence_proj_update_mom)
        self.__setImageParams(image_proj_params, image_proj_update_mom)
    
    def getState(self):
        """ Return state sequence (meta parameters plus networks parameters)."""
        
        # call to BaseEstimator class method
        meta_params              = self.get_params()
        sentence_proj_params     = [p for p in self.sentence_proj.params]
        sentence_proj_update_mom = [p for p in self.sentence_proj.updates]
        image_proj_params        = [p for p in self.image_proj.params]
        image_proj_update_mom    = [p for p in self.image_proj.updates]
        
        state = (meta_params,
                 sentence_proj_params, sentence_proj_update_mom,
                 image_proj_params, image_proj_update_mom)
        
        return state
    
    def __init__(self,
                 train_stream = None,
                 valid_stream = None,
                 test_stream  = None,
                 word_embeddings=None,
                 vocabulary_size=100000,
                 word_emb_dim=300,
                 image_fv_dim=4096,
                 multimodal_emb_dim=500,
                 n_layers=1,
                 learning_rate=0.01,
                 n_epochs=100,
                 L1_reg=0.00, L2_reg=0.00,
                 learning_rate_decay=1,
                 activation=T.nnet.sigmoid,
                 final_momentum=0.9,
                 initial_momentum=0.5,
                 momentum_switchover=5,
                 patience=5000,
                 patience_increase=2,
                 improvement_threshold=0.995,
                 finish_after = 100000000,
                 proj_folder='.',
                 proj_name=None,
                 load_model=False,
                 stop_on_load_error=False,
                 last_epoch=-1,
                 time_measures=[],
                 clock_measures=[],
                 all_train_losses=[],
                 all_valid_losses=[],
                 all_test_losses=[]):
        # mandatory parameters
        assert(not train_stream == None and not valid_stream == None)
        assert(n_layers >= 1)
        
        # these parameters are in this constructor
        # for persistence purposes only but ** must never be set **
        # to a value different than their default values
        assert(last_epoch == -1)
        assert(time_measures == clock_measures == all_train_losses
               == all_valid_losses == all_test_losses == [])
        
        # the three parameters below have different names from their
        # corresponding to their setters. this means they will not be pickled!
        self.__train_stream          = train_stream
        self.__valid_stream          = valid_stream
        self.__test_stream           = test_stream
        
        self.word_embeddings       = word_embeddings

        self.vocabulary_size       = int(vocabulary_size)
        self.word_emb_dim          = int(word_emb_dim)
        self.image_fv_dim          = int(image_fv_dim)
        self.multimodal_emb_dim    = int(multimodal_emb_dim)
        self.n_layers              = int(n_layers)
        
        self.learning_rate         = float(learning_rate)
        self.learning_rate_decay   = float(learning_rate_decay)
        self.n_epochs              = int(n_epochs)
        self.L1_reg                = float(L1_reg)
        self.L2_reg                = float(L2_reg)
        self.activation            = activation
        self.initial_momentum      = float(initial_momentum)
        self.final_momentum        = float(final_momentum)
        self.momentum_switchover   = int(momentum_switchover)
        self.patience              = int(patience)
        self.patience_increase     = float(patience_increase)
        self.improvement_threshold = float(improvement_threshold)
        self.finish_after          = int(finish_after)
        
        # last epoch trained, starts from -1
        self.last_epoch            = last_epoch
        # whether we attempt to load pretrained model from disk
        self.load_model            = bool(load_model)
        
        # vars used for plotting training/dev/test progress
        self.time_measures         = time_measures
        self.clock_measures        = clock_measures
        self.all_train_losses      = all_train_losses
        self.all_valid_losses      = all_valid_losses
        self.all_test_losses       = all_test_losses
        
        self.proj_folder           = proj_folder
        self.proj_name             = proj_name
        self.stop_on_load_error    = stop_on_load_error
        
        # super class, implements behaviour to save() and load() from disk
        PersistentObject.__init__(self,
                                  proj_folder=proj_folder,
                                  proj_name=proj_name,
                                  stop_on_load_error=stop_on_load_error)
        
        self.populate_dataset_metadata()
        self.ready()
        
        if self.load_model:
            self.load()
    
        
    def ready(self):
        # vector of sentences, where each sentence input is a vector of word ids
        self.x_sentence         = T.matrix(name='x_sentence',
                                            dtype='int64')
        # masks for padding
        self.x_sentence_mask    = T.matrix(name='x_sentence_mask',
                                            dtype=theano.config.floatX)
        # indices of sentences in the minibatch
        self.x_sentence_indices = T.vector(name='x_sentence_indices',
                                            dtype='int64')
        
        # image input feature vector
        self.x_image            = T.matrix(name='x_image',
                                            dtype=theano.config.floatX)
        # indices of images in the minibatch
        self.x_image_indices    = T.vector(name='x_image_indices',
                                            dtype='int64')
        
        # sentence embeddings RNN
        #self.sentence_proj = RNN()
        #self.sentence_proj = RNN_GRU()
        self.sentence_proj = RNN_Multilayer()
        
        self.sentence_proj.init_parameters(
            n_in                       = self.word_emb_dim,
            n_hidden                   = self.multimodal_emb_dim,
            word_embeddings            = self.word_embeddings,
            vocabulary_size            = self.vocabulary_size,
            n_layers                   = self.n_layers)
        
        self.sentence_proj.create(
            minibatch_sentences        = self.x_sentence,
            minibatch_mask             = self.x_sentence_mask)
        
        # image embeddings linear projection network
        self.image_proj = ImageProjector()
        
        self.image_proj.init_parameters(
            n_in             = self.image_fv_dim,
            n_out            = self.multimodal_emb_dim)
        
        self.image_proj.create(
            input            = self.x_image)
    
    
    def shared_dataset(self, data_xy):
        """ Load the dataset into shared variables / list of shared variables
            X values are word indices, not vectors (at this point).
            Y values are image festure vectors.
        """

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype='int32'))
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX))
        return shared_x, shared_y

    def __setSentenceParams(self, sentence_proj_params, sentence_proj_update_mom):
        """ Set sentence RNN fittable parameters from weights sequence.
            Parameters must be in the order defined by rnn_params and rnn_update_mom.
        """
        assert(len(sentence_proj_params) == len(sentence_proj_update_mom))
        assert(len(self.sentence_proj.params) == len(self.sentence_proj.updates))
        assert(len(sentence_proj_params) == len(self.sentence_proj.params))
        
        c=0
        for param, update in zip(self.sentence_proj.params,
                                 self.sentence_proj.updates):
            param.set_value( sentence_proj_params[c].get_value() )
            update.set_value( sentence_proj_update_mom[c].get_value() )
            c+=1
    
    def __setImageParams(self, image_proj_params, image_proj_update_mom):
        """ Set image network parameters.
            Parameters must be in the right order.
        """
        assert(len(image_proj_params) == len(image_proj_update_mom))
        assert(len(self.image_proj.params) == len(self.image_proj.updates))
        assert(len(image_proj_params) == len(self.image_proj.params))
        
        c=0
        for param, update in zip(self.image_proj.params, self.image_proj.updates):
            param.set_value( image_proj_params[c].get_value() )
            update.set_value( image_proj_update_mom[c].get_value() )
            c+=1
    
    def populate_dataset_metadata(self):
        # populate training set metadata
        # populate mappings from image idx -> sentences ids and vice-versa
        self.training_set_image_idx_to_sentence_indices = OrderedDict() # 1xN mapping
        self.training_set_sentence_idx_to_image_idx     = OrderedDict() # 1x1 mapping
        for minibatch in self.__train_stream.get_epoch_iterator():
            sentence_indices, image_indices, _, _, _ = minibatch
            
            sentence_indices = sentence_indices.flatten()
            image_indices    = image_indices.flatten()
            assert(len(sentence_indices) == len(image_indices))
            
            for image_idx, sentence_idx in zip(image_indices, sentence_indices):
                if not image_idx in self.training_set_image_idx_to_sentence_indices:
                    self.training_set_image_idx_to_sentence_indices[image_idx] = deque()
                
                self.training_set_image_idx_to_sentence_indices[image_idx].append(sentence_idx)
                self.training_set_sentence_idx_to_image_idx[sentence_idx]  = image_idx
        
        self.training_set_n_unique_images = len(self.training_set_image_idx_to_sentence_indices)
        self.training_set_n_unique_sentences = len(self.training_set_sentence_idx_to_image_idx)
        
        #logger.info("training_set_n_unique_images: %s" % str(self.training_set_n_unique_images))
        #logger.info("training_set_n_unique_sentences: %s" % str(self.training_set_n_unique_sentences))
        
        # populate validation set metadata
        # populate mappings from image idx -> sentences ids and vice-versa
        self.valid_set_image_idx_to_sentence_indices = OrderedDict() # 1xN mapping
        self.valid_set_sentence_idx_to_image_idx     = OrderedDict() # 1x1 mapping
        for minibatch in self.__valid_stream.get_epoch_iterator():
            sentence_indices, image_indices, _, _, _ = minibatch
            
            sentence_indices = sentence_indices.flatten()
            image_indices    = image_indices.flatten()
            assert(len(sentence_indices) == len(image_indices))
            
            for image_idx, sentence_idx in zip(image_indices, sentence_indices):
                if not image_idx in self.valid_set_image_idx_to_sentence_indices:
                    self.valid_set_image_idx_to_sentence_indices[image_idx] = deque()
                
                self.valid_set_image_idx_to_sentence_indices[image_idx].append(sentence_idx)
                self.valid_set_sentence_idx_to_image_idx[sentence_idx]  = image_idx
        
        self.valid_set_n_unique_images = len(self.valid_set_image_idx_to_sentence_indices)
        self.valid_set_n_unique_sentences = len(self.valid_set_sentence_idx_to_image_idx)
        
        # populate test set metadata
        # populate mappings from image idx -> sentences ids and vice-versa
        self.test_set_image_idx_to_sentence_indices = OrderedDict() # 1xN mapping
        self.test_set_sentence_idx_to_image_idx     = OrderedDict() # 1x1 mapping
        for minibatch in self.__test_stream.get_epoch_iterator():
            sentence_indices, image_indices, _, _, _ = minibatch
            
            sentence_indices = sentence_indices.flatten()
            image_indices    = image_indices.flatten()
            assert(len(sentence_indices) == len(image_indices))
            
            for image_idx, sentence_idx in zip(image_indices, sentence_indices):
                if not image_idx in self.test_set_image_idx_to_sentence_indices:
                    self.test_set_image_idx_to_sentence_indices[image_idx] = deque()
                
                self.test_set_image_idx_to_sentence_indices[image_idx].append(sentence_idx)
                self.test_set_sentence_idx_to_image_idx[sentence_idx]  = image_idx
        
        self.test_set_n_unique_images = len(self.test_set_image_idx_to_sentence_indices)
        self.test_set_n_unique_sentences = len(self.test_set_sentence_idx_to_image_idx)
        
        #logger.info("test_set_n_unique_images: %s" % str(self.test_set_n_unique_images))
        #logger.info("test_set_n_unique_sentences: %s" % str(self.test_set_n_unique_sentences))
        
    
    def test(self):
        self.populate_dataset_metadata()
        
        logger.info("Generating image and sentence embeddings using trained network...")
        
        n_images_total = int(self.training_set_n_unique_images +
                             self.valid_set_n_unique_images +
                             self.test_set_n_unique_images)
        n_sentences_total = int(self.training_set_n_unique_sentences +
                                self.valid_set_n_unique_sentences +
                                self.test_set_n_unique_sentences)
        
        self.image_embeddings = np.zeros((n_images_total, self.multimodal_emb_dim),
                                         dtype=theano.config.floatX)
        self.sentence_embeddings = np.zeros((n_sentences_total, self.multimodal_emb_dim),
                                            dtype=theano.config.floatX)
        
        #logger.info("sentence_embeddings: %s" % str(np.shape(sentence_embeddings)))
        #logger.info("image_embeddings: %s" % str(np.shape(image_embeddings)))
        
        # dictionary with mappings between sentence id <-> sentence embedding id
        self.sentence_idx_to_sentence_embedding_idx = OrderedDict()
        self.sentence_embedding_idx_to_sentence_idx = OrderedDict()
        # dictionary with mappings between image id    <-> image embedding id
        self.image_idx_to_image_embedding_idx = OrderedDict()
        self.image_embedding_idx_to_image_idx = OrderedDict()
        
        # generate sentence and image embeddings for training+test instances
        # using current network parameters
        start_sentence_idx, start_image_idx = 0, 0
        for minibatch in chain(self.__train_stream.get_epoch_iterator(),
                               self.__valid_stream.get_epoch_iterator(),
                               self.__test_stream.get_epoch_iterator()):
            x_sent_idx, x_image_idx, x_sent, x_mask, x_image = minibatch
            
            x_sent_idx = x_sent_idx.flatten()
            x_image_idx = x_image_idx.flatten()
            
            # sentences ids <-> sentence embeddings ids mapping
            def map_sentences(idx):
                if idx in self.sentence_idx_to_sentence_embedding_idx:
                    return self.sentence_idx_to_sentence_embedding_idx[idx]
                else:
                    retval = len(self.sentence_idx_to_sentence_embedding_idx)
                    self.sentence_idx_to_sentence_embedding_idx[idx] = retval
                    self.sentence_embedding_idx_to_sentence_idx[retval] = idx
                    return retval
            
            minibatch_sentence_indices = x_sent_idx
            sentence_embedding_indices = map(map_sentences, minibatch_sentence_indices)
            
            # images ids <-> image embeddings ids mapping
            def map_images(idx):
                if idx in self.image_idx_to_image_embedding_idx:
                    return self.image_idx_to_image_embedding_idx[idx]
                else:
                    retval = len(self.image_idx_to_image_embedding_idx)
                    self.image_idx_to_image_embedding_idx[idx] = retval
                    self.image_embedding_idx_to_image_idx[retval] = idx
                    return retval
            
            minibatch_image_indices = x_image_idx
            image_embedding_indices = map(map_images, minibatch_image_indices)
            
            #logger.info("")
            #logger.info("sentence_embedding_indices: %s" % str(sentence_embedding_indices))
            #logger.info("minibatch_sentence_indices: %s" % str(minibatch_sentence_indices))
            #logger.info("image_embedding_indices: %s" % str(image_embedding_indices))
            #logger.info("minibatch_image_indices: %s" % str(minibatch_image_indices))
            
            # first, generate sentence embeddings for test set sentences
            # time as first dimension, minibatch as second dimension
            x_sent = np.swapaxes(x_sent, 0, 1)
            x_mask = np.swapaxes(x_mask, 0, 1)
            
            self.sentence_embeddings[sentence_embedding_indices] = self.rnn.predict(x_sent)
            self.image_embeddings[image_embedding_indices]       = self.image_proj.predict(x_image)
        
        logger.info("len(self.sentence_embedding_idx_to_sentence_idx): %i" %
                    len(self.sentence_embedding_idx_to_sentence_idx))
        logger.info("len(self.image_embedding_idx_to_image_idx): %i" %
                    len(self.image_embedding_idx_to_image_idx))
        
        assert(len(self.sentence_idx_to_sentence_embedding_idx) == n_sentences_total and
               len(self.image_idx_to_image_embedding_idx) == n_images_total)
        
        logger.info("Done!")
        
        sentences_given_image_medium_rank  = []
        sentences_given_image_recall_at_1  = []
        sentences_given_image_recall_at_5  = []
        sentences_given_image_recall_at_10 = []
        
        images_given_sentence_medium_rank  = []
        images_given_sentence_recall_at_1  = []
        images_given_sentence_recall_at_5  = []
        images_given_sentence_recall_at_10 = []
        
        # having the test sentences and images in the multimodal embedding, evaluate them
        for test_minibatch in self.__test_stream.get_epoch_iterator():
            x_sentt_idx, x_imaget_idx, x_sentt, x_maskt, x_imaget = test_minibatch
            
            x_sentt_idx = x_sentt_idx.flatten()
            x_imaget_idx = x_imaget_idx.flatten()
            
            # time as first dimension, minibatch as second dimension
            x_sentt = np.swapaxes(x_sentt, 0, 1)
            x_maskt = np.swapaxes(x_maskt, 0, 1)
            
            # process each image in test set one by one
            for sent_idx, image_idx in zip(x_sentt_idx, x_imaget_idx):
                # obtain the other four sentences indices that illustrate
                # the same image as the current sentence
                current_image_idx = self.test_set_sentence_idx_to_image_idx[sent_idx]
                assert(current_image_idx == image_idx)
                
                similar_sentences_idx = self.test_set_image_idx_to_sentence_indices[image_idx]
                assert(sent_idx in similar_sentences_idx)
                
                #logger.info("sent_idx: %i, similar_sentences_ids: %s" % (sent_idx, similar_sentences_idx))
                
                # get distances/rankings for sentence `sent_idx`
                sentence_embedding_idx = self.sentence_idx_to_sentence_embedding_idx[sent_idx]
                this_sentence_embedding = self.sentence_embeddings[sentence_embedding_idx]
                distance_images_given_sentence, ranking_images_given_sentence = \
                        self.get_distances_from_image_embedding(this_sentence_embedding)
                
                # get distances/rankings for image `image_idx`
                image_embedding_idx = self.image_idx_to_image_embedding_idx[image_idx]
                this_image_embedding = self.image_embeddings[image_embedding_idx]
                distance_sentences_given_image, ranking_sentences_given_image = \
                        self.get_distances_from_sentence_embedding(this_image_embedding)
                
                # rankings are all computed on the embeddings IDs
                if image_embedding_idx in ranking_images_given_sentence[:1]:
                    images_given_sentence_recall_at_1.append( sentence_embedding_idx )
                if image_embedding_idx in ranking_images_given_sentence[:5]:
                    images_given_sentence_recall_at_5.append( sentence_embedding_idx )
                if image_embedding_idx in ranking_images_given_sentence[:10]:
                    images_given_sentence_recall_at_10.append( sentence_embedding_idx )
                
                if sentence_embedding_idx in ranking_sentences_given_image[:1]:
                    sentences_given_image_recall_at_1.append( image_embedding_idx )
                if sentence_embedding_idx in ranking_sentences_given_image[:5]:
                    sentences_given_image_recall_at_5.append( image_embedding_idx )
                if sentence_embedding_idx in ranking_sentences_given_image[:10]:
                    sentences_given_image_recall_at_10.append( image_embedding_idx )
                
                # medium rank
                # rankings are computed based on the embeddings ids.
                # dictionaries built using original (minibatch) ids.
                # images given sentence
                for counter, ranked_image_idx in enumerate(ranking_images_given_sentence):
                    if self.image_embedding_idx_to_image_idx[ranked_image_idx] == \
                            self.test_set_sentence_idx_to_image_idx[sent_idx]:
                        images_given_sentence_medium_rank.append(counter)
                        break
                # sentences given image
                for counter, ranked_sentence_idx in enumerate(ranking_sentences_given_image):
                    #if self.sentence_embedding_idx_to_sentence_idx[ranked_sentence_idx] == \
                    #        sent_idx:
                    if self.sentence_embedding_idx_to_sentence_idx[ranked_sentence_idx] in \
                            similar_sentences_idx:
                        sentences_given_image_medium_rank.append(counter)
                        break
            
            logger.info("minibatch %i-%i/%i " %
                        (x_sentt_idx[0], x_sentt_idx[-1], n_sentences_total))
            
            logger.info("[sentence-image %i-%i] images given sentence rank: %i/%i" %
                        (sent_idx, current_image_idx, 
                         np.asarray(images_given_sentence_medium_rank).mean(),
                         n_images_total))
            logger.info("[sentence-image %i-%i] sentences given image rank: %i/%i" %
                        (sent_idx, current_image_idx,
                         np.asarray(sentences_given_image_medium_rank).mean(),
                         n_sentences_total))
        
        logger.info("Final results:")
        logger.info("images given sentence medium rank: %i/%i -- %.2f%%" %
                    (np.asarray(images_given_sentence_medium_rank).mean(),
                     n_images_total,
                     (np.asarray(images_given_sentence_medium_rank).mean() / n_images_total)))
        logger.info("images given sentence R@1: %i" %
                    (len(images_given_sentence_recall_at_1)))
        logger.info("images given sentence R@5: %i" %
                    (len(images_given_sentence_recall_at_5)))
        logger.info("images given sentence R@10: %i" %
                    (len(images_given_sentence_recall_at_10)))
        
        logger.info("sentences given image medium rank: %i/%i -- %.2f%%" %
                    (np.asarray(sentences_given_image_medium_rank).mean(),
                     n_sentences_total,
                     (np.asarray(sentences_given_image_medium_rank).mean() / n_sentences_total)))
        logger.info("sentences given image R@1: %i" %
                    (len(sentences_given_image_recall_at_1)))
        logger.info("sentences given image R@5: %i" %
                    (len(sentences_given_image_recall_at_5)))
        logger.info("sentences given image R@10: %i" %
                    (len(sentences_given_image_recall_at_10)))
    
    
    def fit(self, train_set_size=None,
                  valid_set_size=None,
                  test_set_size=None,
                  minibatch_size=None,
                  validation_frequency=-1,
                  save_frequency=-1):
        assert(not train_set_size == None and not minibatch_size == None)
        
        if valid_set_size is None:
            valid_set_size = train_set_size
        
        # (adaptive) learning rate
        l_r = T.scalar('learning_rate', dtype=theano.config.floatX)
        # momentum
        mom = T.scalar('momentum', dtype=theano.config.floatX)
        
        # cost to be observed (prior to regularisation)
        # minimum squared error between predicted sentence and image vectors
        cost = T.mean((self.image_proj.y_pred - self.sentence_proj.last_h)** 2)
        
        # cost to be minimised
        # in case we are dealing with ONLY positive instances, minimise
        # minimum squared difference between predicted sentence and image vectors
        reg_cost = cost + self.L1_reg * self.sentence_proj.L1 \
                        + self.L1_reg * self.image_proj.L1 \
                        + self.L2_reg * self.sentence_proj.L2_sqr \
                        + self.L2_reg * self.image_proj.L2_sqr
        reg_cost.name = 'cost_with_regularisation'
        
        # total loss is just the sum of image and sentence networks' losses
        total_loss = self.sentence_proj.loss(self.image_proj.y_pred) \
            + self.image_proj.loss(self.sentence_proj.last_h)
        
        # compute loss given minibatch
        compute_loss = theano.function(inputs=[self.x_sentence,
                                               self.x_sentence_mask,
                                               self.x_image],
                                       outputs=total_loss)
        
        # update parameters
        # compute the gradient of cost with respect to model parameters
        # gradients on the weights using BPTT
        gparams_sentence = []
        gparams_image = []
        # text rnn
        for param in self.sentence_proj.params:
            gparam = T.grad(cost, param)
            gparams_sentence.append(gparam)
        # image nn
        for param in self.image_proj.params:
            gparam = T.grad(cost, param)
            gparams_image.append(gparam)
        
        updates = OrderedDict()
        # text rnn
        for param, gparam in zip(self.sentence_proj.params, gparams_sentence):
            weight_update = self.sentence_proj.updates[param]
            upd = mom * weight_update - l_r * gparam
            updates[weight_update] = upd
            updates[param] = param + upd
        # image nn
        for param, gparam in zip(self.image_proj.params, gparams_image):
            weight_update = self.image_proj.updates[param]
            upd = mom * weight_update - l_r * gparam
            updates[weight_update] = upd
            updates[param] = param + upd
        
        # compute cost given minibatch and update model parameters
        train_model = theano.function(
            inputs=[self.x_sentence,
                    self.x_sentence_mask,
                    self.x_image,
                    l_r,
                    mom],
            outputs=reg_cost,
            updates=updates,
            on_unused_input='warn')
        
        n_train_batches = train_set_size / minibatch_size
        
        # go through this many minibatches before checking the network
        # on the validation set; in this case we check every epoch
        if validation_frequency == -1:
            validation_frequency = n_train_batches
            #validation_frequency = min(n_train_batches, patience / 2)
        
        # save model at every 5 epochs
        if save_frequency == -1:
            save_frequency = n_train_batches * 5
            #save_frequency = min(n_train_batches, patience / 2)
        
        logger.info("validation frequency: %i" % validation_frequency)
        logger.info("save frequency: %i" % save_frequency)
        logger.info("training set size: %i" % train_set_size)
        logger.info("number of training batches: %i" % n_train_batches)
        logger.info("minibatch size: %i" % minibatch_size)
        
        best_validation_loss = np.inf
        early_stop = False
        
        logger.info("Training...")
        
        # some variables used for model analysis
        for epoch in range(self.last_epoch+1, self.n_epochs):
            # iterate train set
            for n_iterations, minibatch in enumerate(self.__train_stream.get_epoch_iterator()):
                x_sent_idx, x_image_idx, x_sent, x_mask, x_image = minibatch # unpack minibatch
                
                x_sent_idx = x_sent_idx.flatten()
                x_image_idx = x_image_idx.flatten()
                
                #logger.info("x_sent_idx: %s" % (str(x_sent_idx)))
                #logger.info("x_image_idx: %s" % (str(x_image_idx)))
                
                effective_momentum = self.final_momentum \
                                   if epoch > self.momentum_switchover \
                                   else self.initial_momentum
                
                # swap dimensions to put time as dimension zero,
                # minibatch as dimension one
                x_sent = np.swapaxes(x_sent, 0, 1)
                x_mask = np.swapaxes(x_mask, 0, 1)
                
                # train on minibatch and update model
                minibatch_regularised_cost = train_model(
                    x_sent, x_mask, x_image,
                    self.learning_rate, effective_momentum)
                
                # iteration number (how many weight updates have we made? 0-indexed)
                n_iterations += 1
                iter = (epoch) * n_train_batches + n_iterations
                
                if iter % validation_frequency == 0:
                    valid_losses = []
                    # iterate validation minibatch
                    for valid_minibatch in self.__valid_stream.get_epoch_iterator():
                        _, _, x_sentv, x_maskv, s_imagev = valid_minibatch
                        
                        x_sentv = np.swapaxes(x_sentv, 0, 1)
                        x_maskv = np.swapaxes(x_maskv, 0, 1)
                        
                        this_valid_loss = compute_loss(x_sentv, x_maskv, s_imagev)
                        valid_losses.append(this_valid_loss)
                    this_valid_loss = np.mean(valid_losses)
                    
                    # update best validation loss for early stopping
                    if this_valid_loss < best_validation_loss:
                        # save model and improve patience if loss improvement is good enough
                        if this_valid_loss < best_validation_loss * self.improvement_threshold:
                            self.patience = max(self.patience, iter * self.patience_increase)
                            #logger.info("new patience: %i"%patience)
                        
                        best_validation_loss = this_valid_loss
                        bad_counter = 0
                    
                    # evaluate on test set
                    if not test_set_size==None:
                        test_losses = []
                        # iterate test minibatch
                        for test_minibatch in self.__test_stream.get_epoch_iterator():
                            _, _, x_sentt, x_maskt, x_imaget = test_minibatch
                            
                            x_sentt = np.swapaxes(x_sentt, 0, 1)
                            x_maskt = np.swapaxes(x_maskt, 0, 1)
                            
                            this_test_loss = compute_loss(x_sentt, x_maskt, x_imaget)
                            test_losses.append(this_test_loss)
                        this_test_loss = np.mean(test_losses)
                        self.all_test_losses.append(this_test_loss)
                    
                        logger.info('epoch %i, minibatch %i/%i, iter %i, train loss %f '
                                    ', valid loss %f, test loss %f, patience: %i, lr: %f' % \
                                    (epoch, n_iterations*minibatch_size, train_set_size,
                                     iter, minibatch_regularised_cost, this_valid_loss,
                                     this_test_loss, self.patience, self.learning_rate))
                    else:
                        logger.info('epoch %i, minibatch %i/%i, iter %i, train loss %f '
                                    ', valid loss %f, patience: %i, lr: %f' % \
                                    (epoch, n_iterations*minibatch_size, train_set_size,
                                     iter, minibatch_regularised_cost, this_valid_loss,
                                     self.patience, self.learning_rate))
                    
                    self.all_train_losses.append(minibatch_regularised_cost)
                    self.all_valid_losses.append(this_valid_loss)
                    self.time_measures.append(time.time())
                    self.clock_measures.append(time.clock())
                    
                    #logger.info("len(all_valid_losses): %i"%len(self.all_valid_losses))
                    #logger.info("patience: %i"%patience)
                    
                    if len(self.all_valid_losses) > self.patience and \
                            this_valid_loss >= \
                            np.array(self.all_valid_losses)[:-self.patience].min():
                        logger.info('Bad counter increase (to %d)' % bad_counter)
                        bad_counter += 1
                        if bad_counter > self.patience:
                            logger.info('Early Stop!')
                            early_stop = True
                            break
                    
                    self.__valid_stream.reset()
                    self.__test_stream.reset()
                
                if iter % save_frequency == 0:
                    self.last_epoch = epoch
                    self.save()
                    
            
            # finish after this many updates
            if iter >= self.finish_after:
                logger.info('Finishing after %d iterations!' % iter)
                early_stop = True
            
            if early_stop:
                break
            
            self.learning_rate *= self.learning_rate_decay
            self.__train_stream.reset()
        
        # plot results of model applied on training/valid/test sets
        # currently not using CPU time from time.clock()
        plot_losses_vs_time(train_losses=self.all_train_losses,
                            valid_losses=self.all_valid_losses,
                            test_losses=self.all_test_losses,
                            time_measures=self.time_measures,
                            time_label='Time (secs)')
        logger.info("last epoch: %i, n_epochs: %i" % (self.last_epoch, self.n_epochs))