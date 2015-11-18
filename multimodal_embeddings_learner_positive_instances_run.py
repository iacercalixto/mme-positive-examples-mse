#%load_ext autoreload
#%autoreload 2
#%matplotlib inline

import matplotlib.pyplot as plt
plt.ion()

import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import os, sys, time
import logging
reload(logging)
logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy as np
import theano
import theano.tensor as T

# by loading `load_config` script we make sure our classes are accessible in python path
# as well as loading variables save_file_name and project_path
from load_config import project_path, save_file_name

from Flickr8k import Flickr8k
from FlickrDummy import FlickrDummy
from load_word_vectors import WordVectors
from multimodal_embeddings_learner_positive_instances import MultimodalEmbeddingsLearner


def load_dataset_flickr8k(batch_size, n_images=4, split='train', verbose=False,
                          sources=['sentence_id',
                                   'image_id',
                                   'sentence_words_ids',
                                   'image_features']):
    flickr = Flickr8k()
    data_stream, _, data_set_size = flickr.load_flickr8k(n_images=n_images,
                                     which_sets=[split],
                                     batch_size=batch_size,
                                     verbose=verbose,
                                     sources=sources)
    return data_stream, data_set_size


def load_dataset_flickr_dummy(n_images=None, batch_size=21, split='train',
                              verbose=False,
                              sources=['sentence_id',
                                       'image_id',
                                       'sentence_words_ids',
                                       'image_features']):
    # not using parameter n_images
    flickr = FlickrDummy()
    data_stream, _, data_set_size = flickr.load_flickr_dummy(
                                     which_sets=[split],
                                     batch_size=batch_size,
                                     verbose=verbose,
                                     sources=sources)
    return data_stream, data_set_size


def train_multimodal_embedding(n_epochs=5000, train=False, test=False):
    """ Test RNN with binary outputs. """
    # sentence embeddings and image embeddings are of the same size
    # (the size of the multimodal embeddings)
    #word_emb_dim = 50              # dimension of word embeddings
    #multimodal_emb_dim = 50        # dimension of multimodal embeddings
    word_emb_dim = 300              # dimension of word embeddings
    image_fv_dim = 4096             # dimension of image feature vector
    multimodal_emb_dim = 500        # dimension of multimodal embeddings
    
    #n_images_train = 100
    #n_images_valid = 100
    #n_images_test  = 100
    #minibatch_size = 10
    n_images_train = None
    n_images_valid = None
    n_images_test  = None
    minibatch_size = 40
    
    n_hidden_layers = 1
   
    # total size of data sets is five times the number of images
    load_dataset = load_dataset_flickr8k
    train_stream, train_set_size = load_dataset(n_images=n_images_train,
                                                split='train',
                                                batch_size=minibatch_size)
    valid_stream, valid_set_size = load_dataset(n_images=n_images_valid,
                                                split='other',
                                                batch_size=minibatch_size)
    test_stream, test_set_size = load_dataset(n_images=n_images_test,
                                                split='test',
                                                batch_size=minibatch_size)
    
    # load word vectors' parameters
    vecs_fname = "vectors.%i.txt" % word_emb_dim
    wv_obj = WordVectors(vectors_file=vecs_fname,
                         word_vectors_dimensionality=word_emb_dim,
                         add_eos_token=True)
    word_vecs = wv_obj.getWordVectors()
    vocab_size = len(wv_obj.getVocab())
    
    # convert word vectors dictionary into a numpy float32 array
    word_vecs_np = []
    for vec in word_vecs.values():
        word_vecs_np.append( vec )
    word_vecs_np = np.asarray(word_vecs_np, dtype=theano.config.floatX)
    
    model = MultimodalEmbeddingsLearner(
        train_stream             = train_stream,
        valid_stream             = valid_stream,
        test_stream              = test_stream,
        word_embeddings          = word_vecs_np,
        word_emb_dim             = word_emb_dim,
        image_fv_dim             = image_fv_dim,
        multimodal_emb_dim       = multimodal_emb_dim,
        learning_rate            = 0.001,
        learning_rate_decay      = 0.999,
        n_epochs                 = n_epochs,
        patience                 = 5000,
        activation               = T.nnet.sigmoid,
        vocabulary_size          = vocab_size,
        proj_name                = save_file_name,
        proj_folder              = project_path,
        load_model               = True,
        n_layers                 = n_hidden_layers)
    
    if train:
        model.fit(train_set_size = train_set_size,
                  valid_set_size = valid_set_size,
                  test_set_size  = test_set_size,
                  minibatch_size = minibatch_size,
                  validation_frequency = train_set_size/minibatch_size, # validate at every epoch
                  save_frequency = train_set_size/minibatch_size        # save model at every epoch
                  #validation_frequency = 1, # validate at every X model updates
                  #save_frequency = 1        # save model at every Y model updates
        )
    
    if test:
        model.test()
    
    if not train and not test:
        logger.info("Nothing to do!")


logging.basicConfig(level=logging.INFO)
t0 = time.time()
train_multimodal_embedding(n_epochs=5000, train=True, test=False)
print "Elapsed time: %f" % (time.time() - t0)
