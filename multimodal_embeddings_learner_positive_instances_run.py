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
import argparse


# by loading `load_config` script we make sure our classes are accessible in python path
# as well as loading variables save_file_name and project_path
from load_config import project_path, save_file_name

from Flickr8k import Flickr8k
from Flickr30k import Flickr30k
from FlickrDummy import FlickrDummy
from load_word_vectors import WordVectors
from multimodal_embeddings_learner_positive_instances import MultimodalEmbeddingsLearner


def load_dataset_flickr30k(batch_size, n_images=4, split='train', verbose=False,
                          sources=['sentence_id',
                                   'image_id',
                                   'sentence_words_ids',
                                   'image_features']):
    flickr = Flickr30k()
    data_stream, _, data_set_size = flickr.load_flickr30k(n_images=n_images,
                                     which_sets=[split],
                                     batch_size=batch_size,
                                     verbose=verbose,
                                     sources=sources)
    return data_stream, data_set_size


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


def run_multimodal_embedding(n_epochs=5000, train=False, test=False,
        corpus='flickr8k', create_on_load_error=False,
        save_file_name_cl=None, l1_reg=0., l2_reg=0.):

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
    #minibatch_size = 40
    minibatch_size = 100
    
    n_hidden_layers = 1
    
    if corpus == 'flickr8k':
        load_dataset = load_dataset_flickr8k
    elif corpus == 'flickr30k':
        load_dataset = load_dataset_flickr30k
    else:
        raise Exception("Corpus %s not supported." % str(corpus))

    # total size of data sets is five times the number of images
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
    
    if not save_file_name_cl is None:
        logger.info("Using file name %s as model file (default was %s)." %
                (save_file_name_cl, save_file_name))
        save_file_name_ = save_file_name_cl
    else:
        save_file_name_ = save_file_name

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
        proj_name                = save_file_name_,
        proj_folder              = project_path,
        load_model               = True,
        n_layers                 = n_hidden_layers,
        # parameters from the command line
        stop_on_load_error       = not create_on_load_error,
        L1_reg                   = l1_reg,
        L2_reg                   = l2_reg)
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=False, default=True,
                        action='store_true', help="Train multimodal embedding on training set (choose from corpora).")
    parser.add_argument("--test",  required=False, default=False,
                        action='store_true', help="Test multimodal embedding on test set (choose from corpora).")
    parser.add_argument("--create-on-load-error", required=False, default=False, action='store_true',
                        help="Whether to create a new model file in case a pre-trained model cannot be loaded. " +
                        "If not set, an exception is raised in case presaved model file cannot be loaded.")
    parser.add_argument("--save-base-file-name", required=False, default=None, type=str,
                        help="Base file name to use to save trained model parameters. Default set in `vars.cfg` file.")

    # l1 and l2 regularisers
    parser.add_argument("--L1-reg", required=False, default=0., type=float,
                        help="L1 regularisation coefficient.")
    parser.add_argument("--L2-reg", required=False, default=0., type=float,
                        help="L2 regularisation coefficient.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--flickr8k", action="store_true")
    group.add_argument("--flickr30k", action="store_true")
    args = parser.parse_args()

    if args.flickr8k:
        corpus = 'flickr8k'
    if args.flickr30k:
        corpus = 'flickr30k'

    #print "args: %s" % str(args)
    #print "corpus: %s" % str(corpus)
    #print "create on load error: %s" % str(args.create_on_load_error)
    #sys.exit(0)

    logging.basicConfig(level=logging.INFO)
    t0 = time.time()

    run_multimodal_embedding(
            n_epochs             = 5000,
            train                = args.train,
            test                 = args.test,
            corpus               = corpus,
            create_on_load_error = args.create_on_load_error,
            save_file_name_cl    = args.save_base_file_name,
            l1_reg               = args.L1_reg,
            l2_reg               = args.L2_reg)

    print "Elapsed time: %f" % (time.time() - t0)
