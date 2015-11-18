#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import h5py
import codecs
from collections import defaultdict, OrderedDict
import sys, os


import logging
reload(logging)
logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


from clean_wikitext import CleanText
import re

c = CleanText('en')

class SentenceVector:
    """ contains logic for turning sentences composed of strings
        into arrays of word vectors (or word vector ids),
        accounting for OOV words and end-of-sentence symbol
    """
    def __init__(self, sentence, word_vectors, lower=True,
                language='en', idx=None):
        """ type idx:           int
            idx:                index of the sentence in relation to image,
                                if applicable.

            type sentence:      list or basestring
            sentence:           sentence to be processed into word vectors or
                                word vector ids

            type word_vectors:  object instance of class WordVectors
            word_vectors:       encodes word vectors and word vectors ids

            type lower:         boolean
            lower:              whether to work with lowercased words or not
        """
        # if it is not an array, split sentence into words
        if isinstance(sentence, basestring):
            sentence = sentence.split(" ")
        assert(type(sentence) is list)

        #if not language=='en':
        c = CleanText(language)

        sentence = list(c.process_line( " ".join(sentence) ))
        if not len(sentence)==0:
            sentence = sentence[0].strip().split(" ")
        else:
            sentence = ['']

        self.word_vectors = word_vectors
        self.sentence = self.lowercase(sentence) if lower else sentence
        self.sent_vec = None
        self.sent_ids = None

        if idx is not None:
            self.idx = int(idx)

    def lowercase(self, sent):
        """ lowercase sentence `sent` given as parameter

            type sent:  list(str)
            sent:       sentence to be lowercased
        """
        SPECIAL_TAG_PREFIX = ur"__"
        _sent = []
        for w in sent:
            # make sure w is in network vocabulary
            if not w.startswith(SPECIAL_TAG_PREFIX) and \
                    w.lower() in self.word_vectors.getVocab():
                _sent.append(w.lower())
            elif w.startswith(SPECIAL_TAG_PREFIX) and \
                    w in self.word_vectors.getVocab():
                _sent.append(w)
            else:
                _sent.append( self.word_vectors.OOV_SYMBOL )
        return _sent


    def text_to_vecs(self):
        """ this function returns a list with word vectors for
            the given sentence
        """
        # convert word strings into word vectors
        sent_vec = []
        for w in self.sentence:
            if w in self.word_vectors.getVocab():
                sent_vec.append( self.word_vectors.getWordVectors()[w] )
            else:
                sent_vec.append( self.word_vectors.getOOVWordVector() )
        
        assert(len(self.sentence) == len(sent_vec)) 
        self.sent_vec = sent_vec


    def text_to_ids(self):
        sent_ids = []
        for w in self.sentence:
            if w in self.word_vectors.getVocab():
                sent_ids.append( self.word_vectors.getWords2IdsDictionary()[w] )
            else:
                sent_ids.append( self.word_vectors.getOOVWordId() )
        self.sent_ids = sent_ids

    
    def get_sentence(self):
        return self.sentence
    
    def get_sentence_vector(self):
        if self.sent_vec is None:
            self.text_to_vecs()
        return self.sent_vec

    def get_sentence_ids(self):
        if self.sent_ids is None:
            self.text_to_ids()
        return self.sent_ids
    
    def get_idx(self):
        return self.idx


class WordVectors(object):
    def __init__(self,
            path_to_vectors='/home/icalixto/resources/wikipedia/extract_wikipedia_corpus_from_xml/enwiki',
            vectors_file='vectors.300.txt',
            vocab_file='vocab.txt',
            word_vectors_dimensionality=300,
            add_eos_token=False):

        self.OOV_SYMBOL = "__UNK__"
        self.EOS_SYMBOL = "__EOS__"

        # let's load vocabulary into dictionary
        # and word vectors into another dictionary
        PATH_TO_VEC = path_to_vectors
        vec_fname = vectors_file
        vocab_fname = vocab_file

        full_vec_fname = PATH_TO_VEC+"/"+vec_fname
        full_vocab_fname = PATH_TO_VEC+"/"+vocab_fname
        fh_vec = codecs.open(full_vec_fname, 'r')
        fh_vocab = codecs.open(full_vocab_fname, 'r')

        logger.info( "Loading vocabulary..." )
        vocab = set()
        #vocab = OrderedDict()
        for line in fh_vocab:
            line = line.strip()
            line = line.split(" ")
            #vocab[line[0]] = line[1]
            vocab.add( line[0] )
        self.vocab = vocab

        logger.info( "vocab: loaded %d words."%(len(vocab)) )

        logger.info( "Loading word vectors..." )
        word_vecs = OrderedDict()
        for line in fh_vec:
            line = line.strip()
            line = line.split(" ", 1)
            word = line[0]
            array_str = line[1]
            vec = np.fromstring(array_str, dtype=np.float32, sep=" ", count=word_vectors_dimensionality)
            word_vecs[word] = vec


        logger.info( "adding out-of-vocabulary tag to word vectors..." )
        oov_vec = [0] * word_vectors_dimensionality
        for c, (_,v) in enumerate(word_vecs.iteritems()):
            if c==0:
                oov_vec = oov_vec+v
            else:
                oov_vec = oov_vec+(v/2)
        word_vecs[ self.OOV_SYMBOL ] = oov_vec
        #self.vocab.add( self.OOV_SYMBOL )

        self.eos_vector = np.zeros((word_vectors_dimensionality,), dtype=np.float32)
        if add_eos_token:
            self.vocab.add( self.EOS_SYMBOL )
            logger.info( "adding end-of-sentence token to word vectors..." )
            word_vecs[ self.EOS_SYMBOL ] = self.eos_vector
            logger.info( "done." )

        logger.info( "word vectors: loaded %d word vectors."%(len(word_vecs)) )

        for word, word_vec in word_vecs.iteritems():
            #print type(word_vec)
            if np.isnan(np.min(word_vec)):
                logger.warning( "found NaN: %s" % word_vec )

        self.word_vecs = word_vecs

        # create auxiliary dictionaries
        logger.info( "Creating auxiliary word2id and id2word dictionaries..." )
        self.words_to_ids = OrderedDict()
        for c, word in enumerate(word_vecs.keys()):
            self.words_to_ids[word] = c
        self.ids_to_words = OrderedDict()
        for k,v in self.words_to_ids.iteritems():
            self.ids_to_words[v] = k
        logger.info("Done!")


    def getOOVWordId(self):
        return self.words_to_ids[ self.OOV_SYMBOL ]

    def getOOVWordVector(self):
        return self.word_vecs[ self.OOV_SYMBOL ]

    def getEOSWordId(self):
        return self.words_to_ids[ self.EOS_SYMBOL ]

    def getEOSWordVector(self):
        return self.word_vecs[ self.EOS_SYMBOL ]

    def getWordVectors(self):
        return self.word_vecs

    def getWords2IdsDictionary(self):
        return self.words_to_ids

    def getIds2WordsDictionary(self):
        return self.ids_to_words

    def getVocab(self):
        return self.vocab

if __name__ == "__main__":
    w = WordVectors(add_eos_token=True).getWordVectors()
    logger.info( "Length of word vectors: %i" % len(w) )
