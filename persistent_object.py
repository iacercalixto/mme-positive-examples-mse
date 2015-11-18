from abc import ABCMeta, abstractmethod
import sys, os
import cPickle as pickle
import traceback

import os, sys
import logging
reload(logging)
logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class PersistentObject(object):
    """ Classes that inherit from this class can be easily persistable on disk.
        The methods 'getState(self)' and 'setState(self, state)' are abstract
        and must be implemented by the child class.
        
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, proj_folder='.', proj_name=None, stop_on_load_error=False):
        """
            Folder where model is persisted is `proj_folder/`
            File name for model persistance is `proj_name` (if it ends with `.pkl`
            or `proj_name`+`.sav` if it does not.
            
            proj_folder:        folder to use to persist model
            proj_name:          project name to use to persist model
            stop_on_load_error: whether to exit and stop in case we try to
                                load a model that does not exist or create it
                                on the fly and print a logger.warning() message.
        """
        object.__init__(self)
        assert(not proj_name == None)
        
        self.proj_folder        = proj_folder
        self.proj_name          = proj_name
        self.stop_on_load_error = bool(stop_on_load_error)
    
    @abstractmethod
    def getState(self):
        """ Method that must return a list containing all the
            model's parameters to be persisted on disk.
        """
        return
    
    @abstractmethod
    def setState(self, state):
        """ Method that must set model's parameters,
            given a list with parameters loaded from disk.
        """
        return
    
    def save(self):
        """ Save a pickled representation of model state. """
        #fpath = os.path.join(self.proj_folder, self.proj_name)
        fpath = os.path.join(self.proj_folder)
        fname = self.proj_name + ".sav"
        
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)
            
        elif fname is None:
            import datetime
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)
        
        fabspath = os.path.join(fpath, fname)
        
        logger.info("Saving model to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.getState()
        try:
            pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        except AttributeError:
            logger.error("Method getState() not implemented!")
            sys.exit(1)
        finally:
            file.close()
    
    def load(self):
        """ Load model from disk. """
        #fpath = os.path.join(self.proj_folder, self.proj_name)
        fpath = os.path.join(self.proj_folder)
        fname = self.proj_name + ".sav"
        path = os.path.join(fpath, fname)
        
        if not os.path.isfile(path):
            if self.stop_on_load_error:
                logger.error("Could not find model file %s !" % str(path))
                sys.exit(1)
            logger.warning("Could not find model file %s . Creating model..." % str(path))
        else:
            logger.info("Loading model from %s ..." % str(path))
            file = open(path, 'rb')
            state = pickle.load(file)
            try:
                self.setState(state)
            except AttributeError:
                logger.error("Method setState() not implemented!")
                traceback.print_exc()
                sys.exit(1)
            finally:
                file.close()