import os, sys
import ConfigParser

config_fname = 'vars.cfg'

# first, read some configuration variables from disk
config = ConfigParser.RawConfigParser()
config.read( config_fname )

flickr8k_path = config.get("general", "flickr8k_path")
flickr30k_path = config.get("general", "flickr30k_path")
flickr_dummy_path = config.get("general", "flickr_dummy_path")
word_vectors_path = config.get("general", "word_vectors_path")
project_path = config.get("general", "project_path")
save_file_name = config.get("general", "save_file_name")

sys.path.append(flickr8k_path)
sys.path.append(flickr30k_path)
sys.path.append(flickr_dummy_path)
sys.path.append(word_vectors_path)
sys.path.append(project_path)
assert( project_path == os.path.dirname(os.path.realpath(__file__)) )

