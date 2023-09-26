
import tensorflow as tf
import model
import pprint
import _pickle as cPickle
from glob import glob
import math
import sys
import numpy as np
from utils import TextLoader, UNK_ID
from model import DialogueModel

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("num_epochs", 25, "Epoch to train [25]")
flags.DEFINE_integer("memory_size", 300, "Memory size [300]")
flags.DEFINE_integer("emb_size", 300, "The dimension of embedding matrix [300]")
flags.DEFINE_integer("batch_size", 32, "The size of batch [32]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate [0.001]")
flags.DEFINE_float("keep_prob", 0.5, "Dropout rate [0.5]")
flags.DEFINE_float("grad_clip", 5.0, "Grad clip [5.0]")
flags.DEFINE_integer("temperature", 5, "temperature [5]")
flags.DEFINE_string("checkpoint", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("logdir", "log", "Log directory [log]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(FLAGS.__flags)