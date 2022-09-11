
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pprint
import _pickle as cPickle
from model import DialogueModel
from utils import TextLoader, UNK_ID, PAD_ID
from glob import glob

checkpoint = "/tmp/model.ckpt"

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_string("checkpoint", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("logdir", "log", "Log directory [log]")
flags.DEFINE_float("temperature", 0.5, "temperature")
FLAGS = flags.FLAGS

def main(_):