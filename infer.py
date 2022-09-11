
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
  config = cPickle.load(open(FLAGS.logdir + "/hyperparams.pkl", 'rb'))
  pp.pprint(config)

  try:
    # pre-trained chars embedding
    emb = np.load("./data/emb.npy")
    chars = cPickle.load(open("./data/vocab.pkl", 'rb'))
    vocab_size, emb_size = np.shape(emb)
    data_loader = TextLoader('./data', 1, chars)
  except Exception:
    data_loader = TextLoader('./data', 1)
    emb_size = config["emb_size"]
    vocab_size = data_loader.vocab_size

  checkpoint = FLAGS.checkpoint + '/model.ckpt'