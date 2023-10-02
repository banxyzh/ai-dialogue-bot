# -*- coding: utf-8 -*-

from glob import glob
import os
import codecs
import numpy as np
import re
import _pickle as cPickle
import collections

PAD = "_PAD"
GO = "_GO"
EOS = "_EOS"
UNK = "_UNK"
UNK_ID = 3
PAD_ID = 0
START_VOCAB = [PAD, GO, EOS, UNK]

def normalize_unicodes(text):
  text = normalize_punct