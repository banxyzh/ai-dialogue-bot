import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib.rnn import MultiRNNCell, DropoutWrapper, GRUCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.layers import xavier_initializer as glorot
from grucell_cond import GRUCellCond, CondWrapper

def _count_param_size(tvars):
  # parameters count
  count = 0
  for tvar in tvars:
    c = 1
    for var in list(tvar.shape):
      c = c * int(var)
    count = count + c
  return count

class DialogueModel(obje