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

class DialogueModel(object):
  def __init__(self, batch_size, max_seq_length, vocab_size,
               start_token_id=1, end_token_id=2, pad_token_id=0, unk_token_id=3,
               emb_size=100, memory_size=100, keep_prob=0.5, temperature=0.5, antilm=0.55,
               learning_rate=0.001, grad_clip=5.0, infer=False):

    self._batch_size = batch_size
    self._vocab_size = vocab_size
    self._memory_size = memory_size
    self._start_token_id = start_token_id
    self._end_token_id = end_token_id
    self._max_seq_length = max_seq_length
    self._unk_token_id = unk_token_id
    self._keep_prob = keep_prob
    self._temperature = temperature
    self._start_token_id = start_token_id
    self._end_token_id = end_token_id
    self._pad_token_id = pad_token_id
    self._infer = infer
    self._antilm = antilm

    self.input_data = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_data")
    self.input_lengths = tf.placeholder(tf.int32, shape=[batch_size], name="input_lengths")
    self.output_data = tf.placeholder(tf.int32, [batch_size, max_seq_length], name='output_data')
    self.output_lengths = tf.placeholder(tf.int32, [batch_size], name='output_lengths')
    self.global_step = tf.Var