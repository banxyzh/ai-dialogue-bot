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
    self.global_step = tf.Variable(0, name="global_step", trainable=False)

    with tf.device("/cpu:0"):
      self.embedding = tf.get_variable("embedding", [vocab_size, emb_size])
      inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

    if self._keep_prob < 1 and not infer:
      inputs = tf.nn.dropout(inputs, keep_prob=self._keep_prob)

    with tf.variable_scope("encoder", initializer=glorot()):
      fw_cell = GRUCell(emb_size)
      bw_cell = GRUCell(emb_size)
      if self._keep_prob < 1 and not infer:
        fw_cell = DropoutWrapper(fw_cell, output_keep_prob=self._keep_prob)
        bw_cell = DropoutWrapper(bw_cell, output_keep_prob=self._keep_prob)

    with tf.variable_scope("context", initializer=glorot()):
      ctx_cell = GRUCell(memory_size * 2)
      self.ctx_w = tf.get_variable("context_w", [memory_size * 2, memory_size])
      self.ctx_b = tf.get_variable("context_b", [memory_size], initializer=init_ops.zeros_initializer())
      self.initial_state = ctx_cell.zero_state(self._batch_size, tf.float32)

    with tf.variable_scope("decoder", initializer=glorot()):
      # GRU with conditional distribution in sec 2.2 of https://arxiv.org/pdf/1406.1078.pdf
      dec_cell = GRUCellCond(memory_size)

    self.outputs, self.output_ids, _, self.final_state = self.seq2seq(inputs, fw_cell, bw_cell, ctx_cell, dec_cell)

    loss = self.get_loss(self.outputs)
    self.loss = tf.reduce_mean(loss)
    tf.summary.scalar('loss', self.loss)

    tvars = tf.trainable_variables()

    print("parameter size:", _count_param_size(tvars))

    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

  def get_loss(self, outputs):
    output_maxlen = tf.minimum(tf.shape(outputs)[1], self._max_seq_length)
    out_data_slice = tf.slice(self.output_data, [0, 0], [-1, output_maxlen])
    out_logits_slice = tf.slice(outputs, [0, 0, 0], [-1, output_maxlen, -1])

    with tf.name_scope("costs"):
      # We need to delete zeroed elements in targets, beyond max sequence
      length_mask = tf.sequence_mask(self.output_lengths, maxlen=output_maxlen, dtype=tf.float32)
      final_loss = seq2seq.sequence_loss(out_logits_slice, out_data_slice, length_mask)
      return final_loss

  def seq2seq(self, inputs, fw_cell, bw_cell, ctx_cell, dec_cell, reuse=False):
    with tf.variable_scope("seq2seq") as scope:
      if reuse:
        scope.reuse_variables()
      enc_outputs, enc_state = self.encode(fw_cell, bw_cell, inputs)
      ctx_outputs, ctx_state = self.contextual(ctx_cell, enc_state)
      dec_outputs, dec_sample_id, dec_state = self.decode(dec_cell, enc_outputs, ctx_ou