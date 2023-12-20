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
  text = normalize_punctuation(text)
  text = "".join([Q2B(c) for c in list(text)])
  return text

def replace_all(repls, text):
  # return re.sub('|'.join(repls.keys()), lambda k: repls[k.group(0)], text)
  return re.sub(u'|'.join(re.escape(key) for key in repls.keys()),
                lambda k: repls[k.group(0)], text)

def normalize_punctuation(text):
  cpun = [['	'],
          [u'﹗'],
          [u'“', u'゛', u'〃', u'′'],
          [u'”'],
          [u'´', u'‘', u'’'],
          [u'；', u'﹔'],
          [u'《', u'〈', u'＜'],
          [u'》', u'〉', u'＞'],
          [u'﹑'],
          [u'【', u'『', u'〔', u'﹝', u'｢', u'﹁'],
          [u'】', u'』', u'〕', u'﹞', u'｣', u'﹂'],
          [u'（', u'「'],
          [u'）', u'」'],
          [u'﹖'],
          [u'︰', u'﹕'],
          [u'・', u'．', u'·', u'‧', u'°'],
          [u'●', u'○', u'▲', u'◎', u'◇', u'■', u'□', u'※', u'◆'],
          [u'〜', u'～', u'∼'],
          [u'︱', u'│', u'┼', u''],
          [u'╱'],
          [u'╲'],
          [u'—', u'ー', u'―', u'‐', u'−', u'─', u'﹣', u'–', u'ㄧ']]
  epun = [u' ', u'！', u'"', u'"', u'\'', u';', u'<', u'>', u'、', u'[', u']', u'(', u')', u'？', u'：', u'･', u'•', u'~', u'|', u'/', u'\\', u'-']
  repls = {}

  for i in range(len(cpun)):
    for j in range(len(cpun[i])):
      repls[cpun[i][j]] = epun[i]

  return replace_all(repls, text)

def Q2B(uchar):
  """全角转半角"""
  inside_code = ord(uchar)
  if inside_code == 0x3000:
    inside_code = 0x0020
  else:
    inside_code -= 0xfee0
  #转完之后不是半角字符返回原来的字符
  if inside_code < 0x0020 or inside_code > 0x7e:
    return uchar
  return chr(inside_code)

class TextLoader(object):
  def __init__(self, data_dir, batch_size, chars=[]):
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.seq_length = 0
    self.input_files = glob(data_dir + '/*.txt')
    self.vocabs = {}
    self.chars = chars
    self.seq_lengths = []

    vocab_file = os.path.join(data_dir, "vocab.pkl")
    data_file = os.path.join(data_dir, "data.pkl")

    if os.path.exists(data_file):
      print("[TextLoader] Load saved data...")
      with open(data_file, 'rb') as f:
        self.data, self.seq_lengths, my_chars = cPickle.load(f)
        self.seq_length = max(self.seq_lengths)
        if my_chars is not None and not len(self.chars) > 0:
          self.chars = my_chars
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.vocab_size = len(self.chars)
    else:
      print("[TextLoader] Reading text file...")
      self.preprocess(self.input_files, data_file, vocab_file)

    print("[TextLoader] Processing...")
    self.create_batches()
    self.reset_batch_pointer()

  def preprocess(self, input_files, data_file, vocab_file):
    sents = []
    seq_length = 0

    for input_file in input_files:
      with codecs.open(input_file, "r", "utf-8") as f:
        lines = normalize_unicodes(f.read()).split("\n")

        for line in lines:
          if len(line) == 0:
            continue
          seq_length = max(seq_length, len(line))
          sents.append(line)

    if not len(self.chars):
      # Compose vocab
      lines = "".join(sents)
      counter = collections.Counter(lines)
      count_pairs = sorted(counter.items(), key=lambda x: -x[1])
      self.chars, _ = list(zip(*count_pairs))
      self.chars = START_VOCAB + list(self.chars)

    self.vocab = dict(zip(self.chars, range(len(self.chars))))
    self.vocab_size = len(self.chars)
    self.seq_length = seq_length + 1 # for additional symbols GO, EOS
    self.data = np.zeros((len(sents), self.seq_length), dtype=np.int32)

    # Convert text to one-hot representation
    for i, sent in enumerate(sents):
      vec, vec_len = self.parse_input(sent)
      self.seq_lengths.append(vec_len)
      self.data[i] = vec

    # Export vocab and data
    with open(vocab_file, "wb") as f:
      cPickle.dump(self.chars, f)
    with open(data_file, "wb") as f:
      cPickle.dump((self.data, self.seq_lengths, self.chars), f)

  def parse_input(self, inputs):
    eos_index = START_VOCAB.index(EOS)
    vec = np.array([self.vocab.get(char, UNK_ID) for char in list(inputs)])
    vec_len = vec.size + 1 # for additional symbols EOS
    # Padding to seq_length
    vec = np.lib.pad(vec, (0, self.seq_length - vec.size), 'constant')
    vec[vec_len - 1] = eos_index

    return vec, vec_len

  def compose_output(self, output):
    res = ""

    for o in output:
      if o == 