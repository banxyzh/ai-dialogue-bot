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
      with ope