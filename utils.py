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
          [u'—', u'ー', u