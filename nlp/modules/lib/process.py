from helpers.hyper_parameters import HyperParameter

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ProcessData():
  def __init__(self, options):
    self.data_obj     = options["data"]
    self.train_data   = []
    self.train_labels = []
    self.test_data    = []
    self.test_labels  = []
    self.tr_data      = []
    self.te_data      = []
    self.hyper_param  = HyperParameter()
    
    if "tokenizer" in options:
      self.tokenizer    = options["tokenizer"]
    else:
      self.tokenizer  = None 
    print(self.data_obj)

  def process(self):
    d_type = self.data_obj.get_type()
    if d_type == "imdb_reviews":
      self.split_data(d_type)
      self.preprocess(d_type)
      return self
    elif d_type == "string":
      data = self.data_obj.get_data()
      data = self.tokenizer.texts_to_sequences(data)
      data = pad_sequences(data, maxlen=self.hyper_param.max_len, truncating=self.hyper_param.trunc_type)
      return data
      

  def split_data(self, d_type):
    if d_type == "imdb_reviews":
      data = self.data_obj.get_data()
      for s,l in data["train"]:
        self.tr_data.append(str(s.numpy()))
        self.train_labels.append(l.numpy())

      for s,l in data["test"]:
        self.te_data.append(str(s.numpy()))
        self.test_labels.append(l.numpy())

      import numpy as np
      self.train_labels = np.array(self.train_labels)
      self.test_labels  = np.array(self.test_labels)
      

  def preprocess(self, d_type):
    if d_type == "imdb_reviews":
     
      tokenizer = Tokenizer(num_words=self.hyper_param.vocab_size, oov_token=self.hyper_param.oov_tok)
      tokenizer.fit_on_texts(self.tr_data)
     
      tr_sequences    = tokenizer.texts_to_sequences(self.tr_data)
      self.train_data = pad_sequences(tr_sequences, maxlen=self.hyper_param.max_len, truncating=self.hyper_param.trunc_type)
      
      te_sequences    = tokenizer.texts_to_sequences(self.te_data)
      self.test_data  = pad_sequences(te_sequences, maxlen=self.hyper_param.max_len, truncating=self.hyper_param.trunc_type)

      self.tokenizer = tokenizer

  def get_train_data(self):
    return self.train_data

  def get_train_labels(self):
    return self.train_labels

  def get_test_data(self):
    return self.test_data
  
  def get_test_labels(self):
    return self.test_labels

  def get_vocab_size(self):
    return self.hyper_param.vocab_size

  def get_max_len(self):
    return self.hyper_param.max_len

  def get_embed_dim(self):
    return self.hyper_param.embed_dim

  def get_num_epochs(self): 
    return self.hyper_param.num_epochs 
