import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TrainData():

  def __init__(self, options):
    self.processed_data_obj = options["data"]
    self.model    = None
    self.history  = None

  def train(self):

    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.processed_data_obj.get_vocab_size(), self.processed_data_obj.get_embed_dim(), input_length=self.processed_data_obj.get_max_len()),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    self.history = model.fit( self.processed_data_obj.get_train_data(), 
              self.processed_data_obj.get_train_labels(),
              epochs=self.processed_data_obj.get_num_epochs(),
              validation_data=(self.processed_data_obj.get_test_data(), self.processed_data_obj.get_test_labels())
    )
  
    self.model = model
    return self

  def get_history(self):
    return self.history

  def get_model(self):
    return self.model


