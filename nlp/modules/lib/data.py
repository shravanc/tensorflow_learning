class LoadData():
  def __init__(self, options={}):
    self.type = options["type"]
    if "text" in options:
      self.text = options["text"]
    self.data = None

  def load(self):
    if self.type == "imdb_reviews":
      import tensorflow_datasets as tfds
      imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
      self.data = imdb
    elif self.type == "string":
      self.data = [self.text]

    return self

  
  def get_type(self):
    return self.type

  def get_data(self):
    return self.data

