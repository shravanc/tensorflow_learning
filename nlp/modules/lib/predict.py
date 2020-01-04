class PredictData():

  def __init__(self, options):
    self.model_obj  = options["model"]
    self.data       = options["data"]
    self.predictions= None

  def predict(self):
    model = self.model_obj.get_model()
    print("Prediction Begins**************", self.data)
    self.predictions = model.predict_classes(self.data)
    return self.predictions


