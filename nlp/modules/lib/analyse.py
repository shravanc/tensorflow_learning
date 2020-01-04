import matplotlib.pyplot as plt
class AnalyseData():
  def __init__(self, options):
    self.model_obj = options["model"]
    self.metrics   = options["type"]

  def analyse(self):
    for string in self.metrics:
      history = self.model_obj.get_history()
      plt.plot(history.history[string])
      plt.plot(history.history['val_'+string])
      plt.xlabel("Epochs")
      plt.ylabel(string)
      plt.legend([string, 'val_'+string])
      plt.show()
