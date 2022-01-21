"""
Class responsible for converting PyTorch models into Foolbox
"""
import foolbox as fb


class TensorFlowToFool(object):
    def __init__(self, model, bounds=None):
        """
        :param model: The Tensorflow/Keras model
        :param bounds: Range of values expected in the input, default (0, 1)
        """
        self.model = model
        if bounds is None:
            bounds = (0, 1)
        self.bounds = bounds

    def __call__(self, *args, **kwargs):
        fmodel = fb.TensorFlowModel(self.model, bounds=self.bounds)
        return fmodel
