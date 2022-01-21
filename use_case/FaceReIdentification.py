import torch
from foolbox.criteria import Misclassification, TargetedMisclassification
import eagerpy as ep


class FaceReIdentificationUntargetedCriterion:
    """
    Class specifically designed for evaluation of Face ReIdentification task. In principle, it is similar to the Cross Entropy
    loss and hence we just return the class from FoolBox.
    """

    @staticmethod
    def get_criterion():
        return Misclassification


class FaceReIdentificationTargetedCriterion:
    """
    Class specifically designed for evaluation of Face ReIdentification task. In principle, it is similar to the Cross Entropy
    loss and hence we just return the class from FoolBox.
    """

    @staticmethod
    def get_criterion():
        return TargetedMisclassification
