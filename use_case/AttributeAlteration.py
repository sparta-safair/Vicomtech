import torch
from foolbox.criteria import Misclassification
import eagerpy as ep

import config


class MultiLabelMisclassification(Misclassification):
    """Considers those perturbed inputs adversarial whose predicted class
    differs from the label.
    Args:
        labels: Tensor with labels of the unperturbed inputs ``(batch,)``.
    """

    def __init__(self, labels):
        """
        :param labels: Labels for the unperturbed instances
        :param threshold: number of attribute changes to define misclassification. Default: 1
        """
        super().__init__(labels)
        self.labels: ep.Tensor = ep.astensor(labels)

    def __repr__(self) -> str:
        return f"custom misclassification"

    def __call__(self, perturbed, outputs):
        """
        One can define a threshold value to indicate change of how many attributes indicates a misclassification. Default is 1
        :param perturbed: perturbed samples
        :param outputs: model output
        :return:
        """
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed, outputs

        classes = outputs_ >= 0
        assert classes.shape == self.labels.shape
        is_adv = (classes != self.labels).sum(axis=1) >= config.ATTR_CHANGE_THRESHOLD
        return restore_type(is_adv)


class AttributeAlterationCriteria:

    @staticmethod
    def get_criterion():
        return MultiLabelMisclassification
