from attacks.base import Attack
from foolbox.attacks import LinfBasicIterativeAttack


class LinfBasicItervativeAttack(Attack):
    """
    Creates an Linf Basic Iterative Attack
    """
    def __init__(self, defense_type):
        """
        :param task_type: attr/reid
        """
        super(LinfBasicItervativeAttack, self).__init__()
        self.defense_type = defense_type

    def instantiate_attack(self):
        """
        Create an instance of the Foolbox LinfBasicIterative attack
        :return: foolbox attack instance
        """
        self.attack = LinfBasicIterativeAttack()
        # monkey patching
        self.attack.get_loss_fn = self.get_use_case_loss_fn
        
        if self.defense_type == 'PredictionSimilarity' or self.defense_type == 'ActivationsDetector':
            
            self.attack.get_loss_fn = self.get_special_use_case_loss_fn
            #self.attack.value_and_grad = self.value_and_grad
        return self.attack

    def attack_description(self):
        """
        String description for the attack. Return `self.attack` for more details. Edit the function as per use case
        :return: string
        """
        # return str(self.attack)
        return "bim"

