from attacks.base import Attack
from foolbox.attacks import LinfFastGradientAttack


class FGSMAttack(Attack):
    """
    Create a FGSM attack instance
    """
    def __init__(self, defense_type):
        """
        :param task_type: reid/attr
        """
        super(FGSMAttack, self).__init__()
        self.defense_type = defense_type

    def instantiate_attack(self):
        """
        Create an instance of the Foolbox FGSM attack
        :return: foolbox attack instance
        """
        self.attack = LinfFastGradientAttack()
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
        return "fgsm"
