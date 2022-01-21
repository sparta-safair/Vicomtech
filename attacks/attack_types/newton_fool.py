from attacks.base import Attack
from foolbox.attacks import NewtonFoolAttack


class NewtonAttack(Attack):
    """
    Creates an LinfAdditiveUniformNoise attack
    """

    def __init__(self, defense_type):
        """
        :param task_type: attr/reid
        """
        super(NewtonAttack, self).__init__()
        self.defense_type = defense_type

    def instantiate_attack(self):
        """
        Create an instance of the Foolbox LinfAdditiveUniformNoise attack
        :return: foolbox attack instance
        """
        self.attack = NewtonFoolAttack()
            
        if self.defense_type == 'PredictionSimilarity' or self.defense_type == 'ActivationsDetector':
            
            self.steps = self.attack.steps
            self.stepsize = self.attack.stepsize
            self.attack.run = self.special_run
            
        return self.attack

    def attack_description(self):
        """
        String description for the attack. Return `self.attack` for more details. Edit the function as per use case
        :return: string
        """
        # return str(self.attack)
        return "newton"
