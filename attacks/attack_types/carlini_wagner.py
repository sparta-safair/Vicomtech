from attacks.base import Attack
from foolbox.attacks import L2CarliniWagnerAttack


class CarliniWagnerL2Attack(Attack):
    """
    Creates an CarliniWagnerL2A attack
    """
    def __init__(self, defense_type):
        """
        :param task_type: attr/reid
        """
        super(CarliniWagnerL2Attack, self).__init__()
        self.defense_type = defense_type

    def instantiate_attack(self):
        """
        Create an instance of the Foolbox L2CarliniWagner attack
        :return: foolbox attack instance
        """
        self.attack = L2CarliniWagnerAttack()
        
        if self.defense_type == 'PredictionSimilarity' or self.defense_type == 'ActivationsDetector':
            
            self.binary_search_steps = self.attack.binary_search_steps
            self.steps = self.attack.steps
            self.stepsize = self.attack.stepsize
            self.confidence = self.attack.confidence
            self.initial_const = self.attack.initial_const
            self.abort_early = self.attack.abort_early
            self.attack.run = self.carlini_wagner_run
            
        return self.attack

    def attack_description(self):
        """
        String description for the attack. Return `self.attack` for more details. Edit the function as per use case
        :return: string
        """
        # return str(self.attack)
        return "carlini_wagner"
