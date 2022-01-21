from attacks.base import Attack
import foolbox as fb

fixed_epsilon = 8./255


class LinfDeepFoolAttack(Attack):
    """
    Creates an LinfDeepFool attack
    """
    def __init__(self, defense_type):
        """
        :param task_type: attr/reid
        """
        super(LinfDeepFoolAttack, self).__init__()
        self.defense_type = defense_type

    def instantiate_attack(self):
        """
        Create an instance of the Foolbox LinfDeepFool attack
        :return: foolbox attack instance
        """
        self.attack = fb.attacks.LinfDeepFoolAttack()
        
        if self.defense_type == 'PredictionSimilarity' or self.defense_type == 'ActivationsDetector':
            
            self.attack._get_loss_fn = self.get_special_deepfool_use_case_loss_fn
            #self.attack.value_and_grad = self.value_and_grad
            
        return self.attack

    def attack_description(self):
        """
        String description for the attack. Return `self.attack` for more details. Edit the function as per use case
        :return: string
        """
        # return str(self.attack)
        return "deep_fool"


if __name__ == '__main__':
    attack = LinfDeepFoolAttack('attr')
    print(attack)