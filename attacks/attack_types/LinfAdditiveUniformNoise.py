from attacks.base import Attack
from foolbox.attacks import LinfAdditiveUniformNoiseAttack


class AdditiveUniformNoiseAttack(Attack):
    """
    Creates an LinfAdditiveUniformNoise attack
    """

    def __init__(self, defense_type):
        """
        :param task_type: attr/reid
        """
        super(AdditiveUniformNoiseAttack, self).__init__()
        self.defense_type = defense_type

    def instantiate_attack(self):
        """
        Create an instance of the Foolbox LinfAdditiveUniformNoise attack
        :return: foolbox attack instance
        """
        self.attack = LinfAdditiveUniformNoiseAttack()
        return self.attack

    def attack_description(self):
        """
        String description for the attack. Return `self.attack` for more details. Edit the function as per use case
        :return: string
        """
        # return str(self.attack)
        return "additive_noise"


if __name__ == '__main__':
    attack = AdditiveUniformNoiseAttack('reid')
    print(dir(attack))
