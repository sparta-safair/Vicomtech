"""
Helper function for loading all the attacks defined in `attacks/attack_types folder`
"""
import importlib
import os
import pkgutil
import sys

from attacks.base import Attack
from environment_setup import PROJECT_ROOT_DIR
import config

def load_all_modules_from_dir(dirname):
    """
    Loads all the attack modules in the current run
    :param dirname: base directory to search from
    :return: None
    """
    for root_dirname, module_name, ispkg in pkgutil.iter_modules([dirname]):
        relative_module_name = f'attacks.attack_types.{module_name}'
        importlib.import_module(relative_module_name, PROJECT_ROOT_DIR)


def get_attacks(defense_type):
    """
    Create a list of all the attack instances
    :param defense_type: reid/attr attacks
    :return: list of all attack instances
    """
    # First load all modules in the
    attack_base_dir = os.path.join(PROJECT_ROOT_DIR, 'attacks', 'attack_types')
    load_all_modules_from_dir(dirname=attack_base_dir)

    attack_subclasses = Attack.__subclasses__()
    attack_list = []
    for subclass_name in attack_subclasses:
        subclass = subclass_name(defense_type=defense_type)
        if subclass.attack_description() not in config.skip_list:
            attack_list.append(subclass)
    return attack_list


if __name__ == '__main__':
    print(get_attacks(task_type='attr'))
