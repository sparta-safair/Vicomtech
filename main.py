# Base class following the Template Design patter
import argparse
import json
import os
import urllib.request

from dataset.BreastCancerDataset import get_data_loader
from environment_setup import PROJECT_ROOT_DIR
from wrapper.tensorflow_to_foolbox import TensorFlowToFool
from tqdm import tqdm
from foolbox.criteria import Misclassification
import tensorflow as tf
from attacks.attack_list import get_attacks

FIXED_EPSILON = 8. / 255


class AttackExecuter:
    def __init__(self):
        pass
    
    def download_url(self, url, save_path):
        with urllib.request.urlopen(url) as dl_file:
            with open(save_path, 'wb') as out_file:
                out_file.write(dl_file.read())

    def compute_original_accuracy(self, device, dataloader, foolbox_model, target_label):
        """
        Computes the accuracy of model before any perturbations
        :param device: gpu/cpu
        :param dataloader: dataloader for the task
        :param foolbox_model: model wrapped in foolbox framework
        :param target_label: The label to be used in targeted attack. If None -> Untargeted. Default: None
        :return: accuracy value
        """
        # Compute the initial accuracy
        total = 0
        correct = 0
        for images, labels in tqdm(dataloader.as_numpy_iterator()):
            total += images.shape[0]
            correct += dataloader.accuracy(foolbox_model, images, labels)
        acc = correct / total
        return acc

    def load_model(self, checkpoint_dir, defense_type, model_numer=8):
        """
        Loads a pretrained model.
        :param checkpoint_dir: folder to check model weights from
        :param defense_type: AdversarialTrain/DimensionalityReduction/PredictionSimilarity/ActivationsDetector
        :param model_numer: model identifier. We give a base name to model and identify different instances using `model_number`.(Similar to SAFAIR Contest nomenclature)
        :return: (model, bounds) model and bound of values in input
        """
        # Load the pretrained model
        model, bounds = self.select_model_type(defense_type=defense_type, checkpoint_dir=checkpoint_dir)
        #model.load(model_number=model_numer, checkpoint_dir=checkpoint_dir)
        #model.eval()
        return model, bounds

    def select_model_type(self, defense_type, checkpoint_dir):
        """
        Helper function for loading model weights. Please change this function when defining new networks accordingly.
        :param defense_type: AdversarialTrain/DimensionalityReduction/PredictionSimilarity/ActivationsDetector
        :param checkpoint_dir: folder to check model weights from
        :return: (model, bounds) model and bound of values in input
        """
        BASE_PATH = os.path.join(os.getcwd(), "saved_models")
        if defense_type == 'AdversarialTrain':
            MODEL_PATH = os.path.join(BASE_PATH, 'adversarial_training_model.h5')
            self.download_url("https://vicomtech.box.com/shared/static/uomnz65vu1y71j538j1yr15nizo6t4mj.h5",MODEL_PATH)
            defended_model = tf.keras.models.load_model('saved_models/adversarial_training_model.h5')
            bounds = (0, 255)
        elif defense_type == 'DimentionalityReduction':
            where = str(input('Where would you like the Autoencoder:\n1.At the begining (top)\n2.In between (middle)\nSelect the defense: '))
            while where not in ['top','middle']:
                print('You must introduce top of middle')
                where = str(input('Where would you like the Autoencoder:\n1.At the begining (top)\n2.In between (middle)\nSelect the defense: '))
            MODEL_PATH = os.path.join(BASE_PATH, str(where)+'_dimensionality_reduction_model.h5')
            if where == 'top':
                self.download_url("https://vicomtech.box.com/shared/static/bi3yatnfdyli57e4609xx5qkgjxd6p48.h5",MODEL_PATH)
            else:
                self.download_url('https://vicomtech.box.com/shared/static/zpixxqhcgns2dcyglaafhuryeig5v0tn.h5',MODEL_PATH)
            defended_model = tf.keras.models.load_model('saved_models/'+str(where)+'_dimensionality_reduction_model.h5')
            bounds = (0, 255)
        return defended_model, bounds

    def convert_model(self, model, bounds):
        """
        Function to help in conversion of model to Foolbox.
        :param model: model defined in PyTorch (can be extended to Tensorflow in future)
        :return: foolbox model
        """
        # Initialize the converter
        converter = TensorFlowToFool(model=model, bounds=bounds)
        fmodel = converter()
        return fmodel

    def create_attack(self, defense_type):
        """
        Function to load all the attacks defined.
        :param defense_type: AdversarialTrain/DimensionalityReduction/PredictionSimilarity/ActivationsDetector
        :return: a list of attacks as defined in `attaks/attack_types` folder
        """
        return get_attacks(defense_type=defense_type)

    def compute_adv_acc(self, dataloader, device, task_criterion, foolbox_model, attack, target_label=None):
        """
        Computes the adversarial accuracy of model
        :param dataloader: dataloader for the task
        :param device: gpu/cpu
        :param task_criterion: Misclassification/MultiLabelMisclassification
        :param foolbox_model: model wrapped in foolbox framework
        :param attack: Specific attack defined using foolbox framework
        :param target_label: The label to be used in targeted attack. If None -> Untargeted. Default: None
        :return: adversarial accuracy of the model
        """
        # Now run it for the new set of input
        total = 0
        adv_correct = 0
        for images, labels in tqdm(dataloader.as_numpy_iterator()):
            total += images.shape[0]
            if target_label is not None:
                labels = torch.ones_like(labels) * int(target_label)
            criterion = task_criterion(tf.cast(tf.argmax(labels, axis = -1), tf.int32))
            raw, clipped, is_adv = attack(foolbox_model, tf.cast(images, tf.float32), criterion=criterion, epsilons=FIXED_EPSILON)
            adv_correct += dataloader.accuracy(foolbox_model, raw, labels)
        adv_acc = adv_correct / total
        return adv_acc

    def create_dataloader(self, defense_type, batch_size):
        """
        Creates the dataloader.
        :param defense_type: reid/attr
        :param batch_size: size of batch used for loading the values
        :return: dataloader
        """
        return get_data_loader(batch_size=batch_size, split="test")

    def execute_attack(self, args):
        """
        The main execution method
        :param args: argparse object
        :return: None
        """
        json_data = {}
        dataloader = self.create_dataloader(defense_type=args.defense_type, batch_size=args.batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = os.path.join(PROJECT_ROOT_DIR, args.checkpoint_dir)
        model, bounds = self.load_model(checkpoint_dir=checkpoint_dir, defense_type=args.defense_type,
                                        model_numer=args.model_number)
        foolbox_model = self.convert_model(model=model, bounds=bounds)
        initial_acc = self.compute_original_accuracy(device=device, dataloader=dataloader, foolbox_model=foolbox_model,
                                                     target_label=args.target_label)
        json_data['orig_acc'] = initial_acc
        attack_class_list = self.create_attack(defense_type=args.defense_type)

        for attack_class in attack_class_list:
            try:
                task_criterion = Misclassification
                attack = attack_class.instantiate_attack()
                adv_acc = self.compute_adv_acc(dataloader=dataloader, device=device, task_criterion=task_criterion,
                                               foolbox_model=foolbox_model, attack=attack,
                                               target_label=args.target_label)
                json_data[attack_class.attack_description()] = adv_acc
            except TypeError as a:
                print(a)
            except ValueError as v:
                print(v)
                print(f"Targeted Attack not defined for {attack_class.attack_description()}. Skipping!!")
        print(json.dumps(json_data, indent=4))
        
def parse_args():
    """parse input arguments
    :return: args
    """

    parser = argparse.ArgumentParser(description='adversarial ml')
    parser.add_argument('--batch_size', help='Batch size. Default: 32', type=int, default='32')
    parser.add_argument('--defense_type', help='reid/attr/classification. Default:Original', type=str, default='DimentionalityReduction')
    parser.add_argument('--checkpoint_dir', help='Directory to load model from. Default: saved_models/', type=str,
                        default='C:/Users/xetxeberria/Desktop/PythonScript/Machine_learning_Attack&Defenses/Defensas')
    parser.add_argument('--model_number',
                        help='Model to test final results on. Needed for test and adv mode. Default: 0', type=int,
                        default='0')
    parser.add_argument('--target_label', help='Target label to use in targeted attack. An integer value. Default:None',
                        type=str, default=None)
    # Now parse all the values
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.target_label is not None:
        print(f"Using TARGETED attack with target label {args.target_label}")
        assert args.defense_type != 'attr', "Targeted Attack only defined for Re-identification task"
    else:
        print("Using UN-TARGETED attack")
    executor = AttackExecuter()
    dataloader = executor.execute_attack(args)
