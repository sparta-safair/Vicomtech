"""
This file is going to act as an abstraction for the celebA dataset and load samples from the dataset
We are using the general pytorch framework for the purpose
"""

from tensorflow.data import Dataset, Iterator
import torchvision
from torchvision.transforms import transforms
import os
import eagerpy as ep
import numpy as np
from environment_setup import PROJECT_ROOT_DIR
import tensorflow as tf
import urllib.request



class BreastCancer():
    def __init__(self, n_sample = None, root=os.path.join(PROJECT_ROOT_DIR, "data")):
        """
        Generate dataset object. Please set `self.min_val` and `self.max_val` .
        It would be needed for Adversarial perturbation generation.
        :param split: (string) train/valid/test
        :param transform: Transforms to be applied on model. If None, resizes image to
                (256, 256) and scales pixels to [0, 1]
        :param dataset_max_val: maximum_value of input images
        :param dataset_min_val: minimum_value of input images
        """
        self.root = root
        self.n_sample = n_sample
        self.dataset = Dataset.from_tensor_slices(self.load())
        # Range of values for the input
    
    def load(self):
        SHARED_X_TEST = "https://vicomtech.box.com/shared/static/ye3e8xioe01bt62psedjemozswtw8qga.npy"
        SHARED_Y_TEST = "https://vicomtech.box.com/shared/static/z63n3kwixb5sstozfoliuvn548ecp2es.npy"

        BASE_PATH = os.path.join(os.path.dirname(__file__), "data_download")
        if not os.path.exists(BASE_PATH):
                os.makedirs(BASE_PATH)
        X_TEST_PATH = os.path.join(BASE_PATH, "X.npy")
        Y_TEST_PATH = os.path.join(BASE_PATH, "y.npy")

        urllib.request.urlretrieve(
            SHARED_X_TEST, # que url
            X_TEST_PATH # donde guardar
        )
        urllib.request.urlretrieve(SHARED_Y_TEST, Y_TEST_PATH)

        X_test = np.load(X_TEST_PATH)
        Y_test = np.load(Y_TEST_PATH)
        
        if self.n_sample == None or self.n_sample>=len(X_test):
            return X_test, Y_test
        
        return X_test[:self.n_sample], Y_test[:self.n_sample]

    def batch(self, batch_size):
        self.dataset = self.dataset.batch(batch_size)

        
    def shuffle(self, size):
        self.dataset = self.dataset.shuffle(size)

    def take(self, num):
        return self.dataset.take(num)
    def as_numpy_iterator(self):
        return self.dataset.as_numpy_iterator()

    @staticmethod
    def accuracy(fmodel, inputs, labels):
        inputs_, labels_ = ep.astensors(inputs,labels)
        del inputs, labels
        predictions = fmodel(inputs_) 
        predictions = (predictions >= 0.5).float32()
        accuracy = (predictions * labels_).float32().sum()
        return accuracy.item()
    
    def __len__(self):
        return len(self.dataset)


def get_data_loader(batch_size, split, transform=None, shuffle=False, num_workers=4, dataset_min_val=0, dataset_max_val=1):
    """
    Return the dataloader object. Shared method for all train/val/test splits
    :param dataset_max_val: maximum_value of input images
    :param dataset_min_val: minimum_value of input images
    :param batch_size: int
    :param split: train/valid/test
    :param transform: Transform object. Default: None
    :param shuffle: boolean. Default: False
    :param num_workers: int. number of cores. Default 4
    :return: Dataloader object
    """
    dataset = BreastCancer(n_sample = 100)
    dataset.batch(batch_size)
    dataset.shuffle(len(dataset))
    return dataset


if __name__ == '__main__':
    dataloader = get_data_loader(4, 'valid')
    dataset = dataloader.dataset
    for img_name, img, label in dataloader:
        print(img_name)
        print(img.shape)
        print(img.max())
        print(img.min())
        print(label.shape)
        # print(dataloader.dataset.pred_acc)
        break
