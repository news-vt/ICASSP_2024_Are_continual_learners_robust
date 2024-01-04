# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from backbone.MNISTMLP import MNISTMLP

from datasets.perm_mnist import store_mnist_loaders
from datasets.transforms.rotation import Rotation
from datasets.utils.continual_dataset import ContinualDataset


class RotatedMNIST(ContinualDataset):
    NAME = 'rot-mnist'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 7
    MIN_DEGREES = 0
    MAX_DEGREES = 150
    # def get_data_loaders(self):
    #     transform = transforms.Compose((Rotation(), transforms.ToTensor()))
    #     train, test = store_mnist_loaders(transform, self) #it return train and test loader roated with specified angles and also appends the curretn test loader to self.test_loaders
    #     return train, test

    def get_data_loaders(self, heldout_index):
        transform = transforms.Compose((Rotation(), transforms.ToTensor()))
        train, heldout_index, heldout_angle = store_mnist_loaders(RotatedMNIST.N_TASKS, 
                                          RotatedMNIST.MIN_DEGREES, 
                                          RotatedMNIST.MAX_DEGREES, heldout_index,self) #it return train and test loader roated with specified angles and also appends the curretn test loader to self.test_loaders
        return train, heldout_index, heldout_angle

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, RotatedMNIST.N_CLASSES_PER_TASK) #28*28 is the input-size and N_classes_per_task is the output size

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        # return StepLR(model.opt, step_size=1, gamma=0.99999)
        return None

    @staticmethod
    def get_batch_size() -> int:
        return 128

    @staticmethod
    def get_minibatch_size() -> int:
        return RotatedMNIST.get_batch_size()
