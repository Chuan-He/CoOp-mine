import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
import time
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

import clip

import converter_domainbed


@DATASET_REGISTRY.register()
class office_home(DatasetBase):

    dataset_dir = "OfficeHome"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, ".")
        #self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")
        train_datasets, val_datasets, test_datasets, class_names = \
            converter_domainbed.get_domainbed_datasets(dataset_name="OfficeHome", root="datasets/domainbed/data", targets=[0], holdout=0.2)
        super().__init__(train_x=train_datasets, val=val_datasets, test=test_datasets)

