import json
import os
from collections import namedtuple
import zipfile

import torch
from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets import VisionDataset
from PIL import Image
import pandas as pd


class Deepfashion(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    DeepfashionClass = namedtuple('DeepfashionClass', ['name', 'id', 'category_id', 'color'])
    DeepfashionCategory = namedtuple('DeepfashionCategory', ['category_name', 'category_id', 'color'])    

    classes = [
        DeepfashionClass('background', 0, 7, (0,0,0)),
        DeepfashionClass('hat', 1, 0, (128,0,0)),
        DeepfashionClass('hair', 2, 0, (255,0,0)),
        DeepfashionClass('glove', 3, 3, (0,85,0)),
        DeepfashionClass('sunglasses', 4, 1, (170,0,51)),
        DeepfashionClass('upperclothes', 5, 3, (255,85,0)),
        DeepfashionClass('dress', 6, 3, (0,0,85)),
        DeepfashionClass('coat', 7, 3, (0,119,221)),
        DeepfashionClass('socks', 8, 5, (85,85,0)),
        DeepfashionClass('pants', 9, 4, (0,85,85)),
        DeepfashionClass('tosor-skin', 10, 2, (85,51,0)),
        DeepfashionClass('scarf', 11, 3, (52,81,128)),
        DeepfashionClass('skirt', 12, 4, (0,128,0)),
        DeepfashionClass('face', 13, 1, (0,0,255)),
        DeepfashionClass('leftArm', 14, 2, (51,170,221)),
        DeepfashionClass('rightArm', 15, 2, (0,255,255)),
        DeepfashionClass('leftLeg', 16, 2, (85,255,170)),
        DeepfashionClass('rightLeg', 17, 2, (170,255,85)),
        DeepfashionClass('leftShoe', 18, 6, (255,255,0)),
        DeepfashionClass('rightShoe', 19, 6, (255,170,0))
    ]

    eclasses = [
        ('hair', 0, (0, 128, 0)),
        ('face', 1, (128, 128, 0)),
        ('skin', 2, (0, 128, 128)),
        ('top-clothes', 3, (0, 0, 128)),
        ('botom-clothes', 4, (128, 0, 128)),
        ('socks', 5, (128, 128, 128)),
        ('shoes', 6, (128, 0, 0)),
        ('background', 7, (0,0,0))
    ]

    def __init__(self, root, split='train', transform=None, target_transform=None, transforms=None):
        super(Deepfashion, self).__init__(root, transforms, transform, target_transform)
        self.images_dir = os.path.join(self.root, 'img')
        self.targets_dir = os.path.join(self.root, 'lbl')
        self.split = split
        self.images = []
        self.targets = []

        valid_modes = ("train", "test", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            image_dir_zip = os.path.join(self.root, '{}'.format('img.zip'))
            target_dir_zip = os.path.join(self.root, '{}'.format('lbl.zip'))

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')

        data_list = pd.read_csv(os.path.join(self.root, 'list_eval_partition.txt'), sep='\t', skiprows=1)
        data_list = data_list[data_list['evaluation_status'] == self.split]
        for image_path in data_list['image_name']:
            target_path = 'lbl/' + '/'.join(image_path.split('/')[1:])
            self.images.append(image_path)
            self.targets.append(target_path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index][i])
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        target_ = torch.zeros((len(eclasses), target.size()[1], target.size()[2]), device=target.device)
        for cls in classes:
            target_[cls.category_id] = target_[cas.category_id] | target[cls.id]
        return image, target

    def __len__(self):
        return len(self.images)
