import os

from data import common
from data import ImageData

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class BenchmarkDenoise(ImageData.ImageData):
    def __init__(self, args, name = '', train=True):
        super(BenchmarkDenoise, self).__init__(args, name = name, train=train, benchmark=True)

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'BenchmarkDenoise', self.args.data_test)

    def _name_tarbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_tar.npy'.format(self.split)
        )

    def _name_inputbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_x{}_sigma{}.npy'.format(self.split, self.scale, self.sigma)
        )
